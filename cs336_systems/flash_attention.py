from __future__ import annotations

import math

import torch
from einops import reduce, einsum

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is only required on CUDA hosts.
    triton = None
    tl = None


MIN_TILE_SIZE = 16


def _validate_flash_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected q, k, and v to have shape (batch, seq_len, d).")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Expected q, k, and v to have the same batch size.")
    if k.shape != v.shape:
        raise ValueError("Expected k and v to have the same shape.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("Expected q, k, and v to share the same head dimension.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Expected q, k, and v to share the same dtype.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Expected q, k, and v to be on the same device.")


def _choose_query_tile_size(n_queries: int) -> int:
    # Handout guidance: choose tiles at least 16x16 for the tiled implementation.
    return min(n_queries, MIN_TILE_SIZE)


def _choose_key_tile_size(n_keys: int) -> int:
    return min(n_keys, MIN_TILE_SIZE)


# Forward helpers
def flash_attention_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Non-tiled reference implementation for debugging the tiled forward pass.

    This is intentionally the direct formulation from Eqs. (4)-(6) and (12):
    - S = Q K^T / sqrt(d)
    - O = softmax(S) V
    - L = logsumexp(S)
    """
    _validate_flash_attention_inputs(q, k, v)

    d = q.shape[-1]
    scale = d ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        query_positions = torch.arange(n_queries, device=q.device)
        key_positions = torch.arange(n_keys, device=q.device)
        causal_mask = query_positions[:, None] >= key_positions[None, :]
        scores = torch.where(causal_mask.unsqueeze(0), scores, scores.new_full((), -1e6))

    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v)
    logsumexp = torch.logsumexp(scores, dim=-1)
    return output, logsumexp


if triton is not None:
    @triton.jit
    def flash_attention_forward_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # TODO(student): follow handout Algorithm 1 in Triton.
        # Suggested order:
        # 1. Build block pointers for the current batch/query tile of Q, O, and L.
        # 2. Initialize on-chip running buffers for O_i, l_i, and m_i in tl.float32.
        # 3. Loop over key/value tiles, update online softmax state, and advance pointers.
        # 4. Normalize the output tile, compute L_i, and store both back to global memory.
        # 5. When implementing part (c), use is_causal to mask out future keys.

        # Program Indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            base=l_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq, ),
            offsets=(query_tile_index * Q_TILE_SIZE, ),
            block_shape=(Q_TILE_SIZE, ),
            order=(0, ),
        )
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_options="zero")
        O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        running_l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        running_m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_options="zero")
            V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_options="zero")

            S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale
            prev_running_m = running_m
            running_m = tl.maximum(running_m, tl.max(S_tile, 1))
            P_tile = tl.exp(S_tile - running_m[:, None])
            running_l = tl.exp(prev_running_m - running_m) * running_l + tl.sum(P_tile, 1)
            O_tile = O_tile * tl.exp(prev_running_m - running_m)[:, None] + tl.dot(P_tile, V_tile)

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        O_tile = O_tile / running_l[:, None]
        L_tile = running_m + tl.log(running_l)
        tl.store(O_block_ptr, O_tile, boundary_check=(0, 1))
        tl.store(L_block_ptr, L_tile, boundary_check=(0, 1))

else:
    flash_attention_forward_kernel = None


def _flash_attention_forward_pytorch_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_tile_size: int,
    k_tile_size: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Skeleton for Assignment 2, Section 1.3.2(a).

    Expected shapes:
    - q: (batch, n_queries, d)
    - k: (batch, n_keys, d)
    - v: (batch, n_keys, d)

    Returns:
    - output: (batch, n_queries, d)
    - logsumexp: (batch, n_queries)
    """
    _validate_flash_attention_inputs(q, k, v)
    # Handout 1.3.2(a) allows us to ignore causal masking for the pure PyTorch debug path.
    _ = is_causal

    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    scale = q.shape[-1] ** -0.5
    num_query_tiles = math.ceil(n_queries / q_tile_size)
    num_key_tiles = math.ceil(n_keys / k_tile_size)

    output = torch.zeros_like(q)
    logsumexp = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)

    for query_tile_idx in range(num_query_tiles):
        q_start = query_tile_idx * q_tile_size
        q_end = min((query_tile_idx + 1) * q_tile_size, n_queries)
        q_tile = q[:, q_start:q_end, :]

        running_output = torch.zeros_like(q_tile)
        running_l = torch.zeros(q_tile.shape[:-1], dtype=q.dtype, device=q.device)
        running_m = torch.full(q_tile.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)

        for key_tile_idx in range(num_key_tiles):
            k_start = key_tile_idx * k_tile_size
            k_end = min((key_tile_idx + 1) * k_tile_size, n_keys)
            k_tile = k[:, k_start:k_end, :]
            v_tile = v[:, k_start:k_end, :]

            scores_tile = einsum(q_tile, k_tile, "... q d, ... k d -> ... q k") * scale

            prev_running_m = running_m
            running_m = torch.maximum(running_m, reduce(scores_tile, "... q k -> ... q", "max"))
            unnormalized_probs = torch.exp(scores_tile - running_m.unsqueeze(-1))
            running_l = (
                torch.exp(prev_running_m - running_m) * running_l
                + reduce(unnormalized_probs, "... q k -> ... q", "sum")
            )

            # Rescale the previous partial output before accumulating the new key/value tile.
            running_output = (
                running_output * torch.exp(prev_running_m - running_m).unsqueeze(-1)
                + einsum(unnormalized_probs, v_tile, "... q k, ... k d -> ... q d")
            )

        output[:, q_start:q_end, :] = running_output / running_l.unsqueeze(-1)
        logsumexp[:, q_start:q_end] = running_m + torch.log(running_l)

    return output, logsumexp


def _flash_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_tile_size: int,
    k_tile_size: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_flash_attention_inputs(q, k, v)

    if triton is None or tl is None or flash_attention_forward_kernel is None:
        raise RuntimeError("Triton is not available in this environment.")
    if q.device.type != "cuda":
        raise ValueError("The Triton FlashAttention forward path requires CUDA tensors.")

    _ = (q, k, v, q_tile_size, k_tile_size, is_causal)

    # TODO(student): allocate output/logsumexp buffers, choose the launch grid, and call
    # flash_attention_forward_kernel[...] with the correct strides and constexpr tile sizes.
    raise NotImplementedError("TODO: wire the Triton FlashAttention forward kernel launch.")


# Backward helpers
def _flash_attention_backward_pytorch_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    lse: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = (q, k, v, o, grad_o, lse, is_causal)

    # TODO(student): implement Section 1.3.2 flash_backward using Eqs. (13)-(19).
    # Hint: compute D = rowsum(O * dO), then recompute P from S and L.
    raise NotImplementedError("TODO: implement the recomputation-based FlashAttention backward pass.")


class FlashAttention2PyTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _validate_flash_attention_inputs(q, k, v)

        q_tile_size = _choose_query_tile_size(q.shape[-2])
        k_tile_size = _choose_key_tile_size(k.shape[-2])

        # TODO(student): if you want easier debugging, start with explicit q_start/q_end
        # and k_start/k_end boundaries before worrying about helper abstractions.
        output, logsumexp = _flash_attention_forward_pytorch_tiled(
            q,
            k,
            v,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            is_causal=is_causal,
        )

        # Tests expect exactly one saved tensor with shape (batch, n_queries), i.e. L.
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        ctx.q_tile_size = q_tile_size
        ctx.k_tile_size = k_tile_size
        return output

    @staticmethod
    def backward(
        ctx,
        grad_o: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        logsumexp, q, k, v, output = ctx.saved_tensors

        grad_q, grad_k, grad_v = _flash_attention_backward_pytorch_recompute(
            q,
            k,
            v,
            output,
            grad_o,
            logsumexp,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


class FlashAttention2TritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _validate_flash_attention_inputs(q, k, v)

        q_tile_size = _choose_query_tile_size(q.shape[-2])
        k_tile_size = _choose_key_tile_size(k.shape[-2])

        output, logsumexp = _flash_attention_forward_triton(
            q,
            k,
            v,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            is_causal=is_causal,
        )

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        ctx.q_tile_size = q_tile_size
        ctx.k_tile_size = k_tile_size
        return output

    @staticmethod
    def backward(
        ctx,
        grad_o: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        logsumexp, q, k, v, output = ctx.saved_tensors

        grad_q, grad_k, grad_v = _flash_attention_backward_pytorch_recompute(
            q,
            k,
            v,
            output,
            grad_o,
            logsumexp,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None

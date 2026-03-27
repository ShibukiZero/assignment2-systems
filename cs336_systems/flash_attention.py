from __future__ import annotations

import functools
import math
from contextlib import contextmanager

import torch
from einops import reduce, einsum

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is only required on CUDA hosts.
    triton = None
    tl = None


MIN_TILE_SIZE = 16
_FLASH_ATTENTION_TILE_SIZE_OVERRIDE: tuple[int | None, int | None] | None = None


def _resolve_tile_size_override(
    requested_tile_size: int | None,
    *,
    n_tokens: int,
    axis_name: str,
) -> int | None:
    if requested_tile_size is None:
        return None
    if requested_tile_size < MIN_TILE_SIZE:
        raise ValueError(
            f"Expected {axis_name} tile size to be at least {MIN_TILE_SIZE}, "
            f"but got {requested_tile_size}."
        )
    return min(n_tokens, requested_tile_size)


@contextmanager
def flash_attention_tile_size_override(
    *,
    q_tile_size: int | None = None,
    k_tile_size: int | None = None,
):
    """
    Temporarily override the tile sizes used by FlashAttention helpers.

    This is intended for benchmark sweeps, while keeping the public
    autograd.Function interface unchanged for the assignment tests.
    """
    global _FLASH_ATTENTION_TILE_SIZE_OVERRIDE
    previous_override = _FLASH_ATTENTION_TILE_SIZE_OVERRIDE
    _FLASH_ATTENTION_TILE_SIZE_OVERRIDE = (q_tile_size, k_tile_size)
    try:
        yield
    finally:
        _FLASH_ATTENTION_TILE_SIZE_OVERRIDE = previous_override


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
    if _FLASH_ATTENTION_TILE_SIZE_OVERRIDE is not None:
        q_tile_size, _ = _FLASH_ATTENTION_TILE_SIZE_OVERRIDE
        overridden_tile_size = _resolve_tile_size_override(
            q_tile_size,
            n_tokens=n_queries,
            axis_name="query",
        )
        if overridden_tile_size is not None:
            return overridden_tile_size
    return min(n_queries, MIN_TILE_SIZE)


def _choose_key_tile_size(n_keys: int) -> int:
    if _FLASH_ATTENTION_TILE_SIZE_OVERRIDE is not None:
        _, k_tile_size = _FLASH_ATTENTION_TILE_SIZE_OVERRIDE
        overridden_tile_size = _resolve_tile_size_override(
            k_tile_size,
            n_tokens=n_keys,
            axis_name="key",
        )
        if overridden_tile_size is not None:
            return overridden_tile_size
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
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        l_block_ptr = tl.make_block_ptr(
            base=l_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        q_tile = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        output_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        running_l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        running_m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k_tile = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

            scores_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale
            valid_key_mask = key_offsets[None, :] < N_KEYS
            if IS_CAUSAL:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                score_mask = causal_mask & valid_key_mask
            else:
                score_mask = valid_key_mask
            scores_tile = tl.where(score_mask, scores_tile, -1e6)

            prev_running_m = running_m
            running_m = tl.maximum(running_m, tl.max(scores_tile, axis=1))
            unnormalized_probs = tl.exp(scores_tile - running_m[:, None])
            running_l = tl.exp(prev_running_m - running_m) * running_l + tl.sum(unnormalized_probs, axis=1)
            output_tile = output_tile * tl.exp(prev_running_m - running_m)[:, None]
            probs_for_value = unnormalized_probs.to(v_tile.dtype)
            output_tile = tl.dot(probs_for_value, v_tile, acc=output_tile)

            k_block_ptr = tl.advance(k_block_ptr, (K_TILE_SIZE, 0))
            v_block_ptr = tl.advance(v_block_ptr, (K_TILE_SIZE, 0))
        output_tile = output_tile / running_l[:, None]
        logsumexp_tile = running_m + tl.log(running_l)
        output_to_store = output_tile.to(o_block_ptr.type.element_ty)
        tl.store(o_block_ptr, output_to_store, boundary_check=(0, 1))
        tl.store(l_block_ptr, logsumexp_tile, boundary_check=(0,))

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
    Pure PyTorch tiled forward pass used to debug the Triton kernel.

    Expected shapes:
    - q: (batch, n_queries, d)
    - k: (batch, n_keys, d)
    - v: (batch, n_keys, d)

    Returns:
    - output: (batch, n_queries, d)
    - logsumexp: (batch, n_queries)
    """
    _validate_flash_attention_inputs(q, k, v)

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
            if is_causal:
                query_positions = torch.arange(q_start, q_end, device=q.device)
                key_positions = torch.arange(k_start, k_end, device=q.device)
                causal_mask = query_positions[:, None] >= key_positions[None, :]
                scores_tile = torch.where(
                    causal_mask.unsqueeze(0),
                    scores_tile,
                    scores_tile.new_full((), -1e6),
                )

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

    batch_size, n_queries, d = q.shape
    n_keys = k.shape[-2]
    scale = q.shape[-1] ** -0.5
    output = torch.empty_like(q)
    logsumexp = torch.empty((batch_size, n_queries), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(n_queries, q_tile_size), batch_size)

    flash_attention_forward_kernel[grid](
        q, k, v, output, logsumexp,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        logsumexp.stride(0), logsumexp.stride(1),
        n_queries, n_keys, scale,
        D=d,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
        IS_CAUSAL=is_causal,
    )
    return output, logsumexp


# Backward helpers
def _flash_attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_o: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        output_ref, _ = flash_attention_forward_reference(
            q_ref,
            k_ref,
            v_ref,
            is_causal=is_causal,
        )
        grad_q, grad_k, grad_v = torch.autograd.grad(
            outputs=output_ref,
            inputs=(q_ref, k_ref, v_ref),
            grad_outputs=grad_o,
        )
    return grad_q, grad_k, grad_v


def _flash_attention_backward_pytorch_recompute_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    lse: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = q.shape[-1] ** -0.5
    row_dot_grad = reduce(grad_o * o, '... q d -> ... q', 'sum')
    scores = einsum(q, k, '... q d, ... k d -> ... q k') * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        query_positions = torch.arange(n_queries, device=q.device)
        key_positions = torch.arange(n_keys, device=q.device)
        causal_mask = query_positions[:, None] >= key_positions[None, :]
        scores = torch.where(causal_mask.unsqueeze(0), scores, scores.new_full((), -1e6))
    probabilities = torch.exp(scores - lse.unsqueeze(-1))
    grad_v = einsum(probabilities, grad_o, '... q k, ... q d -> ... k d')
    grad_p = einsum(grad_o, v, '... q d, ... k d -> ... q k')
    grad_s = probabilities * (grad_p - row_dot_grad.unsqueeze(-1))
    grad_q = einsum(grad_s, k, '... q k, ... k d -> ... q d') * scale
    grad_k = einsum(grad_s, q, '... q k, ... q d -> ... k d') * scale
    return grad_q, grad_k, grad_v


@functools.lru_cache(maxsize=1)
def _get_compiled_flash_attention_backward():
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return _flash_attention_backward_pytorch_recompute_impl
    return compile_fn(_flash_attention_backward_pytorch_recompute_impl)


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
    compiled_backward = _get_compiled_flash_attention_backward()
    return compiled_backward(
        q,
        k,
        v,
        o,
        grad_o,
        lse,
        is_causal=is_causal,
    )


def _flash_attention_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    lse: torch.Tensor,
    *,
    q_tile_size: int,
    k_tile_size: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton backward entry point for FlashAttention.

    TODO(student): Replace the temporary PyTorch fallback below with a Triton
    implementation of Algorithm 2 from the handout. Keep this wrapper boundary
    stable so the autograd wiring does not need to change again.

    Suggested implementation plan:
    1. Compute the per-row delta term D_i = sum_j dO_ij * O_ij in fp32.
       Hint: this is `row_dot_grad` in the current PyTorch recompute helper.
    2. Launch one Triton kernel for dK and dV, and a separate Triton kernel for dQ.
       Hint: the handout suggests two passes over P to avoid atomics/synchronization.
    3. Recompute scores/probabilities tile-by-tile from Q, K, and saved LSE instead of
       materializing the full attention matrix.
    4. Keep accumulator math in fp32 even when inputs are bf16.
    5. For causal masking, early-exit or skip tiles that are entirely above the diagonal.

    Shape checkpoints before you start filling kernels:
    - q, k, v, o, grad_o: (batch, seq_len, d_head)
    - lse: (batch, seq_len)
    - grad_q, grad_k, grad_v: same shapes as q, k, v
    """
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available in this environment.")
    if q.device.type != "cuda":
        raise ValueError("The Triton FlashAttention backward path requires CUDA tensors.")

    # TODO(student): these are the outputs your Triton kernels should write into.
    # Allocating them here keeps the wrapper signature and dtype/device behavior explicit.
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)

    # TODO(student): compute/store the per-row delta term used in dS = P * (dP - D).
    # A small standalone kernel can be a good first milestone before writing the
    # full tiled backward kernels.
    _unused_delta_shape_hint = (q.shape[0], q.shape[1])

    # Temporary fallback:
    # keep correctness while you incrementally replace this path with Triton kernels.
    del grad_q, grad_k, grad_v, _unused_delta_shape_hint, q_tile_size, k_tile_size
    return _flash_attention_backward_pytorch_recompute(
        q,
        k,
        v,
        o,
        grad_o,
        lse,
        is_causal=is_causal,
    )


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

        grad_q, grad_k, grad_v = _flash_attention_backward_triton(
            q,
            k,
            v,
            output,
            grad_o,
            logsumexp,
            q_tile_size=ctx.q_tile_size,
            k_tile_size=ctx.k_tile_size,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None

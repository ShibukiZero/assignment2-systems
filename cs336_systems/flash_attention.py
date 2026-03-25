from __future__ import annotations

import torch
import math
from torch import Tensor
from einops import reduce, einsum
from jaxtyping import Float


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
    return min(n_queries, 16)


def _choose_key_tile_size(n_keys: int) -> int:
    return min(n_keys, 16)


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
    _ = is_causal


    # Debugging suggestion:
    # compare each intermediate tile update against flash_attention_forward_reference(...)
    # on a small input before trusting the full tiled loop.

    # TODO(student): implement Algorithm 1 in pure PyTorch.
    # Suggested structure:
    # 1. Loop over query tiles of shape (batch, B_q, d).
    # 2. For each query tile, initialize:
    #    - running output O_i^(0) with shape (batch, B_q, d)
    #    - running row max m_i^(0) with shape (batch, B_q)
    #    - running denominator proxy l_i^(0) with shape (batch, B_q)
    # 3. Loop over key/value tiles of shape (batch, B_k, d).
    # 4. Compute S_i^(j), update m and l online, and accumulate the unnormalized output.
    # 5. Normalize the final output tile and write:
    #    - O[:, q_start:q_end, :]
    #    - L[:, q_start:q_end]

    Tq = math.ceil(q.shape[-2] / q_tile_size)
    Tk = math.ceil(k.shape[-2] / k_tile_size)
    o: Float[Tensor, '... Bq d'] = torch.zeros_like(q)
    L: Float[Tensor, '... Bq'] = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)

    for i in range(Tq):
        q_start = i * q_tile_size
        q_end = (i + 1) * q_tile_size
        q_tile: Float[Tensor, '... Bq d'] = q[:, q_start:q_end, :]
        o_tile: Float[Tensor, '... Bq d'] = torch.zeros_like(q_tile)
        l: Float[Tensor, '... Bq'] = torch.zeros(q_tile.shape[:-1], dtype=q_tile.dtype, device=q_tile.device)
        m: Float[Tensor, '... Bq'] = torch.full(q_tile.shape[:-1], float("-inf"), dtype=q_tile.dtype, device=q_tile.device)

        for j in range(Tk):
            k_start = j * k_tile_size
            k_end = (j + 1) * k_tile_size
            k_tile: Float[Tensor, '... Bk d'] = k[:, k_start:k_end, :]
            v_tile: Float[Tensor, '... Bk d'] = v[:, k_start:k_end, :]
            s_tile: Float[Tensor, '... Bq Bk'] = einsum(q_tile, k_tile, '... Bq d, ... Bk d -> ... Bq Bk') / math.sqrt(q_tile.shape[-1])
            m_prev = m
            m: Float[Tensor, '... Bq'] = torch.maximum(m, reduce(s_tile, '... Bq Bk -> ... Bq', 'max'))
            p_tile: Float[Tensor, '... Bq Bk'] = torch.exp(s_tile - m.unsqueeze(-1))
            l: Float[Tensor, '... Bq'] = torch.exp(m_prev - m) * l + reduce(p_tile, '... Bq Bk -> ... Bq', 'sum')
            o_tile: Float[Tensor, '... Bq d'] = o_tile * torch.exp(m_prev - m).unsqueeze(-1) + einsum(p_tile, v_tile, '... Bq Bk, ... Bk d -> ... Bq d')
        o_tile = o_tile / l.unsqueeze(-1)
        L_tile = m + torch.log(l)
        o[:, q_start:q_end, :] = o_tile
        L[:, q_start:q_end] = L_tile

    return o, L


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

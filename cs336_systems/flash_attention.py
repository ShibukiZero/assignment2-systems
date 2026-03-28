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


    @triton.jit
    def flash_attention_backward_delta_kernel(
        o_ptr, grad_o_ptr, delta_ptr,
        stride_ob, stride_oq, stride_od,
        stride_gob, stride_goq, stride_god,
        stride_db, stride_dq,
        N_QUERIES,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
    ):
        """
        TODO(student): compute the per-row delta term
            D_i = sum_j dO_ij * O_ij
        for one query tile.

        Suggested milestone:
        - implement this kernel first
        - compare its output against `reduce(grad_o * o, '... q d -> ... q', 'sum')`
        - keep accumulation in fp32
        """
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # TODO(student): replace this placeholder body with a real tiled reduction.
        # Shape checkpoint:
        # - output tile: (Q_TILE_SIZE, D)
        # - grad_o tile: (Q_TILE_SIZE, D)
        # - delta tile: (Q_TILE_SIZE,)
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        grad_o_block_ptr = tl.make_block_ptr(
            base=grad_o_ptr + batch_index * stride_gob,
            shape=(N_QUERIES, D),
            strides=(stride_goq, stride_god),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        delta_block_ptr = tl.make_block_ptr(
            base=delta_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        o_tile = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o_tile = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")
        delta_tile = tl.sum(grad_o_tile * o_tile, axis=1)
        tl.store(delta_block_ptr, delta_tile, boundary_check=(0,))


    @triton.jit
    def flash_attention_backward_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, grad_o_ptr, lse_ptr, delta_ptr, grad_k_ptr, grad_v_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_gob, stride_goq, stride_god,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_gkb, stride_gkk, stride_gkd,
        stride_gvb, stride_gvk, stride_gvd,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        TODO(student): implement the dK / dV pass from Algorithm 2.

        Hints:
        - one program instance should usually own one (batch, key-tile) output tile
        - recompute scores/probabilities from Q, K, and LSE instead of loading P
        - write one K/V tile of grad_k and grad_v without atomics if you keep ownership disjoint
        - for causal masking, skip query tiles that are fully above the diagonal
        """
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # TODO(student): replace this placeholder body with the tiled dK/dV kernel.
        #
        # Recommended flow for this pass:
        # 1. Let this program instance own exactly one (batch, key_tile) output tile.
        #    That means only this block should write the corresponding grad_k / grad_v tile.
        # 2. Load the current K_j and V_j tile once, and initialize fp32 accumulators
        #    for grad_k_j and grad_v_j with shape (K_TILE_SIZE, D).
        # 3. Sweep over all query tiles Q_i:
        #    - load Q_i, grad_O_i, LSE_i, and delta_i
        #    - recompute local scores S_ij = Q_i K_j^T / sqrt(d)
        #    - apply causal masking for the current (query_tile, key_tile) pair
        #    - recover local probabilities P_ij = exp(S_ij - LSE_i)
        # 4. Accumulate grad_v_j using the local contribution
        #       grad_v_j += P_ij^T @ grad_O_i
        # 5. Compute the local dP tile using the owned V_j tile
        #       dP_ij = grad_O_i @ V_j^T
        #    then form
        #       dS_ij = P_ij * (dP_ij - delta_i)
        #    where delta_i is broadcast along the key dimension.
        # 6. Accumulate grad_k_j using the local contribution
        #       grad_k_j += dS_ij^T @ Q_i
        # 7. After all query tiles are processed, cast/store the final grad_k_j and
        #    grad_v_j tile exactly once.
        #
        # Shape checkpoints:
        # - owned key/value tile:      (K_TILE_SIZE, D)
        # - one query tile Q_i:        (Q_TILE_SIZE, D)
        # - local scores/probs P_ij:   (Q_TILE_SIZE, K_TILE_SIZE)
        # - local grad_O_i:            (Q_TILE_SIZE, D)
        # - local delta_i:             (Q_TILE_SIZE,)
        # - accum grad_k_j / grad_v_j: (K_TILE_SIZE, D)
        #
        # Causal masking hint:
        # - if this key tile is strictly to the right of a query tile, that query tile
        #   contributes nothing and can be skipped entirely.
        # - only the diagonal-intersecting tiles need per-element masking.
        _ = key_tile_index
        _ = batch_index
        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        grad_k_block_ptr = tl.make_block_ptr(
            base=grad_k_ptr + batch_index * stride_gkb,
            shape=(N_KEYS, D),
            strides=(stride_gkk, stride_gkd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        grad_v_block_ptr = tl.make_block_ptr(
            base=grad_v_ptr + batch_index * stride_gvb,
            shape=(N_KEYS, D),
            strides=(stride_gvk, stride_gvd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        grad_o_ptr = tl.make_block_ptr(
            base=grad_o_ptr + batch_index * stride_gob,
            shape=(N_QUERIES, D),
            strides=(stride_goq, stride_god),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        lse_ptr = tl.make_block_ptr(
            base=lse_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(0,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        delta_ptr = tl.make_block_ptr(
            base=delta_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(0,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        k_tile = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_k_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        grad_v_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
            q_tile = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_o_tile = tl.load(grad_o_ptr, boundary_check=(0, 1), padding_option="zero")
            lse_tile = tl.load(lse_ptr, boundary_check=(0,), padding_option="zero")
            delta_tile = tl.load(delta_ptr, boundary_check=(0,), padding_option="zero")
            scores_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale
            query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            valid_query_mask = query_offsets[:, None] < N_QUERIES
            valid_key_mask = key_offsets[None, :] < N_KEYS
            if IS_CAUSAL:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                score_mask = causal_mask & valid_query_mask & valid_key_mask
            else:
                score_mask = valid_query_mask & valid_key_mask
            scores_tile = tl.where(score_mask, scores_tile, -1e6)
            probabilities_tile = tl.exp(scores_tile - lse_tile[:, None])
            probs_for_grad_o = probabilities_tile.to(grad_o_tile.dtype)
            grad_v_tile = tl.dot(tl.trans(probs_for_grad_o), grad_o_tile, acc=grad_v_tile)
            grad_s_tile = probabilities_tile * (tl.dot(grad_o_tile, tl.trans(v_tile)) - delta_tile[:, None])
            grad_k_tile = tl.dot(tl.trans(grad_s_tile.to(q_tile.dtype)), q_tile, acc=grad_k_tile)
            q_block_ptr = tl.advance(q_block_ptr, (Q_TILE_SIZE, 0))
            grad_o_ptr = tl.advance(grad_o_ptr, (Q_TILE_SIZE, 0))
            lse_ptr = tl.advance(lse_ptr, (Q_TILE_SIZE,))
            delta_ptr = tl.advance(delta_ptr, (Q_TILE_SIZE,))
        grad_k_tile = grad_k_tile * scale
        tl.store(grad_k_block_ptr, grad_k_tile.to(grad_k_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(grad_v_block_ptr, grad_v_tile.to(grad_v_block_ptr.type.element_ty), boundary_check=(0, 1))



    @triton.jit
    def flash_attention_backward_dq_kernel(
        q_ptr, k_ptr, v_ptr, grad_o_ptr, lse_ptr, delta_ptr, grad_q_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_gob, stride_goq, stride_god,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_gqb, stride_gqq, stride_gqd,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        TODO(student): implement the dQ pass from Algorithm 2.

        Hints:
        - one program instance should usually own one (batch, query-tile) output tile
        - accumulate contributions from all relevant key tiles in fp32
        - use the same recomputed probability logic as the dK/dV pass
        - this pass is naturally separate from dK/dV, which avoids synchronization
        """
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # TODO(student): replace this placeholder body with the tiled dQ kernel.
        #
        # Recommended flow for this pass:
        # 1. Let this program instance own exactly one (batch, query_tile) output tile.
        #    That means only this block should write the corresponding grad_q tile.
        # 2. Load the current Q_i tile once, together with the matching grad_O_i,
        #    LSE_i, and delta_i values for that same query tile.
        # 3. Initialize an fp32 accumulator grad_q_i with shape (Q_TILE_SIZE, D).
        # 4. Sweep over all key/value tiles K_j / V_j:
        #    - load K_j and V_j
        #    - recompute local scores S_ij = Q_i K_j^T / sqrt(d)
        #    - apply causal masking for the current (query_tile, key_tile) pair
        #    - recover local probabilities P_ij = exp(S_ij - LSE_i)
        # 5. Compute the local dP tile using the owned V_j tile
        #       dP_ij = grad_O_i @ V_j^T
        #    then form
        #       dS_ij = P_ij * (dP_ij - delta_i)
        #    where delta_i is broadcast along the key dimension.
        # 6. Accumulate grad_q_i using the local contribution
        #       grad_q_i += dS_ij @ K_j
        #    and remember the final scale factor:
        #       grad_q_i = grad_q_i * scale
        #    The multiplication by `scale` should happen exactly once overall.
        # 7. After all key tiles are processed, cast/store the final grad_q_i tile
        #    exactly once.
        #
        # Shape checkpoints:
        # - owned query tile Q_i:      (Q_TILE_SIZE, D)
        # - local grad_O_i:            (Q_TILE_SIZE, D)
        # - local LSE_i / delta_i:     (Q_TILE_SIZE,)
        # - one key/value tile K_j:    (K_TILE_SIZE, D)
        # - local scores/probs P_ij:   (Q_TILE_SIZE, K_TILE_SIZE)
        # - accum grad_q_i:            (Q_TILE_SIZE, D)
        #
        # Causal masking hint:
        # - if a key tile is strictly to the right of this query tile, it contributes
        #   nothing and can be skipped entirely.
        # - only the diagonal-intersecting tiles need per-element masking.
        #
        # Debugging hint:
        # - this pass is the mirror image of the dK/dV pass:
        #   here the output owner is the query tile, and the reduction dimension is
        #   the key/value axis.
        _ = query_tile_index
        _ = batch_index
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
        grad_o_ptr = tl.make_block_ptr(
            base=grad_o_ptr + batch_index * stride_gob,
            shape=(N_QUERIES, D),
            strides=(stride_goq, stride_god),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        lse_ptr = tl.make_block_ptr(
            base=lse_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        delta_ptr = tl.make_block_ptr(
            base=delta_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        grad_q_block_ptr = tl.make_block_ptr(
            base=grad_q_ptr + batch_index * stride_gqb,
            shape=(N_QUERIES, D),
            strides=(stride_gqq, stride_gqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        q_tile = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o_tile = tl.load(grad_o_ptr, boundary_check=(0, 1), padding_option="zero")
        lse_tile = tl.load(lse_ptr, boundary_check=(0,), padding_option="zero")
        delta_tile = tl.load(delta_ptr, boundary_check=(0,), padding_option="zero")
        grad_q_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k_tile = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            scores_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale
            query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            valid_query_mask = query_offsets[:, None] < N_QUERIES
            valid_key_mask = key_offsets[None, :] < N_KEYS
            if IS_CAUSAL:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                score_mask = causal_mask & valid_query_mask & valid_key_mask
            else:
                score_mask = valid_query_mask & valid_key_mask
            scores_tile = tl.where(score_mask, scores_tile, -1e6)
            probabilities_tile = tl.exp(scores_tile - lse_tile[:, None])
            grad_s_tile = probabilities_tile * (tl.dot(grad_o_tile, tl.trans(v_tile)) - delta_tile[:, None])
            grad_q_tile = tl.dot(grad_s_tile.to(k_tile.dtype), k_tile, acc=grad_q_tile)
            k_block_ptr = tl.advance(k_block_ptr, (K_TILE_SIZE, 0))
            v_block_ptr = tl.advance(v_block_ptr, (K_TILE_SIZE, 0))
        grad_q_tile = grad_q_tile * scale
        tl.store(grad_q_block_ptr, grad_q_tile.to(grad_q_block_ptr.type.element_ty), boundary_check=(0, 1))
        


else:
    flash_attention_forward_kernel = None
    flash_attention_backward_delta_kernel = None
    flash_attention_backward_dkdv_kernel = None
    flash_attention_backward_dq_kernel = None


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


# Triton backward helper + Triton autograd wiring are intentionally colocated
# with the Triton forward helper above so kernel iteration stays in one region.
def _flash_attention_backward_delta_reference(
    o: torch.Tensor,
    grad_o: torch.Tensor,
) -> torch.Tensor:
    if o.shape != grad_o.shape:
        raise ValueError("Expected output and grad_o to have the same shape.")
    return (o.to(torch.float32) * grad_o.to(torch.float32)).sum(dim=-1)


def _flash_attention_backward_delta_triton(
    o: torch.Tensor,
    grad_o: torch.Tensor,
    *,
    q_tile_size: int,
) -> torch.Tensor:
    if o.shape != grad_o.shape:
        raise ValueError("Expected output and grad_o to have the same shape.")
    if triton is None or tl is None or flash_attention_backward_delta_kernel is None:
        raise RuntimeError("Triton is not available in this environment.")
    if o.device.type != "cuda" or grad_o.device.type != "cuda":
        raise ValueError("The Triton FlashAttention delta path requires CUDA tensors.")

    batch_size, n_queries, d = o.shape
    delta = torch.empty((batch_size, n_queries), dtype=torch.float32, device=o.device)
    grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
    flash_attention_backward_delta_kernel[grid](
        o, grad_o, delta,
        o.stride(0), o.stride(1), o.stride(2),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        delta.stride(0), delta.stride(1),
        n_queries,
        D=d,
        Q_TILE_SIZE=q_tile_size,
    )
    return delta


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

    # TODO(student): once the dQ / dK / dV Triton kernels are ready, thread this delta
    # tensor through them instead of falling back to the PyTorch recompute helper.
    delta = _flash_attention_backward_delta_triton(
        o,
        grad_o,
        q_tile_size=q_tile_size,
    )
    batch_size, n_queries, d = q.shape
    n_keys = k.shape[-2]
    scale = q.shape[-1] ** -0.5

    dkdv_grid = (triton.cdiv(n_keys, k_tile_size), batch_size)
    flash_attention_backward_dkdv_kernel[dkdv_grid](
        q, k, v, grad_o, lse, delta, grad_k, grad_v,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2),
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        n_queries, n_keys, scale,
        D=d,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
        IS_CAUSAL=is_causal,
    )
    dq_grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
    flash_attention_backward_dq_kernel[dq_grid](
        q, k, v, grad_o, lse, delta, grad_q,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
        n_queries, n_keys, scale,
        D=d,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
        IS_CAUSAL=is_causal,
    )
    return grad_q, grad_k, grad_v


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
    output_dtype_q = q.dtype
    output_dtype_k = k.dtype
    output_dtype_v = v.dtype
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
    return (
        grad_q.to(output_dtype_q),
        grad_k.to(output_dtype_k),
        grad_v.to(output_dtype_v),
    )


@functools.lru_cache(maxsize=8)
def _get_compiled_flash_attention_backward(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    o_dtype: torch.dtype,
    grad_o_dtype: torch.dtype,
    lse_dtype: torch.dtype,
):
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
    compiled_backward = _get_compiled_flash_attention_backward(
        q.dtype,
        k.dtype,
        v.dtype,
        o.dtype,
        grad_o.dtype,
        lse.dtype,
    )
    return compiled_backward(
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

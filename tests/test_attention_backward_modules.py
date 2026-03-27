import pytest
import torch

from cs336_systems.flash_attention import (
    _flash_attention_backward_delta_reference,
    _flash_attention_backward_delta_triton,
    _flash_attention_backward_pytorch_recompute,
    _flash_attention_backward_reference,
    _flash_attention_backward_triton,
    _flash_attention_forward_triton,
    flash_attention_forward_reference,
)


def _make_backward_module_inputs(device: str | None = None, dtype: torch.dtype = torch.float32):
    torch.random.manual_seed(0)
    batch_size = 2
    n_queries = 64
    n_keys = 64
    d_head = 32

    q = torch.randn(batch_size, n_queries, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, n_keys, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, n_keys, d_head, device=device, dtype=dtype, requires_grad=True)
    grad_o = torch.randn(batch_size, n_queries, d_head, device=device, dtype=dtype)
    return q, k, v, grad_o


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_backward_reference_matches_pytorch_recompute(is_causal):
    q, k, v, grad_o = _make_backward_module_inputs()
    output, lse = flash_attention_forward_reference(q, k, v, is_causal=is_causal)

    grad_q_ref, grad_k_ref, grad_v_ref = _flash_attention_backward_reference(
        q,
        k,
        v,
        grad_o,
        is_causal=is_causal,
    )
    grad_q_recompute, grad_k_recompute, grad_v_recompute = _flash_attention_backward_pytorch_recompute(
        q,
        k,
        v,
        output,
        grad_o,
        lse,
        is_causal=is_causal,
    )

    torch.testing.assert_close(grad_q_ref, grad_q_recompute, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad_k_ref, grad_k_recompute, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad_v_ref, grad_v_recompute, rtol=1e-2, atol=1e-2)


def test_flash_backward_delta_reference_matches_manual_row_dot():
    q, k, v, grad_o = _make_backward_module_inputs()
    output, _lse = flash_attention_forward_reference(q, k, v, is_causal=False)

    delta_expected = (output.to(torch.float32) * grad_o.to(torch.float32)).sum(dim=-1)
    delta_actual = _flash_attention_backward_delta_reference(output, grad_o)

    torch.testing.assert_close(delta_expected, delta_actual, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_backward_delta_triton_matches_reference(dtype):
    q, k, v, grad_o = _make_backward_module_inputs(device="cuda", dtype=dtype)
    output, _lse = flash_attention_forward_reference(q, k, v, is_causal=False)

    delta_expected = _flash_attention_backward_delta_reference(output, grad_o)
    delta_actual = _flash_attention_backward_delta_triton(
        output,
        grad_o,
        q_tile_size=16,
    )

    torch.testing.assert_close(delta_expected, delta_actual, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_backward_triton_helper_matches_pytorch_recompute(is_causal, dtype):
    q, k, v, grad_o = _make_backward_module_inputs(device="cuda", dtype=dtype)
    output, lse = _flash_attention_forward_triton(
        q,
        k,
        v,
        q_tile_size=16,
        k_tile_size=16,
        is_causal=is_causal,
    )

    grad_q_expected, grad_k_expected, grad_v_expected = _flash_attention_backward_pytorch_recompute(
        q,
        k,
        v,
        output,
        grad_o,
        lse,
        is_causal=is_causal,
    )
    grad_q_actual, grad_k_actual, grad_v_actual = _flash_attention_backward_triton(
        q,
        k,
        v,
        output,
        grad_o,
        lse,
        q_tile_size=16,
        k_tile_size=16,
        is_causal=is_causal,
    )

    torch.testing.assert_close(grad_q_expected, grad_q_actual, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad_k_expected, grad_k_actual, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad_v_expected, grad_v_actual, rtol=1e-2, atol=1e-2)

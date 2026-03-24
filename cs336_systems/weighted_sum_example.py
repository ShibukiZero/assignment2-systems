from __future__ import annotations

"""Runnable Triton version of the weighted-sum example from Assignment 2, Section 1.3.1."""

import argparse
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_x_row,
    stride_x_dim,
    stride_weight_dim,
    stride_output_row,
    num_rows,
    d,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(num_rows, d),
        strides=(stride_x_row, stride_x_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(d,),
        strides=(stride_weight_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(num_rows,),
        strides=(stride_output_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for _ in range(tl.cdiv(d, D_TILE_SIZE)):
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(x_tile * weight_tile[None, :], axis=1)
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_bwd(
    x_ptr,
    weight_ptr,
    grad_output_ptr,
    grad_x_ptr,
    partial_grad_weight_ptr,
    stride_x_row,
    stride_x_dim,
    stride_weight_dim,
    stride_grad_output_row,
    stride_grad_x_row,
    stride_grad_x_dim,
    stride_partial_grad_weight_row,
    stride_partial_grad_weight_dim,
    num_rows,
    d,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    num_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(num_rows,),
        strides=(stride_grad_output_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(num_rows, d),
        strides=(stride_x_row, stride_x_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(d,),
        strides=(stride_weight_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(num_rows, d),
        strides=(stride_grad_x_row, stride_grad_x_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(num_row_tiles, d),
        strides=(stride_partial_grad_weight_row, stride_partial_grad_weight_dim),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    for _ in range(tl.cdiv(d, D_TILE_SIZE)):
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        grad_x_tile = grad_output[:, None] * weight_tile[None, :]
        tl.store(grad_x_block_ptr, grad_x_tile, boundary_check=(0, 1))

        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        partial_grad_weight_tile = tl.sum(x_tile * grad_output[:, None], axis=0)[None, :]
        tl.store(partial_grad_weight_block_ptr, partial_grad_weight_tile, boundary_check=(0, 1))

        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
        grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = tl.advance(partial_grad_weight_block_ptr, (0, D_TILE_SIZE))


def weighted_sum_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x * weight).sum(dim=-1)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_arg]


def choose_d_tile_size(d: int) -> int:
    target = max(16, (d + 7) // 8)
    return min(128, triton.next_power_of_2(target))


class WeightedSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("Expected x to have at least one dimension.")
        if x.shape[-1] != weight.shape[0]:
            raise ValueError("Expected weight.shape == (x.shape[-1],).")
        if weight.ndim != 1:
            raise ValueError("Expected a 1D weight vector.")
        if x.device.type != "cuda" or weight.device.type != "cuda":
            raise ValueError("This example requires CUDA tensors.")
        if x.dtype != weight.dtype:
            raise ValueError("For this demo, x and weight must use the same dtype.")

        input_shape = x.shape
        # The handout kernel is easiest to express over a 2D matrix of rows.
        x_2d = x.contiguous().reshape(-1, input_shape[-1])
        weight = weight.contiguous()

        rows_tile_size = 16
        d_tile_size = choose_d_tile_size(input_shape[-1])
        output = torch.empty((x_2d.shape[0],), device=x.device, dtype=x.dtype)

        weighted_sum_fwd[(triton.cdiv(x_2d.shape[0], rows_tile_size),)](
            x_2d,
            weight,
            output,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            output.stride(0),
            num_rows=x_2d.shape[0],
            d=input_shape[-1],
            ROWS_TILE_SIZE=rows_tile_size,
            D_TILE_SIZE=d_tile_size,
        )

        ctx.save_for_backward(x_2d, weight)
        ctx.input_shape = input_shape
        ctx.rows_tile_size = rows_tile_size
        ctx.d_tile_size = d_tile_size
        return output.view(*input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_2d, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous().reshape(-1)
        grad_x = torch.empty_like(x_2d)
        # Each row tile writes one partial gradient row; we reduce them in PyTorch.
        partial_grad_weight = torch.empty(
            (triton.cdiv(x_2d.shape[0], ctx.rows_tile_size), x_2d.shape[1]),
            device=x_2d.device,
            dtype=torch.float32,
        )

        weighted_sum_bwd[(partial_grad_weight.shape[0],)](
            x_2d,
            weight,
            grad_out,
            grad_x,
            partial_grad_weight,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            num_rows=x_2d.shape[0],
            d=x_2d.shape[1],
            ROWS_TILE_SIZE=ctx.rows_tile_size,
            D_TILE_SIZE=ctx.d_tile_size,
        )

        grad_weight = partial_grad_weight.sum(dim=0).to(weight.dtype)
        return grad_x.view(*ctx.input_shape), grad_weight


weighted_sum_triton = WeightedSumFunction.apply


@dataclass(frozen=True)
class DemoConfig:
    batch_size: int
    sequence_length: int
    embedding_dim: int
    dtype: str
    device: str
    seed: int
    atol: float
    rtol: float
    print_sample: bool


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run the weighted-sum Triton example from the Assignment 2 handout and "
            "compare it against a PyTorch reference implementation."
        )
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Input dtype for both x and weight.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda"),
        default="auto",
        help="Triton requires CUDA, so auto will prefer CUDA and error otherwise.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="Print a small slice of the forward outputs for quick inspection.",
    )
    args = parser.parse_args()
    return DemoConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        print_sample=args.print_sample,
    )


def run_demo(config: DemoConfig) -> None:
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise ValueError("This Triton example requires a CUDA device.")

    dtype = resolve_dtype(config.dtype)
    torch.manual_seed(config.seed)

    shape = (config.batch_size, config.sequence_length, config.embedding_dim)
    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(config.embedding_dim, device=device, dtype=dtype, requires_grad=True)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)

    triton_output = weighted_sum_triton(x, weight)
    reference_output = weighted_sum_reference(ref_x, ref_weight)
    grad_out = torch.randn_like(reference_output)

    triton_output.backward(grad_out)
    reference_output.backward(grad_out)

    torch.cuda.synchronize(device)

    forward_max_abs_diff = (triton_output - reference_output).abs().max().item()
    grad_x_max_abs_diff = (x.grad - ref_x.grad).abs().max().item()
    grad_weight_max_abs_diff = (weight.grad - ref_weight.grad).abs().max().item()

    print("Weighted sum Triton demo")
    print(f"  input shape: {shape}")
    print(f"  flattened rows: {config.batch_size * config.sequence_length}")
    print(f"  rows tile size: 16")
    print(f"  dim tile size: {choose_d_tile_size(config.embedding_dim)}")
    print(f"  launch grid: ({triton.cdiv(config.batch_size * config.sequence_length, 16)},)")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    print(f"  forward max abs diff: {forward_max_abs_diff:.6e}")
    print(f"  grad_x max abs diff: {grad_x_max_abs_diff:.6e}")
    print(f"  grad_weight max abs diff: {grad_weight_max_abs_diff:.6e}")

    if config.print_sample:
        sample = triton_output.reshape(-1)[: min(8, triton_output.numel())]
        print(f"  output sample: {sample}")

    if not torch.allclose(triton_output, reference_output, atol=config.atol, rtol=config.rtol):
        raise SystemExit("Forward outputs do not match the PyTorch reference.")
    if not torch.allclose(x.grad, ref_x.grad, atol=config.atol, rtol=config.rtol):
        raise SystemExit("grad_x does not match the PyTorch reference.")
    if not torch.allclose(weight.grad, ref_weight.grad, atol=config.atol, rtol=config.rtol):
        raise SystemExit("grad_weight does not match the PyTorch reference.")

    print("  status: all checks passed")


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()

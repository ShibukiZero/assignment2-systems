from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the dtypes that appear in the Assignment 2 mixed-precision toy model "
            "under FP16 autocast on CUDA."
        )
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the observed dtypes as JSON.",
    )
    return parser.parse_args()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA because it inspects CUDA autocast behavior.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = ToyModel(args.in_features, args.out_features).to(device=device, dtype=torch.float32)
    x = torch.randn(args.batch_size, args.in_features, device=device, dtype=torch.float32)
    targets = torch.randint(args.out_features, (args.batch_size,), device=device, dtype=torch.long)

    observed: dict[str, str] = {}

    def capture_output(name: str):
        def hook(_module, _inputs, output):
            observed[name] = dtype_name(output.dtype)

        return hook

    handles = [
        model.fc1.register_forward_hook(capture_output("fc1_output")),
        model.ln.register_forward_hook(capture_output("layer_norm_output")),
        model.fc2.register_forward_hook(capture_output("logits")),
    ]

    try:
        observed["parameters"] = dtype_name(next(model.parameters()).dtype)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, targets)

        observed["loss"] = dtype_name(loss.dtype)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_dtypes = sorted({dtype_name(param.grad.dtype) for param in model.parameters() if param.grad is not None})
        observed["gradients"] = ", ".join(grad_dtypes)
    finally:
        for handle in handles:
            handle.remove()

    print(json.dumps(observed, indent=2, sort_keys=True))

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

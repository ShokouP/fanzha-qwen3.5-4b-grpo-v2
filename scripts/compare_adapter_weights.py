"""Read-only, tensor-wise comparison of two saved LoRA adapters."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = [Path(p) / "adapter_model.safetensors" for p in (args.before, args.after)]
    report = {"before": str(paths[0]), "after": str(paths[1]), "tensors": 0,
              "elements": 0, "changed_elements": 0, "max_abs_diff": 0.0}
    squared = 0.0
    with safe_open(str(paths[0]), framework="pt", device="cpu") as before, safe_open(
        str(paths[1]), framework="pt", device="cpu"
    ) as after:
        if set(before.keys()) != set(after.keys()):
            raise ValueError("Adapter tensor keys differ")
        for key in before.keys():
            a, b = before.get_tensor(key), after.get_tensor(key)
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {key}")
            delta = b.float() - a.float()
            if not torch.isfinite(delta).all():
                raise ValueError(f"Nonfinite weights: {key}")
            report["tensors"] += 1
            report["elements"] += delta.numel()
            report["changed_elements"] += int(torch.count_nonzero(delta))
            report["max_abs_diff"] = max(report["max_abs_diff"], float(delta.abs().max()))
            squared += float((delta.double() ** 2).sum())
    report["l2_diff"] = squared ** 0.5
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

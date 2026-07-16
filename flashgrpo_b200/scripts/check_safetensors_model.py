#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def shard_paths(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(data.get("weight_map", {}).values()))
        return [model_dir / name for name in names]
    return sorted(model_dir.glob("*.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate safetensors shards without loading a model on GPU.")
    parser.add_argument("model_dir")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    shards = shard_paths(model_dir)
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in {model_dir}")

    print(f"model_dir={model_dir}")
    print(f"num_shards={len(shards)}")
    ok = True
    for shard in shards:
        if not shard.exists():
            print(f"MISSING {shard}")
            ok = False
            continue
        try:
            with safe_open(shard, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                metadata = f.metadata()
            print(f"OK {shard.name} bytes={shard.stat().st_size} tensors={len(keys)} metadata={metadata}")
        except Exception as exc:
            ok = False
            print(f"BAD {shard.name} bytes={shard.stat().st_size} error={type(exc).__name__}: {exc}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

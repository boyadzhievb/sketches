#!/usr/bin/env python3
"""
Resize extracted doodles for web and generate a JSON manifest.

Reads full-size PNGs from extracted_doodles/, resizes to max 600px,
saves optimized versions to images/doodles/, and writes doodles.json.
"""

import json
import sys
from pathlib import Path

from PIL import Image

MAX_SIZE = 600
QUALITY = 85


def optimize(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src_dir.glob("*.png"))
    pngs = [p for p in pngs if not p.name.startswith("_")]

    print(f"Optimizing {len(pngs)} doodles → {dst_dir}/")

    manifest = []
    total_saved = 0

    for p in pngs:
        img = Image.open(p)
        w, h = img.size

        # Resize if larger than MAX_SIZE in either dimension
        if max(w, h) > MAX_SIZE:
            ratio = MAX_SIZE / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        out_path = dst_dir / p.name
        img.save(out_path, "PNG", optimize=True)

        src_size = p.stat().st_size
        dst_size = out_path.stat().st_size
        saved = src_size - dst_size
        total_saved += saved

        # Parse source sheet from filename
        stem = p.stem
        parts = stem.rsplit("_doodle_", 1)
        sheet = parts[0] if len(parts) == 2 else stem
        idx = int(parts[1]) if len(parts) == 2 else 0

        manifest.append({
            "file": p.name,
            "sheet": sheet,
            "index": idx,
            "width": img.size[0],
            "height": img.size[1],
        })

        print(f"  {p.name}: {src_size // 1024}K → {dst_size // 1024}K")

    # Write manifest
    manifest_path = dst_dir.parent.parent / "doodles.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} doodles optimized.")
    print(f"Saved {total_saved // (1024 * 1024)}MB total.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("extracted_doodles")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("images/doodles")
    optimize(src, dst)

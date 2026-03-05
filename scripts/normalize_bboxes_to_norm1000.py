#!/usr/bin/env python3
"""
Normalize ms-swift grounding jsonl bboxes from pixel coords to norm1000 coords.

Why:
  Qwen-VL style grounding typically uses [0, 1000] normalized coordinates.
  If your dataset's objects.bbox is in pixel space (e.g. 640x480), this script
  converts it to norm1000 using the *actual image size*.

Input format (ms-swift grounding):
  {
    "messages": [{"role":"user","content":"<image>..."}, {"role":"assistant","content":"<ref-object><bbox>"}],
    "images": [".../img.png"],
    "objects": {"ref": ["tomato fork"], "bbox": [[x1,y1,x2,y2]]}
  }

It only rewrites objects.bbox. It does NOT touch images or messages.

Examples:
  # Convert one file
  python scripts/normalize_bboxes_to_norm1000.py \
    --input dataset/session_2026023_142020/tomato_fork_train.ms_swift.jsonl \
    --output dataset/session_2026023_142020/tomato_fork_train_norm1000.ms_swift.jsonl

  # Convert all *.ms_swift.jsonl in a directory (writes *_norm1000.ms_swift.jsonl)
  python scripts/normalize_bboxes_to_norm1000.py \
    --input dataset/session_2026023_142020 \
    --output dataset/session_2026023_142020
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


_HOST_ROOT = "/data/shengbao.li_home/workspace/ms-swift"
_CONTAINER_ROOT = "/workspace/ms-swift"


def _resolve_image_path(p: str) -> str:
    if os.path.exists(p):
        return p
    cand = p.replace(_CONTAINER_ROOT, _HOST_ROOT, 1)
    if cand != p and os.path.exists(cand):
        return cand
    cand = p.replace(_HOST_ROOT, _CONTAINER_ROOT, 1)
    if cand != p and os.path.exists(cand):
        return cand
    return p


def _iter_input_files(inp: Path) -> Iterable[Path]:
    if inp.is_file():
        yield inp
        return
    if inp.is_dir():
        for p in sorted(inp.glob("*.ms_swift.jsonl")):
            yield p
        return
    raise FileNotFoundError(f"input not found: {inp}")


def _get_image_path(rec: Dict[str, Any]) -> str:
    images = rec.get("images") or []
    if not images:
        raise ValueError("record has no images")
    img0 = images[0]
    if isinstance(img0, str):
        return img0
    if isinstance(img0, dict) and isinstance(img0.get("path"), str):
        return img0["path"]
    raise ValueError(f"unsupported images[0] type: {type(img0)}")


def _normalize_bbox_norm1000(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    # clamp in pixel space first (defensive)
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # pixel -> norm1000
    return [
        round(x1 / w * 1000.0, 3),
        round(y1 / h * 1000.0, 3),
        round(x2 / w * 1000.0, 3),
        round(y2 / h * 1000.0, 3),
    ]


def _convert_file(in_path: Path, out_path: Path, *, assume_pixel: bool) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size_cache: Dict[str, Tuple[int, int]] = {}

    n = 0
    changed = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1

            objs = rec.get("objects") or {}
            bboxes = objs.get("bbox") or []
            if not bboxes:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            img_path = _get_image_path(rec)
            img_fs = _resolve_image_path(img_path)
            if img_fs not in size_cache:
                with Image.open(img_fs) as im:
                    size_cache[img_fs] = im.size  # (w,h)
            w, h = size_cache[img_fs]

            # Heuristic: if assume_pixel=False and bbox looks already norm1000, skip.
            # Note: for 640x480 data, pixel bboxes can be <=1000 too, so user can force --assume_pixel.
            if not assume_pixel:
                looks_norm = all(0.0 <= float(v) <= 1000.0 for bb in bboxes for v in bb)
                if looks_norm:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

            new_bboxes = [_normalize_bbox_norm1000(bb, w, h) for bb in bboxes]
            rec.setdefault("objects", {})
            rec["objects"]["bbox"] = new_bboxes
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            changed += 1

    return n, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input jsonl file or directory")
    ap.add_argument("--output", required=True, help="output jsonl file or directory")
    ap.add_argument(
        "--assume_pixel",
        action="store_true",
        help="Force treating objects.bbox as pixel coords (recommended for your 640x480 labels).",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    total = 0
    total_changed = 0
    for in_path in _iter_input_files(inp):
        if out.is_dir() or str(args.output).endswith("/"):
            out_path = out / (in_path.stem + "_norm1000" + in_path.suffix)
        else:
            # single file mode
            out_path = out

        n, changed = _convert_file(in_path, out_path, assume_pixel=args.assume_pixel)
        print(f"{in_path} -> {out_path}  records={n} changed={changed}")
        total += n
        total_changed += changed

    print(f"total_records={total} total_changed={total_changed}")


if __name__ == "__main__":
    main()


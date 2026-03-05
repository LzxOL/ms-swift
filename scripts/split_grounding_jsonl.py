#!/usr/bin/env python3
"""
Split an ms-swift grounding jsonl into train/val/test.

Important: if the dataset contains augmented images (e.g. *_aug00.png),
we group by the *base image id* so augmented variants do not leak across splits.

Example:
  python scripts/split_grounding_jsonl.py \
    --input_jsonl dataset/session_2026023_142020/tomato_fork_train_aug.ms_swift.jsonl \
    --output_dir dataset/session_2026023_142020 \
    --train_name tomato_fork_train.ms_swift.jsonl \
    --val_name tomato_fork_val.ms_swift.jsonl \
    --test_name tomato_fork_test.ms_swift.jsonl \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _group_key_from_image_path(img_path: str) -> str:
    stem = Path(img_path).stem
    # fork_6_aug00 -> fork_6
    m = re.match(r"^(.*)_aug\\d+$", stem)
    if m:
        return m.group(1)
    return stem


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _split_counts(n_groups: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if n_groups <= 0:
        return 0, 0, 0
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))
    # Ensure all groups assigned.
    n_test = n_groups - n_train - n_val

    # Fix edge cases: keep all counts non-negative and sum correct.
    if n_test < 0:
        # Prefer to shrink val then train.
        take = min(n_val, -n_test)
        n_val -= take
        n_test += take
    if n_test < 0:
        take = min(n_train, -n_test)
        n_train -= take
        n_test += take

    return n_train, n_val, n_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_name", default="train.ms_swift.jsonl")
    ap.add_argument("--val_name", default="val.ms_swift.jsonl")
    ap.add_argument("--test_name", default="test.ms_swift.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--group_by_base_image", action="store_true", default=True)
    ap.add_argument("--no_group_by_base_image", dest="group_by_base_image", action="store_false")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_records = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_records += 1
            if args.group_by_base_image:
                key = _group_key_from_image_path(_get_image_path(rec))
            else:
                key = str(total_records)  # unique key per record
            groups[key].append(rec)

    keys = list(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    n_groups = len(keys)
    n_train, n_val, n_test = _split_counts(n_groups, args.train_ratio, args.val_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []
    test_recs: List[Dict[str, Any]] = []

    for k in keys:
        if k in train_keys:
            train_recs.extend(groups[k])
        elif k in val_keys:
            val_recs.extend(groups[k])
        else:
            test_recs.extend(groups[k])

    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name
    test_path = out_dir / args.test_name

    _write_jsonl(train_path, train_recs)
    _write_jsonl(val_path, val_recs)
    _write_jsonl(test_path, test_recs)

    print(f"input_records={total_records} input_groups={n_groups} (group_by_base_image={args.group_by_base_image})")
    print(f"train_groups={len(train_keys)} val_groups={len(val_keys)} test_groups={len(test_keys)}")
    print(f"train_records={len(train_recs)} val_records={len(val_recs)} test_records={len(test_recs)}")
    print(f"train_path={train_path}")
    print(f"val_path={val_path}")
    print(f"test_path={test_path}")


if __name__ == "__main__":
    main()


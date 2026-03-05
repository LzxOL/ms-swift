#!/usr/bin/env python3
"""
Augment an ms-swift grounding jsonl dataset.

This is intentionally "safe": only photometric noise + optional horizontal flip,
so bbox stays correct (flip updates bbox; other transforms keep bbox unchanged).

Typical usage (expand ~108 -> ~648):
  python scripts/augment_grounding_dataset.py \
    --input_jsonl dataset/session_20260128_173739/tomato_fork_train.ms_swift.jsonl \
    --output_jsonl dataset/session_2026023_142020/tomato_fork_train_aug.ms_swift.jsonl \
    --output_image_dir dataset/session_2026023_142020/images_aug \
    --num_aug_per_sample 5 \
    --keep_original
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

_HOST_ROOT = "/data/shengbao.li_home/workspace/ms-swift"
_CONTAINER_ROOT = "/workspace/ms-swift"

# Prompting: for small models, prompt diversity + domain anchoring matters a lot.
DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced agricultural AI assistant specializing in computer vision for robotic pruning. "
    "Your task is to precisely locate the structural forks and axillary buds (suckers) on tomato plants. "
    "Reply with a single tight bounding box for the tomato fork / axillary bud region. "
    "Do not explain your reasoning."
)

# Three tiers + a small pool of negative-constraint prompts.
PROMPTS_TIER1_BASIC = [
    "Detect the branching nodes on the tomato stem.",
    "Locate the Y-shaped junctions formed by the main stem and branches.",
    "Find the bifurcation points of the plant structure.",
    "Identify the structural forks on this tomato plant.",
    "Where does the tomato stem split into branches?",
    "Find the stem fork region on the tomato plant.",
    "Locate the stem junction (fork) on the tomato plant.",
    "Find the plant stem junction where the structure branches.",
    "Locate the branching junction on the tomato stem.",
    "Detect the fork-shaped junction on the stem.",
    "Find the node where the stem divides into two directions.",
    "Identify the branching crotch of the tomato stem.",
    "Locate the stem branching point (junction).",
    "Mark the fork region (branch junction) on the plant.",
]

PROMPTS_TIER2_AGRONOMY = [
    "Locate the axillary buds (suckers) growing between the main stem and the leaf petiole.",
    "Identify the side shoots that emerge from the leaf axils (suckers).",
    "Find the nodes where new vegetative growth is occurring (axillary bud / sucker).",
    "Detect the suckers that should be removed for better fruit growth.",
    "Highlight the intersection between the main stem and the petiole (leaf axil).",
    "Locate the pruning target at the leaf axil (axillary bud).",
    "Find the axillary bud at the leaf axil (between stem and petiole).",
    "Locate the sucker base (origin point) on the tomato stem.",
    "Identify the lateral shoot origin (axillary bud) on the plant.",
    "Mark the leaf axil junction where a sucker emerges.",
    "Detect the tomato side shoot (sucker) emergence node.",
    "Locate the vegetative growth node at the stem-petiole junction.",
    "Find the axillary meristem region (sucker) at the node.",
    "Locate the junction between the main stem and the leaf petiole where suckers appear.",
    "Identify the pruning node (axillary bud) for de-suckering.",
]

PROMPTS_TIER3_ROBOT = [
    "Identify the target zone for the robotic pruning effector.",
    "Locate the precise point where the robotic arm should cut the sucker.",
    "Find the safe intervention location for de-suckering (axillary bud removal).",
    "Detect the target coordinates for robotic pruning preparation.",
    "Show the cutting target on the tomato stem fork.",
    "Mark the robotic cutting/grasping target at the leaf axil.",
    "Find the optimal cut location for robotic sucker removal.",
    "Mark the target region for a pruning tool to remove the sucker.",
    "Locate the robot end-effector target for pruning at the node.",
    "Identify the safe cutting window at the leaf axil for robotic pruning.",
    "Detect the intervention point for a robotic scissor/cutter.",
    "Show the target bounding box for robotic pruning at the stem junction.",
    "Locate the grasp/cut target so the robot can prune the sucker.",
    "Identify where the robot should approach to remove the axillary bud.",
]

PROMPTS_NEGATIVE = [
    "Find the tomato stem forks, ignoring the leaves.",
    "Locate the branching points on the green stem only, ignoring the background.",
    "Identify the pruning points, excluding the main vertical stem.",
    "Find the axillary bud region; ignore support poles, wires, and tools.",
    "Locate the fork region; ignore the pot rim, table, and background text.",
    "Find the axillary bud (sucker) region; do not mark leaves or leaf blades.",
    "Detect the stem junction; ignore robot arms, clamps, and fixtures.",
    "Mark the pruning target; ignore any background crosses and supports.",
]

DOMAIN_DEFINITION_TEMPLATES = [
    "In this task, the target is the tomato axillary bud (sucker) at the leaf axil (stem-petiole junction).",
    "Here, 'tomato fork' refers to the axillary bud/sucker between the main stem and the leaf petiole.",
    "The 'fork' is the plant axil/branching region to be pruned (not a utensil fork).",
    "Focus on the sucker origin at the node where the side shoot emerges from the main stem.",
]

OUTPUT_CONSTRAINT_TEMPLATES = [
    "Return only the bounding box.",
    "Output a single tight bounding box and nothing else.",
    "Reply with exactly one bounding box for the target region.",
]


def _choose_prompt(rng: random.Random) -> str:
    """Sample a prompt from the tiered matrix with roughly 40/30/30 split.

    Also inject a small percentage of negative-constraint prompts.
    """
    parts: List[str] = []

    # ~10% negative prompts to reduce background/leaf confusion.
    if rng.random() < 0.10:
        parts.append(rng.choice(PROMPTS_NEGATIVE))
    else:
        r = rng.random()
        if r < 0.40:
            parts.append(rng.choice(PROMPTS_TIER1_BASIC))
        elif r < 0.70:
            parts.append(rng.choice(PROMPTS_TIER2_AGRONOMY))
        else:
            parts.append(rng.choice(PROMPTS_TIER3_ROBOT))

    # Add explicit domain definition sometimes. This helps 2B disambiguate "fork".
    if rng.random() < 0.35:
        parts.append(rng.choice(DOMAIN_DEFINITION_TEMPLATES))

    # A small portion of samples explicitly constrain output format.
    if rng.random() < 0.15:
        parts.append(rng.choice(OUTPUT_CONSTRAINT_TEMPLATES))

    # Join as short, multi-sentence instruction.
    return " ".join(parts)


def _set_system_and_user_prompt(
    rec: Dict[str, Any],
    user_prompt: str,
    system_prompt: Optional[str],
) -> None:
    """Set/insert system + user messages in ms-swift 'messages' format."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("record has no messages")

    # Ensure a user message exists.
    user_idx = None
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        # Fallback: append a user message.
        user_idx = len(msgs)
        msgs.append({"role": "user", "content": ""})

    # Optionally ensure a system message at the start.
    if system_prompt:
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            msgs[0]["content"] = system_prompt
        else:
            msgs.insert(0, {"role": "system", "content": system_prompt})
            user_idx += 1

    # Always include <image> tag for Qwen3-VL template to place the image.
    msgs[user_idx]["content"] = "<image>" + user_prompt
    rec["messages"] = msgs


def _resolve_image_path(p: str) -> str:
    """Resolve common host<->container path differences."""
    if os.path.exists(p):
        return p

    # Container -> host (common in this project).
    cand = p.replace("/workspace/ms-swift", "/data/shengbao.li_home/workspace/ms-swift", 1)
    if cand != p and os.path.exists(cand):
        return cand

    # Host -> container (reverse).
    cand = p.replace("/data/shengbao.li_home/workspace/ms-swift", "/workspace/ms-swift", 1)
    if cand != p and os.path.exists(cand):
        return cand

    return p  # let caller fail with a useful message


def _to_same_path_style(fs_path: str, ref_path: str) -> str:
    """Convert fs_path to the same absolute path style used by ref_path."""
    if ref_path.startswith(_CONTAINER_ROOT) and fs_path.startswith(_HOST_ROOT):
        return fs_path.replace(_HOST_ROOT, _CONTAINER_ROOT, 1)
    if ref_path.startswith(_HOST_ROOT) and fs_path.startswith(_CONTAINER_ROOT):
        return fs_path.replace(_CONTAINER_ROOT, _HOST_ROOT, 1)
    return fs_path


def _get_image_path_from_record(rec: Dict[str, Any]) -> str:
    images = rec.get("images") or []
    if not images:
        raise ValueError("record has no images")
    img0 = images[0]
    if isinstance(img0, str):
        return img0
    if isinstance(img0, dict) and "path" in img0 and isinstance(img0["path"], str):
        return img0["path"]
    raise ValueError(f"unsupported images[0] type: {type(img0)}")


def _get_bboxes_from_record(rec: Dict[str, Any]) -> List[List[float]]:
    objs = rec.get("objects") or {}
    bboxes = objs.get("bbox") or []
    if not isinstance(bboxes, list):
        raise ValueError("objects.bbox must be a list")
    # Expect list of [x1,y1,x2,y2]
    out: List[List[float]] = []
    for b in bboxes:
        if not (isinstance(b, list) and len(b) == 4):
            raise ValueError(f"bad bbox: {b}")
        out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return out


def _set_bboxes_in_record(rec: Dict[str, Any], bboxes: List[List[float]]) -> None:
    rec.setdefault("objects", {})
    rec["objects"]["bbox"] = bboxes


def _clip_bbox(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    # Ensure order
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _flip_bbox_h(b: List[float], w: int) -> List[float]:
    x1, y1, x2, y2 = b
    return [float(w) - x2, y1, float(w) - x1, y2]


def _apply_photometric_aug(img: Image.Image, rng: random.Random) -> Image.Image:
    # Brightness/contrast/color/sharpness.
    if rng.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.6:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.3:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.70, 1.50))

    # Blur
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.5)))

    # Noise
    if rng.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        sigma = rng.uniform(2.0, 10.0)
        # per-pixel gaussian noise (fast enough for small datasets)
        arr = arr + np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

    # JPEG artifacts
    if rng.random() < 0.25:
        import io

        buf = io.BytesIO()
        q = int(rng.uniform(35, 95))
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def _make_aug(img: Image.Image, bboxes: List[List[float]], rng: random.Random) -> Tuple[Image.Image, List[List[float]]]:
    w, h = img.size
    # Horizontal flip is the only geometric transform here.
    do_flip = rng.random() < 0.5
    if do_flip:
        img = ImageOps.mirror(img)
        bboxes = [_flip_bbox_h(b, w) for b in bboxes]

    img = _apply_photometric_aug(img, rng)
    bboxes = [_clip_bbox(b, w, h) for b in bboxes]
    return img, bboxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--output_image_dir", required=True)
    ap.add_argument("--num_aug_per_sample", type=int, default=5)
    ap.add_argument("--keep_original", action="store_true")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument(
        "--rewrite_original_prompts",
        action="store_true",
        help="If set, also rewrite prompts for the kept original samples.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit (debug)")
    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    out_img_dir = Path(args.output_image_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    base_rng = random.Random(args.seed)

    total_in = 0
    total_out = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total_in += 1
            if args.max_samples and total_in > args.max_samples:
                break

            rec = json.loads(line)

            # Optionally keep original sample.
            if args.keep_original:
                if args.rewrite_original_prompts:
                    # Stable per-record rng (do not depend on augmentation loop).
                    rng = random.Random(args.seed ^ (total_in * 1000003))
                    _set_system_and_user_prompt(rec, _choose_prompt(rng), args.system_prompt)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_out += 1

            img_path = _get_image_path_from_record(rec)
            img_path_fs = _resolve_image_path(img_path)
            if not os.path.exists(img_path_fs):
                raise FileNotFoundError(f"image not found: {img_path} (resolved: {img_path_fs})")

            img = Image.open(img_path_fs).convert("RGB")
            bboxes = _get_bboxes_from_record(rec)

            stem = Path(img_path).stem
            for j in range(args.num_aug_per_sample):
                # Stable-ish per-sample seed
                local_seed = base_rng.randint(0, 2**31 - 1) ^ (total_in * 1000003 + j)
                rng = random.Random(local_seed)

                aug_img, aug_bboxes = _make_aug(img, bboxes, rng)

                out_name = f"{stem}_aug{j:02d}.png"
                out_path = out_img_dir / out_name
                aug_img.save(out_path)

                new_rec = copy.deepcopy(rec)
                # Keep output image path style consistent with the input jsonl (host vs container).
                new_img_path_fs = str(out_path.resolve())
                new_img_path = _to_same_path_style(new_img_path_fs, img_path)
                new_rec["images"] = [new_img_path]
                _set_bboxes_in_record(new_rec, aug_bboxes)
                # Rewrite prompt (diverse & domain-specialized) for augmented samples.
                _set_system_and_user_prompt(new_rec, _choose_prompt(rng), args.system_prompt)

                fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"input_samples={total_in} output_samples={total_out}")
    print(f"output_jsonl={output_jsonl}")
    print(f"output_image_dir={out_img_dir}")


if __name__ == "__main__":
    main()

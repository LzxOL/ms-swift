#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_pred(pred: str):
    pattern = re.compile(
        r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    )
    refs = []
    boxes = []
    for m in pattern.findall(pred or ''):
        refs.append(m[0])
        boxes.append([float(m[1]), float(m[2]), float(m[3]), float(m[4])])
    return refs, boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='preds_val.jsonl or infer_val.jsonl')
    parser.add_argument('--output_dir', required=True, help='dir to save images')
    parser.add_argument('--norm_bbox', default='norm1000', choices=['norm1000', 'none'], help='bbox normalization')
    parser.add_argument('--path_replace', default=None, help='optional path replace: src=dst')
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all')
    args = parser.parse_args()

    from swift.template import draw_bbox, load_image

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            images = row.get('images') or []
            if not images:
                continue
            pred = row.get('pred') or row.get('response', '')
            refs, boxes = parse_pred(pred)
            if not refs or not boxes:
                continue

            image_entry = images[0]
            image_path = image_entry.get('path') if isinstance(image_entry, dict) else image_entry
            if args.path_replace and image_path:
                src, dst = args.path_replace.split('=', 1)
                image_path = image_path.replace(src, dst)
            image = load_image(image_path)
            draw_bbox(image, refs, boxes, norm_bbox=args.norm_bbox)

            stem = Path(image_path).stem
            out_path = output_dir / f'{stem}_pred.png'
            image.save(out_path)

            count += 1
            if args.max_samples and count >= args.max_samples:
                break

    print(f'saved {count} images to {output_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List


def batched(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', required=True, help='LoRA checkpoint path (e.g., output/.../checkpoint-xxx)')
    parser.add_argument('--model', default=None, help='Base model path or model id; if omitted, read from adapters')
    parser.add_argument('--input', required=True, help='Input jsonl (ms-swift grounding format)')
    parser.add_argument('--output', required=True, help='Output jsonl with predictions')
    parser.add_argument('--save_images_dir', default=None, help='If set, save images with predicted bboxes')
    parser.add_argument('--norm_bbox', default='norm1000', choices=['norm1000', 'none'], help='bbox normalization')
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all')
    args = parser.parse_args()

    from swift import TransformersEngine, InferRequest, RequestConfig, BaseArguments
    if args.save_images_dir:
        from swift.template import draw_bbox, load_image
        import re

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

    adapter_path = args.adapters
    if args.model:
        model = args.model
    else:
        base_args = BaseArguments.from_pretrained(adapter_path)
        model = base_args.model

    engine = TransformersEngine(model, adapters=[adapter_path], max_batch_size=args.batch_size)
    request_config = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]

    if args.save_images_dir:
        save_dir = Path(args.save_images_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as out:
        for batch in batched(rows, args.batch_size):
            infer_requests = [InferRequest(**r) for r in batch]
            resp_list = engine.infer(infer_requests, request_config)
            for r, resp in zip(batch, resp_list):
                pred = resp.choices[0].message.content
                if args.save_images_dir:
                    images = r.get('images') or []
                    if images:
                        refs, boxes = parse_pred(pred)
                        if refs and boxes:
                            image = load_image(images[0])
                            draw_bbox(image, refs, boxes, norm_bbox=args.norm_bbox)
                            stem = Path(images[0]).stem
                            out_img = save_dir / f'{stem}_pred.png'
                            image.save(out_img)
                out.write(json.dumps({**r, 'pred': pred}, ensure_ascii=False) + '\n')

    print(f'wrote {output_path} ({len(rows)} samples)')


if __name__ == '__main__':
    main()

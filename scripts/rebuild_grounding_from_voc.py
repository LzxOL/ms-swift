#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_voc_bbox(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find('object')
    if obj is None:
        return None
    bnd = obj.find('bndbox')
    if bnd is None:
        return None
    xmin = float(bnd.findtext('xmin'))
    ymin = float(bnd.findtext('ymin'))
    xmax = float(bnd.findtext('xmax'))
    ymax = float(bnd.findtext('ymax'))
    return [xmin, ymin, xmax, ymax]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='ms-swift jsonl with messages/images/objects')
    parser.add_argument('--annotations', required=True, help='VOC xml folder')
    parser.add_argument('--output', required=True, help='output jsonl')
    args = parser.parse_args()

    in_path = Path(args.input)
    ann_dir = Path(args.annotations)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed = 0
    total = 0
    missing = 0

    with in_path.open('r', encoding='utf-8') as f, out_path.open('w', encoding='utf-8') as out:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            images = row.get('images') or []
            if not images:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                continue
            img = images[0]
            if isinstance(img, dict):
                img = img.get('path')
            if not img:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                continue
            stem = Path(img).stem
            xml_path = ann_dir / f'{stem}.xml'
            if not xml_path.exists():
                missing += 1
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                continue
            bbox = parse_voc_bbox(xml_path)
            if bbox is None:
                missing += 1
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                continue
            # replace bbox
            row.setdefault('objects', {})
            row['objects']['bbox'] = [bbox]
            fixed += 1
            out.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'total={total} fixed={fixed} missing_xml_or_bbox={missing}')


if __name__ == '__main__':
    main()

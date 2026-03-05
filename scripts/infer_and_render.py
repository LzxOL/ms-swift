import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_DIR = "/workspace/ms-swift/output_tomato_fork/merged"
IMAGE_PATH = "/workspace/ms-swift/dataset/session_20260128_173739/images/fork_6.png"
PROMPT = "Highlight the bifurcation points of the tomato stem."
OUTPUT_IMAGE = "/workspace/ms-swift/output_tomato_fork/vis_fork_6.png"
NORM_BBOX = "norm1000"  # set to None if model outputs pixel coords

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

text = processor.batch_decode(outputs, skip_special_tokens=False)[0]

# 提取 bbox
m = re.search(r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>", text)
if not m:
    raise ValueError("未在输出中找到 bbox")

x1, y1, x2, y2 = map(int, m.groups())

if NORM_BBOX == "norm1000":
    w, h = image.size
    x1 = round(x1 / 1000 * w)
    y1 = round(y1 / 1000 * h)
    x2 = round(x2 / 1000 * w)
    y2 = round(y2 / 1000 * h)

print("bbox_px =", (x1, y1, x2, y2))

# 画框保存
vis = image.copy()
draw = ImageDraw.Draw(vis)
draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
vis.save(OUTPUT_IMAGE)
print("saved:", OUTPUT_IMAGE)

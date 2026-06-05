#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script nvidia_vision_model.py
=============================
Este script é um exemplo de como usar o modelo Locate Anything 3B da
Nvidia para localizar um objeto em uma imagem baseado em uma query.

run
---
uv run nvidia_vision_model.py

Example of usage
----------------
uv run nvidia_vision_model.py --image "path/to/image.jpg" --query "Is there a dog in the image?"
"""

import argparse
import os
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_GRID_LOW = 333.3
_GRID_HIGH = 666.6


class LocateAnything3B:
    def __init__(self, model_id="nvidia/LocateAnything-3B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor and model with high efficiency VRAM management:
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.device == "cuda":
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch.float32,
            ).to(self.device)

    def locate_object(self, image: Image.Image, query: str):
        # Format prompt according to LocateAnything chat template
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Detect {query}."}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Run inference
        inputs = self.processor(images=[image.convert("RGB")], text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, tokenizer=self.processor.tokenizer, max_new_tokens=256, do_sample=False, use_cache=True
            )
        res_text = self.processor.batch_decode(
            [o[len(i) :] for i, o in zip(inputs.input_ids, outputs, strict=False)], skip_special_tokens=True
        )[0]

        # Parse output box coordinates: [ymin, xmin, ymax, xmax] (normalized 0 to 1000)
        box_match = re.search(r"<box>(.*?)</box>", res_text, re.IGNORECASE)
        box = [int(x) for x in re.findall(r"\d+", box_match.group(1))] if box_match else [0, 0, 1000, 1000]

        # Calculate center coordinates
        cx = (box[1] + box[3]) / 2.0
        cy = (box[0] + box[2]) / 2.0

        # Determine 3x3 grid sector based on coordinates
        row = "TOP" if cy < _GRID_LOW else "MIDDLE" if cy < _GRID_HIGH else "BOTTOM"
        col = "LEFT" if cx < _GRID_LOW else "CENTER" if cx < _GRID_HIGH else "RIGHT"
        sector = f"{row}-{col}"

        return {
            "query": query,
            "response_text": res_text.strip(),
            "box_normalized": box,
            "center_normalized": (cx, cy),
            "sector": sector,
            "instruction": f"The requested {query} is in the {sector} of your view.",
        }


def main():
    parser = argparse.ArgumentParser(description="Locate Anything 3B")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--query", type=str, required=True, help="Target text query to locate")
    args = parser.parse_args()

    # Ensure the input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image file '{args.image}' not found.")
        return

    # Print the single console status message
    print("Generating final results..")

    # Initialize LocateAnything3B and run local spatial detection
    try:
        nvidia_model = LocateAnything3B()
        pil_img = Image.open(args.image)
        res = nvidia_model.locate_object(pil_img, args.query)

        # Write outputs to markdown file
        md_content = (
            "# Locate Anything 3B Test Results\n\n"
            "| Image ID | Query | Box | Center | Sector | Instruction |\n"
            "|---|---|---|---|---|---|\n"
            f"| {os.path.basename(args.image)} | {res['query']}"
            f" | `{res['box_normalized']}` | `{res['center_normalized']}`"
            f" | **{res['sector']}** | {res['instruction']} |\n"
        )
        with open("test_results.md", "w") as f:
            f.write(md_content)

    except Exception as e:
        print(f"Error processing visual grounding: {e}")


if __name__ == "__main__":
    main()

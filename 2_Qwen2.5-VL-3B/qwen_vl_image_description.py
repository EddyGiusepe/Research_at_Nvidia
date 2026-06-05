#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script qwen_vl_image_description.py
====================================
Usa o modelo Qwen2.5-VL-3B-Instruct (VLM generalista da família Qwen,
backbone do LocateAnything-3B) para gerar descrições livres de imagens
em português brasileiro a partir de um prompt do usuário.

run
---
uv run qwen_vl_image_description.py
"""

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_model(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    return processor, model


def describe_image(model, processor, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids, strict=False)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    image_path = "../1_LocateAnything-3B/a6d20c83d4ca445eb28fb1516bbe397c.jpg"
    prompt = "Liste os objetos presentes na imagem."

    processor, model = load_model()
    description = describe_image(model, processor, image_path, prompt)
    print("Descrição:\n", description)


if __name__ == "__main__":
    main()

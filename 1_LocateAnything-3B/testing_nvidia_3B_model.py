#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script testing_nvidia_3B_model.py
==================================
O modelo LocateAnything-3B da NVIDIA é um modelo multimodal
de visão-linguagem voltado para visual grounding e detecção,
capaz de identificar, localizar e retornar caixas delimitadoras
(bounding boxes) para objetos ou regiões em imagens com base
em prompts de texto. Quando utilizado, o modelo recebe uma
imagem e uma descrição textual (prompt) do que deve ser
localizado e retorna como saída as coordenadas das caixas
delimitadoras encontradas, podendo também incluir informações
textuais adicionais relacionadas à localização dos objetos requisitados.

run
---
uv run testing_nvidia_3B_model.py
"""

import re

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer


def load_model(model_id: str = "nvidia/LocateAnything-3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    return tokenizer, processor, model


def build_prompt(processor, query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": query}],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_inference(model, processor, tokenizer, image: Image.Image, query: str) -> str:
    prompt = build_prompt(processor, query)
    inputs = processor(images=[image], text=prompt, return_tensors="pt")
    return model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        pixel_values=inputs["pixel_values"].to(model.device),
        image_grid_hws=torch.tensor(inputs["image_grid_hws"]).to(model.device),
        tokenizer=tokenizer,
        use_cache=True,
        max_new_tokens=64,
    )


def parse_boxes(text: str) -> list:
    boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text)
    boxes = [tuple(map(int, b)) for b in boxes]

    seen = set()
    unique = []
    for b in boxes:
        if b not in seen:
            seen.add(b)
            unique.append(b)

    return [b for b in unique if (b[2] - b[0]) * (b[3] - b[1]) <= 0.8 * 1000 * 1000]


def draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h],
            outline="red",
            width=2,
        )
    return image


def main():
    #image_path = "./53737492-fofa-cachorro-e-gatinho-partilha-uma-aconchegar-se-em-uma-acolhedor-cama-foto.jpg"
    image_path = "./a6d20c83d4ca445eb28fb1516bbe397c.jpg"
    query = "O que contém a imagem?"
    output_path = "result_with_boxes.jpg"

    tokenizer, processor, model = load_model()
    image = Image.open(image_path).convert("RGB")

    decoded = run_inference(model, processor, tokenizer, image, query)
    print("Resposta do modelo:", decoded)

    boxes = parse_boxes(decoded)
    result = draw_boxes(image, boxes)
    result.save(output_path)
    print(f"Imagem salva em {output_path}")


if __name__ == "__main__":
    main()

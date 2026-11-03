"""Carregamento do modelo (4 bits, com o adaptador treinado) e geração, iguais no treino e no uso."""
import math

import torch
from PIL import Image

from common import BASE_MODEL, KEY_TOKENS, OCR_PIXELS, PAGE_PIXELS, PAGE_TOKENS


def fit(img, pixels):
    """Redimensiona para a área ficar dentro de `pixels` (mín., máx.) com lados múltiplos de 28.

    É o que o Qwen2-VL espera; o processor então usa a imagem como está.
    """
    lo, hi = pixels
    w, h = img.size
    area = w * h
    scale = math.sqrt(hi / area) if area > hi else math.sqrt(lo / area) if area < lo else 1.0
    nw, nh = max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28)
    while nw * nh > hi:
        nw, nh = (nw - 28, nh) if nw >= nh else (nw, nh - 28)
    return img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.BICUBIC)


def messages(prompt, answer=None, images=1):
    out = [{"role": "user", "content": [{"type": "image"}] * images + [{"type": "text", "text": prompt}]}]
    if answer is not None:
        out.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return out


def load_processor(base=BASE_MODEL):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(base, min_pixels=OCR_PIXELS[0], max_pixels=28 * 28 * max(KEY_TOKENS, PAGE_TOKENS, PAGE_PIXELS[1] // 784))


def load_model(base=BASE_MODEL, adapter=None):
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base, quantization_config=quant, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def request_images(images, page_tokens=PAGE_TOKENS):
    """Imagens de um pedido de tradução (páginas primeiro, chave por último) no tamanho usado no treino.

    A chave fica com KEY_TOKENS para as letras continuarem legíveis; as páginas dividem `page_tokens`,
    que é menor nos pedidos longos demais para a placa.
    """
    *pages, key = [img.convert("RGB") for img in images]
    out = [fit(p, (28 * 28 * 64, 28 * 28 * min(KEY_TOKENS, page_tokens // len(pages)))) for p in pages]
    return out + [fit(key, (28 * 28 * 64, 28 * 28 * KEY_TOKENS))]


@torch.no_grad()
def ask(model, processor, img, prompt, pixels=None, max_new_tokens=256):
    """`img` é uma imagem (ajustada a `pixels`) ou a lista de imagens de um pedido de tradução."""
    imgs = request_images(img) if isinstance(img, list) else [fit(img.convert("RGB"), pixels)]
    text = processor.apply_chat_template(messages(prompt, images=len(imgs)), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

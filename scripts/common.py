"""Constantes e funções compartilhadas: caminhos, modelo base, prompts e orçamentos de imagem."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# Tamanhos de imagem em pixels. No Qwen2-VL cada bloco de 28x28 vira um token.
OCR_PIXELS = (28 * 28 * 16, 28 * 28 * 192)
PAGE_PIXELS = (28 * 28 * 256, 28 * 28 * 640)

# Pedidos de tradução do Manga Index (scripts/train.py). A imagem-chave (as caixas recortadas) fica com até
# KEY_TOKENS; a página e suas fatias dividem PAGE_TOKENS. Um pedido acima de MAX_TOKENS recebe páginas
# menores (o primeiro de PAGE_TOKENS_SHRUNK que couber) e só fica de fora se nenhum couber. Numa placa de
# 6 GB o treino estoura perto de 3,9k tokens (scripts/memory_probe.py).
KEY_TOKENS = 640
PAGE_TOKENS = 1024
PAGE_TOKENS_SHRUNK = (768, 512, 256)
MAX_TOKENS = 3500

OCR_PROMPT = "Transcribe the English text in this image exactly, as one line. Answer with nothing if there is none."


def narrate_prompt(lines):
    """Prompt da tarefa de página: as falas lidas, em ordem, e o pedido de prosa."""
    if lines:
        listed = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
        said = f"Falas da página, na ordem de leitura:\n{listed}\n\n"
    else:
        said = "A página não tem falas.\n\n"
    return (
        f"{said}Reescreva esta página de mangá como um trecho de light novel em português do Brasil, "
        "com as falas traduzidas."
    )


def reading_order(boxes, frames=None, rtl=True):
    """Índices de `boxes` ([x0, y0, x1, y1]) na ordem de leitura.

    Quadros de cima para baixo e da direita para a esquerda (rtl=False para quadrinhos e manhwa); dentro
    de cada quadro, a mesma regra. Sem quadros, a página inteira conta como um só.
    """
    frames = frames or []

    def centre(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def frame_of(b):
        cx, cy = centre(b)
        for k, f in enumerate(frames):
            if f[0] <= cx <= f[2] and f[1] <= cy <= f[3]:
                return k
        return -1

    order_frames = {k: rank for rank, k in enumerate(band_order(frames, rtl))}
    groups = {}
    for i, b in enumerate(boxes):
        groups.setdefault(order_frames.get(frame_of(b), len(frames)), []).append(i)
    out = []
    for rank in sorted(groups):
        idx = groups[rank]
        out += [idx[j] for j in band_order([boxes[i] for i in idx], rtl)]
    return out


def band_order(boxes, rtl=True):
    """Agrupa em linhas as caixas que dividem a maior parte da altura; linhas de cima para baixo."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1])
    rows = []
    for i in order:
        b = boxes[i]
        for row in rows:
            top, bottom = row["top"], row["bottom"]
            if min(bottom, b[3]) - max(top, b[1]) > 0.5 * min(bottom - top, b[3] - b[1]):
                row["items"].append(i)
                row["top"], row["bottom"] = min(top, b[1]), max(bottom, b[3])
                break
        else:
            rows.append({"top": b[1], "bottom": b[3], "items": [i]})
    out = []
    for row in rows:
        out += sorted(row["items"], key=lambda i: (boxes[i][0] + boxes[i][2]) * (-1 if rtl else 1))
    return out

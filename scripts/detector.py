"""Detector de texto do Manga Index (scripts/manga_vision.py de lá), sem o modelo de inpainting."""
import os
import sys
from pathlib import Path

import numpy as np

from common import reading_order

MANGA_INDEX = Path(os.environ.get("MANGA_INDEX", Path(r"D:\Dev\Manga Index")))
PAD = 6
# Tiras de webtoon são cortadas em pedaços com esta proporção (altura / largura), cada um lido como página.
PAGE_ASPECT = 1.5


def load_detector(manga_index=MANGA_INDEX):
    sys.path.insert(0, str(manga_index / "scripts"))
    import manga_vision as mv
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    v = mv.Vision.__new__(mv.Vision)
    v.torch = torch
    v.device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = (
        [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
        if v.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    v.run_options = ort.RunOptions()
    if v.device == "cuda":
        v.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    models = mv.MODELS
    v.ctd = ort.InferenceSession(str(models / "comictextdetector.pt.onnx"), providers=providers)
    v.bubble = YOLO(str(models / "bubble-seg.pt"))
    kinds = models / mv.KINDS_MODEL
    v.kinds = ort.InferenceSession(str(kinds), providers=providers) if kinds.exists() else None
    return v


def segments(img, boxes):
    """Faixas [(y0, y1)] de uma tira longa; uma página normal vira uma faixa só.

    O corte cai na linha mais uniforme (a sarjeta entre quadros) que não atravessa nenhum bloco de texto.
    """
    W, H = img.size
    target = int(W * PAGE_ASPECT)
    if H <= target * 1.3:
        return [(0, H)]
    gray = np.asarray(img.convert("L"), np.float32)
    flatness = gray.std(1)
    blocked = np.zeros(H, bool)
    for _, y0, _, y1 in boxes:
        blocked[y0:y1] = True
    out, top = [], 0
    while H - top > target * 1.3:
        lo, hi = top + int(target * 0.6), min(H - int(target * 0.3), top + int(target * 1.4))
        rows = np.arange(lo, hi)
        free = rows[~blocked[lo:hi]]
        if not len(free):
            free = rows
        # Empate de uniformidade: vence a linha mais perto da altura alvo.
        score = flatness[free] + np.abs(free - (top + target)) / target
        cut = int(free[score.argmin()])
        out.append((top, cut))
        top = cut
    out.append((top, H))
    return out


def text_boxes(detector, img, rtl=True):
    """Blocos de texto da página (imagem PIL) com margem, na ordem de leitura."""
    blocks, _, _ = detector.detect(np.asarray(img.convert("RGB")))
    detector.release()
    W, H = img.size
    boxes = [b["box"] for b in blocks]
    return [
        [max(0, x0 - PAD), max(0, y0 - PAD), min(W, x1 + PAD), min(H, y1 + PAD)]
        for x0, y0, x1, y1 in (boxes[i] for i in reading_order(boxes, rtl=rtl))
    ]

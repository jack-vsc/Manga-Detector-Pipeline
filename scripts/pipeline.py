"""Página de mangá em inglês na entrada, trecho de light novel em português na saída.

1. O detector do Manga Index acha os blocos de texto e os põe em ordem de leitura.
2. O modelo treinado lê cada bloco.
3. O mesmo modelo recebe a página e as falas e escreve a prosa.

Imprime a prosa e grava <página>.json (falas, caixas, prosa) ao lado de cada página, ou em --out.

Uso: python scripts/pipeline.py pagina.jpg [outras...] --adapter runs/manga-ln/final
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from common import OCR_PIXELS, OCR_PROMPT, PAGE_PIXELS, ROOT, narrate_prompt
from detector import MANGA_INDEX, load_detector, segments, text_boxes
from model import ask, load_model, load_processor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pages", nargs="+", type=Path)
    p.add_argument("--adapter", default=str(ROOT / "runs" / "manga-ln" / "final"), help="adaptador treinado ('' = modelo base)")
    p.add_argument("--ltr", action="store_true", help="leitura da esquerda para a direita (quadrinhos, manhwa)")
    p.add_argument("--manga-index", type=Path, default=MANGA_INDEX)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def read_page(detector, model, processor, path, rtl=True):
    """Uma página, ou uma tira de webtoon pedaço a pedaço: as falas lidas e a prosa."""
    img = Image.open(path).convert("RGB")
    boxes = text_boxes(detector, img, rtl)
    lines, prose = [], []
    for y0, y1 in segments(img, boxes):
        mine = []
        for box in boxes:
            if y0 <= (box[1] + box[3]) / 2 < y1:
                text = " ".join(ask(model, processor, img.crop(box), OCR_PROMPT, OCR_PIXELS, max_new_tokens=96).split())
                if text:
                    mine.append({"box": box, "text": text})
        piece = img.crop((0, y0, img.width, y1))
        prose.append(ask(model, processor, piece, narrate_prompt([l["text"] for l in mine]), PAGE_PIXELS, max_new_tokens=768))
        lines += mine
    return {"page": str(path), "lines": lines, "prose": "\n\n".join(prose)}


def main():
    args = parse_args()
    detector = load_detector(args.manga_index)
    processor = load_processor()
    model = load_model(adapter=args.adapter or None)
    for path in args.pages:
        result = read_page(detector, model, processor, path, rtl=not args.ltr)
        out_dir = args.out or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== {path.name} ({len(result['lines'])} falas)\n{result['prose']}")


if __name__ == "__main__":
    main()

"""Lista as páginas de treino com os blocos de texto do detector do Manga Index, em ordem de leitura.

Cada pasta que contém imagens é um capítulo, e capítulos vão inteiros para treino ou validação. Tiras de
webtoon são cortadas em pedaços do tamanho de uma página. Grava data/pages.jsonl; rodar de novo só
acrescenta as páginas que faltam.

Uso: python scripts/prepare_pages.py <pasta> [<pasta> ...] [--ltr]
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from common import DATA
from detector import MANGA_INDEX, load_detector, segments, text_boxes

IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--val-every", type=int, default=10, help="um a cada N capítulos vai para validação")
    p.add_argument("--ltr", action="store_true", help="leitura da esquerda para a direita (quadrinhos, manhwa)")
    p.add_argument("--manga-index", type=Path, default=MANGA_INDEX)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = DATA / "pages.jsonl"
    DATA.mkdir(parents=True, exist_ok=True)
    known = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            known = {json.loads(line)["image"] for line in f if line.strip()}
    chapters = sorted({p.parent for folder in args.folders for p in folder.rglob("*") if p.suffix.lower() in IMAGES})
    detector = load_detector(args.manga_index)
    added = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for n, chapter in enumerate(chapters):
            split = "val" if n % args.val_every == args.val_every - 1 else "train"
            for page in sorted(p for p in chapter.iterdir() if p.suffix.lower() in IMAGES):
                if str(page) in known:
                    continue
                with Image.open(page) as img:
                    boxes = text_boxes(detector, img, rtl=not args.ltr)
                    pieces = segments(img, boxes)
                    width = img.width
                for k, (y0, y1) in enumerate(pieces):
                    record = {
                        "id": f"{chapter.parent.name}/{chapter.name}/{page.stem}" + (f"-{k}" if len(pieces) > 1 else ""),
                        "split": split,
                        "image": str(page),
                        "crop": [0, y0, width, y1],
                        "boxes": [b for b in boxes if y0 <= (b[1] + b[3]) / 2 < y1],
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    added += 1
            print(f"[{n + 1}/{len(chapters)}] {chapter} ({split})")
    print(f"{added} páginas acrescentadas em {out_path}")


if __name__ == "__main__":
    main()

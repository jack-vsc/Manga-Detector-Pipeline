"""Tira capítulos do dataset de treino do Manga Index (uma novel, um capítulo traduzido por engano).

Apaga as anotações de cada capítulo, as imagens usadas só por ele e seus pedidos em recorded.txt, para que
o mesmo pedido traduzido de novo depois volte a ser gravado. Pare o replay antes: ele escreve em
recorded.txt.

Uso: python scripts/dataset_remove.py "<capítulo>" ["<capítulo>" ...]
"""
import argparse
import json

from dataset_stats import DATASET


def records(file):
    with open(file, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chapters", nargs="+", help="nomes como aparecem no painel")
    args = p.parse_args()
    notes = DATASET / "annotations"
    gone = [notes / f"{name}.jsonl" for name in args.chapters]
    for file in gone:
        if not file.exists():
            p.error(f"não está no dataset: {file.stem}")
    keys, images = set(), set()
    for file in gone:
        for r in records(file):
            keys.add(r["key"])
            images |= {i["file"] for i in r["images"]}
    for file in notes.glob("*.jsonl"):
        if file not in gone:
            for r in records(file):
                images -= {i["file"] for i in r["images"]}
    for name in images:
        (DATASET / "images" / name).unlink(missing_ok=True)
    recorded = DATASET / "recorded.txt"
    kept = [k for k in recorded.read_text(encoding="utf-8").split() if k not in keys]
    recorded.write_text("".join(k + "\n" for k in kept), encoding="utf-8")
    for file in gone:
        file.unlink()
    print(f"{len(gone)} capítulos, {len(keys)} pedidos e {len(images)} imagens removidos")


if __name__ == "__main__":
    main()

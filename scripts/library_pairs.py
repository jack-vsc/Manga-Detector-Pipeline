"""Obras da biblioteca do Manga Index que têm os mesmos capítulos em inglês e em português.

Capítulos baixados sem tradução recebem o idioma de destino como rótulo, então o idioma vem da fonte: o
site em PT_HOST (variável de ambiente) é português e qualquer outro é inglês.

Uso: python scripts/library_pairs.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

LIB = Path(os.environ["APPDATA"]) / "Manga Index" / "library"
PT_HOST = os.environ.get("PT_HOST", "mediocrescan")


def ranges(nums):
    nums = sorted({n for n in nums if n is not None})
    out, start = [], None
    for i, n in enumerate(nums):
        start = n if start is None else start
        if i + 1 == len(nums) or nums[i + 1] > n + 1:
            out.append(f"{start:g}" if start == n else f"{start:g}-{n:g}")
            start = None
    return ",".join(out) or "-"


def main():
    works = {w["id"]: w for w in json.loads((LIB / "works.json").read_text(encoding="utf-8"))["works"]}
    by = defaultdict(lambda: {"en": set(), "pt": set()})
    for meta in LIB.glob("*/meta.json"):
        d = json.loads(meta.read_text(encoding="utf-8"))
        host = urlparse(d.get("source") or "").netloc
        by[d.get("workId")]["pt" if PT_HOST in host else "en"].add(d.get("chapter"))
    rows = []
    for wid, w in by.items():
        name = works.get(wid, {}).get("names", ["?"])[0]
        pairs = w["en"] & w["pt"]
        rows.append((len(pairs), name, w))
    for n, name, w in sorted(rows, key=lambda r: (-r[0], r[1])):
        print(f"{name[:48]:48} | EN {len(w['en']):3} [{ranges(w['en'])}] | PT {len(w['pt']):3} [{ranges(w['pt'])}] | pares {n} [{ranges(w['en'] & w['pt'])}]")
    print(f"total de pares: {sum(r[0] for r in rows)}")


if __name__ == "__main__":
    main()

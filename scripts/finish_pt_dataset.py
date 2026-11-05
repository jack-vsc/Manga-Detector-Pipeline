"""Transforma dataset-pt/data (gravado por build_pt_dataset.py) no dataset de treino dataset-pt/final.

Cada resposta fica só com o objeto JSON: uma nota que o Claude às vezes escrevia antes dele ensinaria o
modelo pequeno a conversar. Respostas sem JSON legível ficam de fora. As imagens são compartilhadas com
dataset-pt/data por uma junction, sem cópia. Grava dataset-pt/final/annotations/<capítulo>.jsonl e
dataset-pt/final/stats.json.

Uso: python scripts/finish_pt_dataset.py
"""
import json
import subprocess
from collections import Counter
from pathlib import Path

from common import ROOT

SRC = ROOT / "dataset-pt" / "data"
DST = ROOT / "dataset-pt" / "final"


def clean(answer):
    """(JSON compacto, objeto) da resposta, ou None."""
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(answer[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list):
        return None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")), data


def main():
    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    if not (DST / "images").exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(DST / "images"), str(SRC / "images")], check=True, capture_output=True)
    stats = Counter()
    works = Counter()
    for file in sorted((SRC / "annotations").glob("*.jsonl")):
        kept = []
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            stats["requests"] += 1
            stats[r["kind"]] += 1
            got = clean(r["answer"])
            if got is None:
                stats["dropped_unreadable"] += 1
                continue
            text, data = got
            if not r["answer"].lstrip().startswith(("{", "```")):
                stats["note_removed"] += 1
            lines = [b for b in data["blocks"] if isinstance(b, dict) and b.get("translation")]
            stats["lines"] += len(lines)
            if r["kind"] == "followup" and not lines:
                stats["followup_empty"] += 1
            r["answer"] = text
            kept.append(json.dumps(r, ensure_ascii=False))
        (DST / "annotations" / file.name).write_text("\n".join(kept) + "\n", encoding="utf-8")
        stats["chapters"] += 1
        works[file.stem.rsplit(" Chapter", 1)[0]] += 1
        stats["kept"] += len(kept)
    out = {**stats, "works": dict(works)}
    (DST / "stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

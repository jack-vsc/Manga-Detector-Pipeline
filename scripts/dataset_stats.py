"""Resumo do dataset de treino do Manga Index, usado pelo painel.

Grava em <pasta> chapters.json (uma linha por capítulo), timeline.json (páginas por janela de 10 minutos,
com o total acumulado) e totals.json (disco, imagens, meta de páginas). scripts/dataset_panel.py serve os
mesmos dados ao vivo.

Uso: python scripts/dataset_stats.py <pasta> [--goal 5000]
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# PANEL_DATASET aponta para outra pasta (ex.: dataset-pt/data enquanto build_pt_dataset.py roda).
DATASET = Path(os.environ.get("PANEL_DATASET") or Path(os.environ["APPDATA"]) / "Manga Index" / "training-dataset")

# Cache por capítulo, relido só quando o arquivo muda.
_chapters = {}


def balloons(answer):
    """Blocos que receberam tradução (créditos e marcas d'água pulados voltam vazios)."""
    try:
        blocks = json.loads(answer).get("blocks", [])
    except (ValueError, AttributeError):
        return 0
    return sum(1 for b in blocks if isinstance(b, dict) and (b.get("translation") or "").strip())


def read_chapter(file):
    stamp = file.stat().st_mtime_ns
    cached = _chapters.get(file)
    if cached and cached[0] == stamp:
        return cached[1]
    pages, requests, said, last = {}, 0, 0, ""
    with open(file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            requests += 1
            said += balloons(r["answer"])
            last = max(last, r["at"])
            if r["kind"] == "page":
                pages.setdefault(r["page"], r["at"])
    summary = {"chapter": file.stem, "pages": len(pages), "requests": requests, "balloons": said, "last_at": last, "page_times": list(pages.values())}
    _chapters[file] = (stamp, summary)
    return summary


def compute(goal=5000):
    """{chapters, timeline, totals} no formato do painel."""
    chapters, slots = [], Counter()
    for file in sorted((DATASET / "annotations").glob("*.jsonl")):
        c = read_chapter(file)
        for at in c["page_times"]:
            slots[at[:15] + "0:00Z"] += 1
        chapters.append({k: v for k, v in c.items() if k != "page_times"})
    timeline, total = [], 0
    for slot in sorted(slots):
        total += slots[slot]
        timeline.append({"hour": slot, "pages": slots[slot], "total": total})
    files = [p for p in DATASET.rglob("*") if p.is_file()] if DATASET.exists() else []
    totals = {
        "goal_pages": goal,
        "disk_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1),
        "images": sum(1 for p in files if p.suffix == ".jpg"),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {"chapters": chapters, "timeline": timeline, "totals": totals}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--goal", type=int, default=5000, help="páginas em que o replay para")
    args = p.parse_args()
    stats = compute(args.goal)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("chapters", "timeline"):
        (args.out / f"{name}.json").write_text(json.dumps(stats[name], ensure_ascii=False), encoding="utf-8")
    (args.out / "totals.json").write_text(json.dumps([stats["totals"]]), encoding="utf-8")
    pages = stats["timeline"][-1]["total"] if stats["timeline"] else 0
    print(f"{len(stats['chapters'])} capítulos, {pages} páginas, {stats['totals']['disk_mb']} MB")


if __name__ == "__main__":
    main()

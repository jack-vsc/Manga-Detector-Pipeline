"""Respostas do Claude e dos modelos locais (compare_models.py) lado a lado, caixa por caixa, numa página.

Grava <dataset>/../results/comparativo.html, que mostra as imagens da pasta do dataset ao lado.

Uso: python scripts/compare_report.py [--dataset test-soccer/dataset]
"""
import argparse
import difflib
import html
import json
import re
from pathlib import Path

from common import ROOT
from compare_models import load_requests

# Os modelos locais rodam do mesmo jeito: GGUF no llama.cpp (o Ollama também é llama.cpp), contexto de 16k.
# "qwen" é o mesmo Qwen rodando no PyTorch (compare_models.py qwen), mantido como registro.
MODELS = [("claude", "Claude Haiku"), ("gemma", "Gemma 4 E4B (Ollama)"), ("gemma-e2b", "Gemma 4 E2B (Ollama)"),
          ("qwen-gguf-q8", "Qwen treinado (llama.cpp, Q8_0)"), ("qwen-gguf-fit", "Qwen treinado (llama.cpp, imagens do treino)"),
          ("qwen-gguf", "Qwen treinado (llama.cpp, imagens inteiras)"),
          ("qwen", "Qwen treinado (PyTorch)"),
          ("qwen2-vl-2b", "Qwen2-VL-2B original"), ("qwen25-vl-3b", "Qwen2.5-VL-3B original"),
          ("qwen3-vl-2b", "Qwen3-VL-2B original"), ("qwen3-vl-4b", "Qwen3-VL-4B original"),
          ("qwen35-2b", "Qwen3.5-2B original"), ("qwen35-4b", "Qwen3.5-4B original"),
          ("qwen35-2b-trained", "Qwen3.5-2B treinado"),
          ("qwen35-2b-trained-v2", "Qwen3.5-2B treinado v2")]


def parse(text):
    """O objeto JSON de uma resposta, ou None se não houver um legível."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and isinstance(data.get("blocks"), list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def ids_of(block):
    """Números das caixas de um bloco; um modelo pode escrever um número em vez de uma lista."""
    ids = block.get("ids")
    ids = ids if isinstance(ids, list) else [ids] if ids is not None else []
    return {i for i in ids if isinstance(i, int)}


def box_ids(answer):
    blocks = [b for b in answer["blocks"] if isinstance(b, dict)]
    said = set().union(*(ids_of(b) for b in blocks))
    skip = answer.get("skip")
    skipped = {i for i in (skip if isinstance(skip, list) else []) if isinstance(i, int)}
    return blocks, said, skipped


def stats_for(name, requests, answers):
    """Por modelo: respostas quebradas e, das falas que o Claude traduziu (diálogo, legendas, placas; sem
    onomatopeias nem marcas d'água), as perdidas e as muito diferentes (texto de outra caixa ou outro
    sentido)."""
    s = {"requests": 0, "broken": 0, "lines": 0, "lost": 0, "close": 0, "far": 0, "seconds": [], "gpu": 0}
    for r in requests:
        claude = parse(r["answer"])
        a = answers.get(name, {}).get(r["key"]) if name != "claude" else {"answer": r["answer"]}
        if not claude or a is None:
            continue
        s["requests"] += 1
        if name != "claude":
            s["seconds"].append(a["seconds"])
            s["gpu"] = max(s["gpu"], a.get("gpu_mb") or 0)
        mine = parse(a["answer"])
        s["broken"] += mine is None
        blocks = box_ids(mine)[0] if mine else []
        for b in box_ids(claude)[0]:
            said = str(b.get("translation", "")).strip().lower()
            if b.get("kind") in ("sfx", "watermark", "skip") or not said or not ids_of(b):
                continue
            s["lines"] += 1
            hit = " ".join(str(x.get("translation", "")).strip().lower() for x in blocks if ids_of(x) & ids_of(b))
            if not hit:
                s["lost"] += 1
                continue
            likeness = difflib.SequenceMatcher(None, said, hit).ratio()
            s["close"] += likeness >= 0.7
            s["far"] += likeness < 0.35
    return s


def cell(blocks):
    if blocks is None:
        return '<td class="none">—</td>'
    if not blocks:
        return '<td class="miss">não traduziu</td>'
    return "<td>" + "<br>".join(html.escape(str(b.get("translation", ""))) or "<i>(vazio)</i>" for b in blocks) + "</td>"


def rows_for(r, answers):
    claude = parse(r["answer"])
    if not claude:
        return ""
    others = {}
    for name, _ in MODELS[1:]:
        a = answers.get(name, {}).get(r["key"])
        others[name] = None if a is None else parse(a["answer"]) or "broken"
    out = []
    claude_blocks, _, _ = box_ids(claude)
    for b in claude_blocks:
        ids = ids_of(b)
        row = [f'<td class="orig">{html.escape(str(b.get("original", "")))}<div class="ids">{" ".join(f"#{i}" for i in sorted(ids)) or "sem caixa"}</div></td>',
               f'<td>{html.escape(str(b.get("translation", "")))}</td>']
        for name, _ in MODELS[1:]:
            o = others[name]
            if o is None:
                row.append(cell(None))
            elif o == "broken":
                row.append('<td class="miss">resposta quebrada</td>')
            elif ids:
                blocks, _, skipped = box_ids(o)
                hit = [x for x in blocks if ids & ids_of(x)]
                row.append(cell(hit) if hit or not ids <= skipped else '<td class="miss">marcou como ignorar</td>')
            else:
                row.append('<td class="none">(texto sem caixa: veja abaixo)</td>')
        out.append("<tr>" + "".join(row) + "</tr>")
    # O que os modelos locais escreveram sem par no Claude: texto que acharam fora das caixas.
    for name, label in MODELS[1:]:
        o = others[name]
        if isinstance(o, dict):
            claude_ids = set().union(*(ids_of(b) for b in claude_blocks))
            extra = [x for x in box_ids(o)[0] if not ids_of(x) & claude_ids]
            if extra:
                texts = "; ".join(html.escape(f'{x.get("original", "")} → {x.get("translation", "")}') for x in extra)
                out.append(f'<tr class="extra"><td colspan="{len(MODELS) + 1}"><b>{label}, a mais:</b> {texts}</td></tr>')
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="test-soccer/dataset")
    args = p.parse_args()
    dataset = ROOT / args.dataset
    results = dataset.parent / "results"
    requests = load_requests(dataset)
    answers = {}
    for name, _ in MODELS[1:]:
        f = results / f"{name}.jsonl"
        if f.exists():
            answers[name] = {(d := json.loads(l))["key"]: d for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}

    head = "".join(f"<th>{label}</th>" for _, label in MODELS)
    summary = []
    for name, label in MODELS:
        s = stats_for(name, requests, answers)
        if not s["requests"]:
            continue
        secs = s["seconds"]
        gpu = f"{s['gpu'] / 1024:.1f} GB" if s["gpu"] else "—"
        summary.append(
            f"<tr><td>{label}</td><td>{s['requests']}</td><td>{s['broken']}</td>"
            f"<td>{s['lost']} de {s['lines']}</td><td>{s['close']}</td><td>{s['far']}</td>"
            f"<td>{f'{sum(secs) / len(secs):.0f} s' if secs else 'API'}</td>"
            f"<td>{gpu}</td></tr>")

    pages = []
    for r in requests:
        rows = rows_for(r, answers)
        if not rows:
            continue
        imgs = "".join(f'<a href="../dataset/images/{i["file"]}" target="_blank"><img src="../dataset/images/{i["file"]}" loading="lazy"></a>'
                       for i in r["images"])
        pages.append(f'<section><h2>{html.escape(r["chapter"])} · página {r["page"]}'
                     f'{" · " + html.escape(r["kind"]) if r.get("kind") != "page" else ""}</h2>'
                     f'<div class="imgs">{imgs}</div>'
                     f'<div class="t"><table><tr><th>Original</th>{head}</tr>{rows}</table></div></section>')

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Comparativo de tradução</title>
<style>
:root {{ --bg:#f6f5f1; --card:#fff; --ink:#1d1d1b; --muted:#6b6a65; --line:#e4e2db; --warn:#a3542b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151614; --card:#1e201d; --ink:#ecebe6; --muted:#9a9993; --line:#30322e; --warn:#e0946a; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,"Segoe UI",sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:15px; margin:0 0 10px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-top:16px; }}
.imgs {{ display:flex; gap:8px; overflow-x:auto; margin-bottom:10px; }} .imgs img {{ height:260px; border-radius:6px; }}
.t {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; }}
th, td {{ text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:500; font-size:12px; }}
td.orig {{ color:var(--muted); max-width:260px; }} .ids {{ font-size:11px; }}
td.miss {{ color:var(--warn); font-weight:600; }} td.none {{ color:var(--muted); }}
tr.extra td {{ font-size:12px; color:var(--muted); }}
</style></head><body><main>
<h1>Comparativo de tradução: Claude × Gemma 4 E4B × Qwen treinado</h1>
<p style="color:var(--muted)">Gemma e Qwen rodam do mesmo jeito: arquivos GGUF de 4 bits no llama.cpp (o Ollama é llama.cpp por baixo), contexto de 16k, temperatura 0. A memória de vídeo inclui o que o Windows usa da placa (~0,4 GB).</p>
<p style="color:var(--muted)">Capítulos nunca vistos no treino. Cada linha é um bloco de texto que o Claude traduziu; as caixas magenta numeradas estão na imagem.</p>
<section><h2>Resumo</h2><div class="t"><table>
<tr><th>Modelo</th><th>Pedidos</th><th>Respostas quebradas</th><th>Falas perdidas</th><th>Falas quase iguais às do Claude</th><th>Falas bem diferentes (balão trocado ou outro sentido)</th><th>Tempo por pedido</th><th>Memória de vídeo (pico)</th></tr>
{''.join(summary)}</table></div></section>
{''.join(pages)}
</main></body></html>"""
    out = results / "comparativo.html"
    results.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{len(pages)} pedidos -> {out}")


if __name__ == "__main__":
    main()

"""Tudo o que foi medido nos capítulos 96 e 97 numa página, e as páginas lado a lado num PDF.

Lê de test-soccer/ as respostas do benchmark (results/*.jsonl, compare_models.py), os capítulos inteiros
que o app traduziu com cada modelo (claude-out/, render/<modelo>/ e seus logs) e os traços de GPU, CPU e
RAM (metrics/*.csv, gpu_monitor.py). Grava results/relatorio.html, results/metricas.json,
results/lado-a-lado.pdf e um PDF por capítulo e modelo em results/capitulos/.

Uso: python scripts/full_report.py
"""
import csv
import difflib
import html
import json
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import ROOT
from compare_models import load_requests
from compare_report import stats_for

T = ROOT / "test-soccer"
RESULTS = T / "results"
CHAPTERS = (96, 97)
# nome, rótulo, pasta com as páginas do app, nome do log em render-<log>-chNN.log, nome do traço
RENDERS = [
    ("claude", "Claude Haiku (API)", T / "claude-out", None, None),
    ("gemma-e4b", "Gemma 4 E4B", T / "render" / "gemma-e4b", "gemma-e4b", "render-gemma-e4b"),
    ("gemma-e2b", "Gemma 4 E2B", T / "render" / "gemma-e2b", "gemma-e2b", "render-gemma-e2b"),
    ("qwen-q4", "Qwen treinado Q4_0", T / "render" / "qwen-q4", "qwen-q4", "render-qwen-q4"),
    ("qwen35-2b", "Qwen3.5-2B treinado Q8_0", T / "render" / "qwen35-2b", "qwen35-2b", "render-qwen35-2b"),    ("qwen35-2b-app", "Qwen3.5-2B treinado, integrado (Neste PC)", T / "render" / "qwen35-2b-app", "qwen35-2b-app",
     "render-qwen35-2b-app"),
]
BENCH = [("gemma", "Gemma 4 E4B"), ("gemma-e2b", "Gemma 4 E2B"), ("qwen-gguf-q8", "Qwen treinado Q8_0"),
         ("qwen-gguf", "Qwen treinado Q4_0"), ("qwen", "Qwen treinado (PyTorch)"),
         ("qwen2-vl-2b", "Qwen2-VL-2B original Q8_0"), ("qwen25-vl-3b", "Qwen2.5-VL-3B original Q8_0 (visão na CPU)"),
         ("qwen3-vl-2b", "Qwen3-VL-2B original Q8_0"), ("qwen3-vl-4b", "Qwen3-VL-4B original Q4_K_M"),
         ("qwen35-2b", "Qwen3.5-2B original Q8_0"), ("qwen35-4b", "Qwen3.5-4B original Q4_K_M"),
         ("qwen35-2b-trained", "Qwen3.5-2B treinado Q8_0"),
         ("qwen35-2b-trained-v2", "Qwen3.5-2B treinado v2 Q8_0 (menos repescagens vazias)")]
SIZES = {"gemma-e4b": "5,1 GB + 1,0 GB visão (Q4)", "gemma-e2b": "3,3 GB + 1,0 GB visão (Q4)",
         "qwen-q4": "0,9 GB + 1,3 GB visão", "qwen35-2b": "2,0 GB + 0,7 GB visão", "qwen35-2b-app": "2,0 GB + 0,7 GB visão", "claude": "na nuvem"}


def trace(name):
    """Médias e picos do traço de GPU, CPU e RAM de uma etapa (uma linha por segundo)."""
    f = T / "metrics" / f"{name}.csv"
    if not f.exists():
        return None
    rows = []
    for r in csv.DictReader(open(f, encoding="utf-8")):
        try:
            rows.append({k: float(v) for k, v in r.items()})
        except (TypeError, ValueError):
            pass
    if len(rows) < 2:
        return None
    col = lambda k: [r[k] for r in rows]
    seconds = rows[-1]["time"] - rows[0]["time"]
    return {"seconds": round(seconds), "vram_peak_mb": max(col("gpu_mem")), "vram_avg_mb": round(statistics.mean(col("gpu_mem"))),
            "gpu_util_avg": round(statistics.mean(col("gpu_util"))), "power_avg_w": round(statistics.mean(col("gpu_power")), 1),
            "power_peak_w": max(col("gpu_power")), "energy_wh": round(sum(col("gpu_power")) / 3600, 2),
            "temp_max_c": max(col("gpu_temp")), "temp_avg_c": round(statistics.mean(col("gpu_temp"))),
            "clock_avg_mhz": round(statistics.mean(col("gpu_clock"))), "cpu_avg": round(statistics.mean(col("cpu"))),
            "ram_peak_mb": max(col("ram_used"))}


LINE = re.compile(r"\[translate\] .*?: (\d+) images, (\d+) tokens in, (\d+) out, ([\d.]+)s \((\w*)\)")


def requests_of(log):
    """Chamadas ao modelo registradas pelo app: tokens de entrada e saída, segundos e como cada uma terminou."""
    calls = []
    for c in CHAPTERS:
        f = T / f"render-{log}-ch{c}.log"
        if f.exists():
            calls += [m.groups() for m in LINE.finditer(f.read_text(encoding="utf-8", errors="replace"))]
    if not calls:
        return None
    secs = [float(c[3]) for c in calls]
    out = sum(int(c[2]) for c in calls)
    return {"requests": len(calls), "tokens_in": sum(int(c[1]) for c in calls), "tokens_out": out,
            "seconds_avg": round(statistics.mean(secs), 1), "seconds_median": round(statistics.median(secs), 1),
            "seconds_max": round(max(secs), 1), "tokens_per_s": round(out / sum(secs), 1) if sum(secs) else None,
            "cut_short": sum(c[4] == "length" for c in calls)}


def pages_of(folder, chapter):
    out = {}
    for f in sorted((folder / f"ch{chapter}").glob("p[0-9][0-9].json")):
        out[int(f.stem[1:])] = json.loads(f.read_text(encoding="utf-8"))
    return out


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def said(b):
    return str(b.get("translation") or "").strip().lower()


def page_quality(name, folder):
    """Capítulos inteiros contra os do Claude.

    O mesmo detector achou as mesmas caixas, então cada bloco do Claude é pareado com o bloco do modelo que
    cobre o mesmo lugar da página.
    """
    s = {"pages": 0, "blocks": 0, "drawn": 0, "left_untranslated": 0, "lines": 0, "lost": 0, "close": 0, "far": 0}
    span = []
    for c in CHAPTERS:
        mine, ref = pages_of(folder, c), pages_of(T / "claude-out", c)
        afters = sorted((folder / f"ch{c}").glob("p*-after.png"))
        if afters:
            span.append(afters[-1].stat().st_mtime - afters[0].stat().st_mtime)
        for n, page in mine.items():
            s["pages"] += 1
            blocks = page["blocks"]
            s["blocks"] += len(blocks)
            s["drawn"] += sum(bool((b.get("overlay") or {}).get("drawn")) for b in blocks)
            s["left_untranslated"] += page.get("orphans", 0)
            for rb in ref.get(n, {}).get("blocks", []):
                if rb.get("kind") in ("sfx", "watermark") or not said(rb) or not rb.get("textBox"):
                    continue
                s["lines"] += 1
                hits = [b for b in blocks if b.get("textBox") and said(b) and iou(rb["textBox"], b["textBox"]) > 0.3]
                if not hits:
                    s["lost"] += 1
                    continue
                likeness = max(difflib.SequenceMatcher(None, said(rb), said(b)).ratio() for b in hits)
                s["close"] += likeness >= 0.7
                s["far"] += likeness < 0.35
    s["span_s"] = round(sum(span)) if span else None
    return s


def side_by_side(out_pdf, col_w=460, chunk_h=2000):
    """Um PDF com, para cada página, o original e a tradução de cada modelo em colunas, cortados em pedaços."""
    cols = [("Original (inglês)", T / "claude-out", "before")] + [(label, folder, "after") for _, label, folder, _, _ in RENDERS]
    try:
        font = ImageFont.truetype("arialbd.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    frames, gap, head = [], 14, 50
    for c in CHAPTERS:
        for n in range(1, 99):
            files = [folder / f"ch{c}" / f"p{n:02d}-{which}.png" for _, folder, which in cols]
            if not files[0].exists():
                break
            imgs = []
            for f in files:
                img = Image.open(f).convert("RGB") if f.exists() else None
                imgs.append(img.resize((col_w, round(img.height * col_w / img.width))) if img else None)
            height = max(i.height for i in imgs if i)
            for top in range(0, height, chunk_h):
                h = min(chunk_h, height - top)
                frame = Image.new("RGB", (len(cols) * (col_w + gap) + gap, h + head), "white")
                draw = ImageDraw.Draw(frame)
                for k, ((label, _, _), img) in enumerate(zip(cols, imgs)):
                    x = gap + k * (col_w + gap)
                    draw.text((x, 12), f"{label} · cap {c} p{n}", fill="black", font=font)
                    if img:
                        frame.paste(img.crop((0, top, col_w, min(top + h, img.height))), (x, head))
                    else:
                        draw.text((x, head + 20), "(não gerado)", fill="red", font=font)
                frames.append(frame)
    if frames:
        frames[0].save(out_pdf, save_all=True, append_images=frames[1:], quality=72, resolution=110)
    return len(frames)


def chapter_pdfs(out_dir):
    """Os capítulos traduzidos por cada modelo como o leitor recebe, um PDF por capítulo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for _, label, folder, _, _ in RENDERS:
        for c in CHAPTERS:
            pages = sorted((folder / f"ch{c}").glob("p*-after.png"))
            if not pages:
                continue
            imgs = [Image.open(p).convert("RGB") for p in pages]
            f = out_dir / f"Cap {c} - {label}.pdf"
            imgs[0].save(f, save_all=True, append_images=imgs[1:], quality=85, resolution=100)
            made.append(f.name)
    return made


def fmt(v, unit=""):
    return "—" if v is None else f"{v}{unit}"


def table(head, rows):
    return ("<div class='t'><table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")


def main():
    metrics = {"bench": {}, "chapters": {}}
    # 1. Os mesmos 111 pedidos para todos os modelos.
    requests = load_requests(T / "dataset")
    answers = {}
    for name, _ in BENCH:
        f = RESULTS / f"{name}.jsonl"
        if f.exists():
            answers[name] = {(d := json.loads(l))["key"]: d for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    bench_rows = []
    for name, label in BENCH:
        if name not in answers:
            continue
        s = stats_for(name, requests, answers)
        secs = sorted(s["seconds"])
        tr = trace(f"bench-{name}")
        m = {**{k: v for k, v in s.items() if k != "seconds"}, "seconds_avg": round(statistics.mean(secs), 1),
             "seconds_median": round(statistics.median(secs), 1), "seconds_p90": secs[int(len(secs) * 0.9)],
             "seconds_total": round(sum(secs)), "trace": tr}
        metrics["bench"][name] = m
        bench_rows.append([label, m["requests"], m["broken"], f"{m['lost']} de {m['lines']}", m["close"], m["far"],
                           f"{m['seconds_avg']} s", f"{m['seconds_median']} s", f"{m['seconds_p90']} s",
                           f"{m['seconds_total'] / 60:.1f} min", f"{m['gpu'] / 1024:.1f} GB"])

    # 2. Capítulos inteiros pelo app.
    steps = {}
    if (T / "metrics" / "steps.tsv").exists():
        for line in (T / "metrics" / "steps.tsv").read_text().splitlines():
            n, a, b, rc = line.split("\t")
            steps[n] = float(b) - float(a)
    qual_rows, speed_rows, hw_rows = [], [], []
    for name, label, folder, log, tr_name in RENDERS:
        if not (folder / "ch96").exists():
            continue
        q = page_quality(name, folder)
        rq = requests_of(log) if log else None
        tr = trace(tr_name) if tr_name else None
        wall = steps.get(tr_name) if tr_name else q.pop("span_s", None)
        q.pop("span_s", None)
        metrics["chapters"][name] = {"quality": q, "requests": rq, "trace": tr, "wall_s": wall, "size": SIZES.get(name)}
        qual_rows.append([label, q["pages"], q["blocks"], q["drawn"], q["left_untranslated"],
                          "—" if name == "claude" else f"{q['lost']} de {q['lines']}",
                          "—" if name == "claude" else q["close"], "—" if name == "claude" else q["far"]])
        speed_rows.append([label, fmt(round(wall / 60, 1) if wall else None, " min"),
                           fmt(round(wall / q["pages"], 1) if wall else None, " s"),
                           fmt(rq and rq["requests"]), fmt(rq and rq["seconds_avg"], " s"), fmt(rq and rq["seconds_max"], " s"),
                           fmt(rq and rq["tokens_in"]), fmt(rq and rq["tokens_out"]), fmt(rq and rq["tokens_per_s"]),
                           fmt(rq and rq["cut_short"])])
        if tr:
            hw_rows.append([label, SIZES.get(name), f"{tr['vram_peak_mb'] / 1024:.1f} GB", f"{tr['vram_avg_mb'] / 1024:.1f} GB",
                            f"{tr['gpu_util_avg']}%", f"{tr['clock_avg_mhz']} MHz", f"{tr['power_avg_w']} W / {tr['power_peak_w']} W",
                            f"{tr['energy_wh']} Wh", f"{tr['temp_avg_c']} / {tr['temp_max_c']} °C", f"{tr['cpu_avg']}%",
                            f"{tr['ram_peak_mb'] / 1024:.1f} GB"])
    e2b = metrics["bench"].get("gemma-e2b", {}).get("trace")
    if e2b:
        hw_rows.append(["Gemma 4 E2B (só os 111 pedidos)", SIZES["gemma-e2b"], f"{e2b['vram_peak_mb'] / 1024:.1f} GB",
                        f"{e2b['vram_avg_mb'] / 1024:.1f} GB", f"{e2b['gpu_util_avg']}%", f"{e2b['clock_avg_mhz']} MHz",
                        f"{e2b['power_avg_w']} W / {e2b['power_peak_w']} W", f"{e2b['energy_wh']} Wh",
                        f"{e2b['temp_avg_c']} / {e2b['temp_max_c']} °C", f"{e2b['cpu_avg']}%", f"{e2b['ram_peak_mb'] / 1024:.1f} GB"])

    frames = side_by_side(RESULTS / "lado-a-lado.pdf")
    chapter_pdfs(RESULTS / "capitulos")
    (RESULTS / "metricas.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Relatório dos testes</title>
<style>
:root {{ --bg:#f6f5f1; --card:#fff; --ink:#1d1d1b; --muted:#6b6a65; --line:#e4e2db; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151614; --card:#1e201d; --ink:#ecebe6; --muted:#9a9993; --line:#30322e; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,"Segoe UI",sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 48px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-top:16px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin:0 0 6px; }} p {{ color:var(--muted); margin:4px 0 10px; }}
.t {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--muted); font-weight:500; font-size:12px; }} td:first-child {{ font-weight:600; }}
@media print {{ @page {{ size:A4 landscape; margin:8mm; }} body {{ background:#fff; font-size:11px; }}
  .t {{ overflow:visible; }} th, td {{ white-space:normal; padding:4px 5px; font-size:10px; }} section {{ break-inside:avoid; }} }}
</style></head><body><main>
<h1>Testes: I'm Not a Soccer Genius! caps 96 e 97 (fora do treino)</h1>
<p>RTX 4050 6 GB. Modelos locais um de cada vez, temperatura 0, contexto 16k. A memória de vídeo inclui ~0,5 GB do Windows e, nos capítulos inteiros, o detector e o LaMa do app.</p>
<section><h2>1. Os mesmos 111 pedidos</h2><p>Cada modelo recebe exatamente as imagens e o texto que o Claude recebeu. "Quase iguais" = parecidas ≥70% com a do Claude; "bem diferentes" = &lt;35% (balão trocado ou outro sentido).</p>
{table(["Modelo", "Pedidos", "Quebradas", "Falas perdidas", "Quase iguais", "Bem diferentes", "Média", "Mediana", "90% abaixo de", "Total", "Pico VRAM"], bench_rows)}</section>
<section><h2>2. Capítulos inteiros no Manga Index — qualidade</h2><p>O app faz tudo como para um leitor: detecta, pede, repesca, apaga e desenha. "Texto deixado em inglês" = áreas de texto achadas pelo detector que ficaram sem tradução na página. A comparação com o Claude casa os balões pela posição.</p>
{table(["Modelo", "Páginas", "Blocos", "Desenhados", "Texto deixado em inglês", "Falas do Claude perdidas", "Quase iguais", "Bem diferentes"], qual_rows)}</section>
<section><h2>3. Capítulos inteiros — tempo e tokens</h2><p>Tempo total inclui abrir o app (~15 s por capítulo), detectar, apagar e desenhar. Tokens/s = tokens escritos ÷ segundos dos pedidos.</p>
{table(["Modelo", "Total (2 caps)", "Por página", "Pedidos", "Média por pedido", "Pedido mais lento", "Tokens lidos", "Tokens escritos", "Tokens/s", "Cortados no limite"], speed_rows)}</section>
<section><h2>4. Máquina durante os capítulos inteiros</h2><p>Uma leitura por segundo (nvidia-smi e psutil). Energia = só a placa de vídeo.</p>
{table(["Modelo", "Arquivos", "VRAM pico", "VRAM média", "GPU média", "Clock médio", "Consumo médio / pico", "Energia", "Temp média / máx", "CPU média", "RAM pico (PC todo)"], hw_rows)}</section>
<p>PDF lado a lado: lado-a-lado.pdf ({frames} páginas). Números brutos: metricas.json. Balão por balão: comparativo.html.</p>
</main></body></html>"""
    (RESULTS / "relatorio.html").write_text(page, encoding="utf-8")
    # O relatório também em PDF, impresso pelo Edge (sempre presente no Windows).
    edge = next((p for p in (Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                             Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")) if p.exists()), None)
    if edge:
        import subprocess
        subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={RESULTS / 'relatorio.pdf'}", (RESULTS / "relatorio.html").as_uri()],
                       capture_output=True, timeout=120)
    print(f"relatorio.html, metricas.json, lado-a-lado.pdf ({frames} páginas)")


if __name__ == "__main__":
    main()

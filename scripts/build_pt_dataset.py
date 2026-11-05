"""Monta o segundo dataset de treino com os capítulos em inglês que também têm edição em português.

Cada capítulo em inglês passa pelo modo de avaliação do próprio Manga Index (o build congelado em
test-soccer/app-out), que detecta o texto, pergunta ao Claude como o app faz e grava cada pedido na pasta
do dataset. No caminho até a Anthropic cada pedido passa por um proxy local, que localiza o mesmo trecho
na edição em português (scripts/pt_align.py) e registra a correspondência. A imagem em português não é
enviada: o Claude recusou usar a tradução de terceiros como referência, e o dataset guarda o pedido
exatamente como o app o fez.

O app lê a própria chave e a manda ao proxy (ANTHROPIC_BASE_URL), que só a repassa.
Grava dataset-pt/data (formato do training-dataset do app), dataset-pt/requests.jsonl (correspondência e
tokens de cada pedido) e dataset-pt/done.txt (capítulos prontos, para retomar depois).

Uso: python scripts/build_pt_dataset.py [--workers 3] [--only N] [--title "Overgeared"]
"""
import argparse
import base64
import http.server
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pymupdf

from common import ROOT
from library_pairs import LIB, PT_HOST
from pt_align import Strip

APP_SRC = Path(os.environ.get("MANGA_INDEX", "D:/Dev/Manga Index"))
APP = ROOT / "test-soccer" / "app-out"
SETTINGS_FROM = ROOT / "test-soccer" / "userdata"
OUT = ROOT / "dataset-pt"
WORK = OUT / "work"
ANTHROPIC = "https://api.anthropic.com"


def pairs():
    """(meta em inglês, meta em português) de cada capítulo nos dois idiomas, obra por obra."""
    by = defaultdict(lambda: {"en": {}, "pt": {}})
    for m in LIB.glob("*/meta.json"):
        d = json.loads(m.read_text(encoding="utf-8"))
        d["dir"] = m.parent
        lang = "pt" if PT_HOST in urlparse(d.get("source") or "").netloc else "en"
        by[d.get("workId")][lang][d.get("chapter")] = d
    out = []
    for w in by.values():
        for ch in sorted(set(w["en"]) & set(w["pt"])):
            out.append((w["en"][ch], w["pt"][ch]))
    return out


def broken(pt):
    """Capítulo em português que baixou como uma imagem pequena no lugar da tira."""
    doc = pymupdf.open(pt["dir"] / "original.pdf")
    return len(doc) == 1 and doc[0].rect.height < 1000


def decode(block):
    data = base64.b64decode(block["source"]["data"])
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Slot:
    """Uma instância do app por vez numa porta do proxy: o capítulo atual e sua tira em português."""

    def __init__(self, port):
        self.port = port
        self.strip = None
        self.chapter = ""
        self.lock = threading.Lock()


log_lock = threading.Lock()


def log(entry):
    with log_lock, open(OUT / "requests.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def locate(slot, body):
    """Onde a página do pedido está na edição em português, para o registro."""
    content = body["messages"][0]["content"]
    images = [b for b in content if b.get("type") == "image"]
    prompt = next(b for b in content if b.get("type") == "text")["text"]
    info = {"chapter": slot.chapter, "images": len(images)}
    if "cut out of a" in prompt:
        info["kind"] = "followup"
        return info
    info["kind"] = "page"
    page_imgs = images[:-1] if "is the key" in prompt and len(images) > 1 else images
    top, bottom, score = slot.strip.find_slices([decode(i) for i in page_imgs])
    info.update(top=top, bottom=bottom, score=round(score, 3))
    return info


def make_handler(slot):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            info = {}
            if self.path.endswith("/messages"):
                try:
                    with slot.lock:
                        info = locate(slot, json.loads(raw))
                except Exception as err:  # uma falha na busca nunca derruba o pedido
                    info = {"chapter": slot.chapter, "error": repr(err)}
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding", "connection")}
            req = urllib.request.Request(ANTHROPIC + self.path, data=raw, headers=headers, method="POST")
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    status, rh, data = r.status, r.headers, r.read()
            except urllib.error.HTTPError as e:
                status, rh, data = e.code, e.headers, e.read()
            if info:
                try:
                    usage = json.loads(data).get("usage", {})
                except ValueError:
                    usage = {}
                log({**info, "status": status, "seconds": round(time.time() - started, 1),
                     "in": usage.get("input_tokens"), "out": usage.get("output_tokens"), "time": time.strftime("%H:%M:%S")})
            self.send_response(status)
            for k in ("content-type", "retry-after", "request-id"):
                if rh.get(k):
                    self.send_header(k, rh.get(k))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def safe(name):
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def userdata(n):
    """Pasta do app para o slot n: as configurações salvas (perfil e chave do Claude), uma página por vez."""
    d = WORK / f"userdata-{n}"
    if not d.exists():
        d.mkdir(parents=True)
        shutil.copy(SETTINGS_FROM / "Local State", d / "Local State")
        s = json.loads((SETTINGS_FROM / "settings.json").read_text(encoding="utf-8"))
        s["settings"]["concurrency"] = 1  # assim uma repescagem sempre vem da página pedida logo antes
        (d / "settings.json").write_text(json.dumps(s), encoding="utf-8")
    return d


def run_chapter(slot, n, en, pt):
    title = safe(en["title"])
    pdf = WORK / "en" / f"{title}.pdf"  # o dataset nomeia o capítulo pelo arquivo
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if not pdf.exists():
        shutil.copy(en["dir"] / "original.pdf", pdf)
    with slot.lock:
        slot.strip, slot.chapter = Strip(pt["dir"] / "original.pdf"), title
    out = WORK / f"out-{n}"
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "MI_USER_DATA": str(userdata(n)), "MI_DATASET_DIR": str(OUT / "data"), "MI_EVAL_PDF": str(pdf),
           "MI_EVAL_OUT": str(out), "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{slot.port}"}
    (WORK / "logs").mkdir(exist_ok=True)
    started = time.time()
    with open(WORK / "logs" / f"{title}.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(f'npx electron-vite preview --skipBuild --outDir "{APP}" --entry "{APP / "main" / "index.js"}"',
                            cwd=APP_SRC, env=env, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=3 * 3600).returncode
    pages = len(list(out.glob("p*-after.png")))
    print(f"[{n}] {title}: {pages} páginas, {time.time() - started:.0f} s, rc {rc}", flush=True)
    if rc == 0 and pages:
        with open(OUT / "done.txt", "a", encoding="utf-8") as f:
            f.write(en["id"] + "\n")
    shutil.rmtree(out, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--only", type=int, default=0, help="para depois de N capítulos")
    p.add_argument("--title", default="", help="só capítulos cujo título contém este texto")
    p.add_argument("--port", type=int, default=8791)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    done = set((OUT / "done.txt").read_text(encoding="utf-8").split()) if (OUT / "done.txt").exists() else set()
    todo, skipped = queue.Queue(), []
    chosen = 0
    # Uma obra de cada vez em rodízio, para que uma parada antecipada ainda deixe um dataset variado.
    by_work = defaultdict(list)
    for en, pt in pairs():
        by_work[en.get("workId")].append((en, pt))
    order = [x for i in range(max(map(len, by_work.values()))) for w in by_work.values() if i < len(w) for x in [w[i]]]
    for en, pt in order:
        if en["id"] in done or args.title.lower() not in en["title"].lower():
            continue
        if broken(pt):
            skipped.append(en["title"])
            continue
        if args.only and chosen >= args.only:
            break
        todo.put((en, pt))
        chosen += 1
    if skipped:
        print("Capítulos em português quebrados (uma imagem pequena), deixados de fora:", ", ".join(skipped))
    print(f"{chosen} capítulos a fazer, {len(done)} feitos antes")

    def worker(n):
        slot = Slot(args.port + n)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", slot.port), make_handler(slot))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        while True:
            try:
                en, pt = todo.get_nowait()
            except queue.Empty:
                break
            try:
                run_chapter(slot, n, en, pt)
            except Exception as err:
                print(f"[{n}] {en['title']}: {err!r}", flush=True)
        server.shutdown()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("FIM")


if __name__ == "__main__":
    main()

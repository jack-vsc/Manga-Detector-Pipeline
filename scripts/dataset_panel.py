"""Painel ao vivo do dataset, do treino e da máquina.

Serve a página em http://localhost:8770. A página consulta a cada 20 s /stats.json (o dataset em disco,
via dataset_stats.py), /training.json (o log do treino) e /system.json (GPU, CPU e RAM). POST /pause e
/resume pausam e retomam o treino. Com --host 0.0.0.0 um celular na mesma rede também abre o painel.

Uso: python scripts/dataset_panel.py [--port 8770] [--host 0.0.0.0] [--run q35-2b-v2 --compare q35-2b]
"""
import argparse
import json
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).parent))
from dataset_stats import compute  # noqa: E402
from training_stats import compute as training  # noqa: E402
import system_stats  # noqa: E402

PAGE = Path(__file__).with_name("dataset_panel.html")
# Várias abas abertas dividem a mesma leitura do disco.
FRESH_FOR = 10
_last, _lock = (0.0, None), Lock()


def stats(goal):
    global _last
    with _lock:
        if time.time() - _last[0] > FRESH_FOR:
            _last = (time.time(), json.dumps(compute(goal), ensure_ascii=False).encode())
        return _last[1]


RUN = "manga-translate"  # --run: treino acompanhado (runs-<nome>.log)
COMPARE = None  # --compare: treino anterior desenhado por baixo


def training_with_compare():
    t = training(RUN)
    if COMPARE:
        c = training(COMPARE)
        if c.get("losses"):
            t["compare"] = {"name": COMPARE, "losses": c["losses"], "evals": c["evals"]}
    return t


_system = (0.0, None)
_system_lock = Lock()


def system(fresh=False):
    """Ler os contadores leva ~2 s; abas que pedem juntas dividem a mesma leitura."""
    global _system
    with _system_lock:
        if fresh or time.time() - _system[0] > 5:
            _system = (time.time(), json.dumps(system_stats.compute()).encode())
        return _system[1]


def handler(goal):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/stats.json":
                body, kind = stats(goal), "application/json"
            elif path == "/training.json":
                body, kind = json.dumps(training_with_compare(), ensure_ascii=False).encode(), "application/json"
            elif path == "/system.json":
                body, kind = system(), "application/json"
            elif path.startswith("/teste/"):
                # O comparativo (compare_report.py) e as imagens que ele mostra, de test-soccer/.
                base = (PAGE.parent.parent / "test-soccer").resolve()
                file = (base / unquote(path[len("/teste/"):])).resolve()
                if not file.is_relative_to(base) or not file.is_file():
                    self.send_error(404)
                    return
                body = file.read_bytes()
                kind = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".png": "image/png"}.get(file.suffix, "application/octet-stream")
            elif path in ("/", "/index.html"):
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/pause", "/resume"):
                self.send_error(404)
                return
            with _system_lock:
                ok = system_stats.set_paused(path == "/pause")
            body = system(fresh=True)
            self.send_response(200 if ok else 409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def lan_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 para abrir de outros aparelhos da rede")
    p.add_argument("--goal", type=int, default=5000, help="meta de páginas do dataset")
    p.add_argument("--dataset", help="outra pasta de dataset (ex.: dataset-pt/data)")
    p.add_argument("--run", default="manga-translate", help="treino acompanhado (runs-<nome>.log)")
    p.add_argument("--compare", help="treino anterior desenhado por baixo (runs-<nome>.log)")
    args = p.parse_args()
    global RUN, COMPARE
    RUN, COMPARE = args.run, args.compare
    if args.dataset:
        import dataset_stats
        dataset_stats.DATASET = Path(args.dataset).resolve()
    server = ThreadingHTTPServer((args.host, args.port), handler(args.goal))
    print(f"Painel do dataset: http://localhost:{args.port}", flush=True)
    if args.host == "0.0.0.0" and (ip := lan_address()):
        print(f"No celular (mesma rede Wi-Fi): http://{ip}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

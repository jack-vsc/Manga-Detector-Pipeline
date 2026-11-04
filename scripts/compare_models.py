"""Roda modelos locais nos pedidos do Manga Index que o Claude já respondeu, para comparar com ele.

Lê um dataset no formato que o Manga Index grava (annotations/<capítulo>.jsonl, images/), envia as imagens
e o prompt de cada pedido como foram enviados ao Claude e grava <dataset>/../results/<nome>.jsonl com a
resposta, os segundos e o pico de memória da GPU. compare_report.py põe as respostas lado a lado.

Uso:
  python scripts/compare_models.py qwen   [--dataset test-soccer/dataset] [--adapter runs/manga-translate/final]
  python scripts/compare_models.py gemma  [--ollama-model gemma4:e4b-it-qat]
  python scripts/compare_models.py server --name qwen-gguf [--server http://localhost:8771/v1]   (llama-server)
"""
import argparse
import base64
import io
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from common import BASE_MODEL, ROOT


def load_requests(dataset):
    out = []
    for file in sorted((dataset / "annotations").glob("*.jsonl")):
        with open(file, encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


class GpuPeak:
    """Maior uso de memória que o nvidia-smi informa durante o bloco (o Ollama roda em outro processo)."""

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()

    def __enter__(self):
        self.peak = 0
        self._stop.clear()
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def _watch(self):
        while not self._stop.is_set():
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(used.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(0.5)

    def __exit__(self, *exc):
        self._stop.set()


def run_qwen(requests, dataset, args):
    import torch
    from PIL import Image

    from model import load_model, load_processor, messages, request_images

    # Ao contrário do treino, o kernel "math" de atenção fica ligado: gerando um token por vez com as
    # cabeças de chave/valor compartilhadas do Qwen2 não há outro, e nesse caso ele é pequeno.
    processor = load_processor(BASE_MODEL)
    model = load_model(BASE_MODEL, adapter=str(ROOT / args.adapter))

    @torch.no_grad()
    def answer(r):
        imgs = request_images([Image.open(dataset / "images" / i["file"]) for i in r["images"]])
        text = processor.apply_chat_template(messages(r["prompt"], images=len(imgs)), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    return answer


def run_gemma(requests, dataset, args):
    def answer(r):
        images = [base64.b64encode((dataset / "images" / i["file"]).read_bytes()).decode() for i in r["images"]]
        body = {"model": args.ollama_model, "stream": False,
                "messages": [{"role": "user", "content": r["prompt"], "images": images}],
                "options": {"temperature": 0, "num_ctx": 16384, "num_predict": args.max_tokens}}
        req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as res:
            return json.loads(res.read())["message"]["content"].strip()

    return answer


def run_server(requests, dataset, args):
    """Servidor no formato da OpenAI (llama-server com os GGUF): imagens como URLs data: antes do texto."""
    from PIL import Image

    from model import request_images

    def encode(img):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode()

    def answer(r):
        files = [dataset / "images" / i["file"] for i in r["images"]]
        # --fit-images: no tamanho do treino (model.request_images); sem ele, como o Claude recebeu.
        data = ([encode(img) for img in request_images([Image.open(f) for f in files])] if args.fit_images
                else [base64.b64encode(f.read_bytes()).decode() for f in files])
        images = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + d}} for d in data]
        body = {"model": "local", "temperature": 0, "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": [*images, {"type": "text", "text": r["prompt"]}]}]}
        req = urllib.request.Request(args.server.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as res:
            return json.loads(res.read())["choices"][0]["message"]["content"].strip()

    return answer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=["qwen", "gemma", "server"])
    p.add_argument("--server", default="http://localhost:8771/v1", help="para 'server': a URL base no formato da OpenAI")
    p.add_argument("--name", help="nome do arquivo de resultados (padrão: o do modelo)")
    p.add_argument("--fit-images", action="store_true", help="para 'server': imagens no tamanho do treino")
    p.add_argument("--dataset", default="test-soccer/dataset")
    p.add_argument("--adapter", default="runs/manga-translate/final")
    p.add_argument("--ollama-model", default="gemma4:e4b-it-qat")
    p.add_argument("--max-tokens", type=int, default=3000)
    args = p.parse_args()

    dataset = ROOT / args.dataset
    requests = load_requests(dataset)
    out = dataset.parent / "results" / f"{args.name or args.model}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    answer = {"qwen": run_qwen, "gemma": run_gemma, "server": run_server}[args.model](requests, dataset, args)
    print(f"{len(requests)} pedidos, {len(done)} já respondidos", flush=True)
    for n, r in enumerate(requests, 1):
        if r["key"] in done:
            continue
        with GpuPeak() as gpu:
            started = time.time()
            try:
                text, error = answer(r), None
            except Exception as e:  # um pedido com erro é registrado e os outros seguem
                text, error = "", f"{type(e).__name__}: {e}"
            seconds = time.time() - started
        line = {"key": r["key"], "chapter": r["chapter"], "page": r["page"], "answer": text, "error": error,
                "seconds": round(seconds, 1), "gpu_mb": gpu.peak}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{n}/{len(requests)} {r['chapter']} p{r['page']}: {seconds:.0f} s, {gpu.peak} MB{', ' + error if error else ''}", flush=True)


if __name__ == "__main__":
    main()

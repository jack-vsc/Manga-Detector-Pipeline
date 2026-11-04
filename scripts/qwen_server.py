"""O modelo treinado atrás de um endpoint no formato da OpenAI, para o Manga Index usar como qualquer API.

No Manga Index, crie um perfil (ou use MI_EVAL_MODEL) com a URL http://localhost:8771/v1 e qualquer nome
de modelo. Só entende o que o app envia: uma mensagem de usuário com texto e imagens (URLs data:). As
imagens são cortadas no tamanho do treino (model.request_images): páginas primeiro, chave por último.

Uso: python scripts/qwen_server.py [--adapter runs/manga-translate/final] [--port 8771]
"""
import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch
from PIL import Image

from common import BASE_MODEL, ROOT
from model import load_model, load_processor, messages, request_images


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="runs/manga-translate/final")
    p.add_argument("--port", type=int, default=8771)
    args = p.parse_args()

    processor = load_processor(BASE_MODEL)
    model = load_model(BASE_MODEL, adapter=str(ROOT / args.adapter))
    one_at_a_time = Lock()

    @torch.no_grad()
    def answer(body):
        content = body["messages"][-1]["content"]
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        prompt = "\n".join(c["text"] for c in parts if c.get("type") == "text")
        images = [Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",", 1)[1])))
                  for c in parts if c.get("type") == "image_url"]
        imgs = request_images(images) if images else []
        text = processor.apply_chat_template(messages(prompt, images=len(imgs)), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=imgs or None, return_tensors="pt").to(model.device)
        limit = min(int(body.get("max_tokens") or 3000), 3000)
        out = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        reply = processor.batch_decode(new, skip_special_tokens=True)[0].strip()
        finished = new.shape[1] < limit or new[0, -1].item() == processor.tokenizer.eos_token_id
        return reply, "stop" if finished else "length", inputs["input_ids"].shape[1], new.shape[1]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            started = time.time()
            with one_at_a_time:
                reply, finish, read, written = answer(body)
            print(f"{read} tokens de entrada, {written} de saída, {time.time() - started:.1f} s ({finish})", flush=True)
            data = json.dumps({"id": "qwen", "object": "chat.completion", "model": body.get("model", "qwen"),
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": finish}],
                               "usage": {"prompt_tokens": read, "completion_tokens": written}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    print(f"Qwen + {args.adapter} em http://localhost:{args.port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

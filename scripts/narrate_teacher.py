"""Rotula as páginas de data/pages.jsonl com um modelo local maior (o professor), uma vez só.

O professor lê cada bloco de texto e reescreve a página como trecho de light novel; o modelo pequeno
aprende a partir dessas respostas. Serve qualquer servidor compatível com a API da OpenAI que leia
imagens; o padrão é o Gemma no Ollama. Grava data/teacher.jsonl e retoma de onde parou.

Uso: python scripts/narrate_teacher.py [--limit 2000]
"""
import argparse
import base64
import io
import json
import time
import urllib.request

from PIL import Image

from common import DATA, OCR_PROMPT

TEACHER_PROMPT = """Você vê uma página de mangá em inglês (lida da direita para a esquerda, de cima para baixo).
{lines}
Reescreva a página como um trecho de light novel em português do Brasil, narrado em terceira pessoa e no
passado, seguindo a ordem de leitura:
- descreva o jeito de cada personagem: aparência (cabelo, roupa, porte), postura e expressão no momento;
- cada fala vira diálogo com travessão (—), traduzida, com quem fala e como fala ("disse ela, franzindo a testa");
- o que acontece entre as falas (gestos, reações, o cenário) entra como narração;
- ignore créditos, marcas d'água e notas de tradutor;
- não invente o que não se vê nem falas que não estão na lista.
Responda só com o trecho."""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:11434/v1/chat/completions")
    p.add_argument("--model", default="gemma4:e4b-it-qat")
    p.add_argument("--split", choices=("train", "val", "all"), default="all")
    p.add_argument("--limit", type=int, default=0, help="para depois de N páginas novas (0 = todas)")
    p.add_argument("--max-boxes", type=int, default=40, help="pula páginas com mais blocos de texto que isso")
    p.add_argument("--max-side", type=int, default=1280, help="lado maior da página enviada, em pixels")
    return p.parse_args()


def jpeg(img, max_side):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ask(url, model, prompt, img, max_side):
    body = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg(img, max_side)}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as res:
        return json.load(res)["choices"][0]["message"]["content"].strip()


def label(args, record):
    img = Image.open(record["image"]).convert("RGB")
    lines = []
    for box in record["boxes"]:
        text = " ".join(ask(args.url, args.model, OCR_PROMPT, img.crop(box), 512).split())
        # Rabiscos e onomatopeias desenhadas voltam vazios ou como recusa.
        lines.append(text if text and any(c.isalpha() for c in text) else "")
    said = [t for t in lines if t]
    listed = "Falas, na ordem de leitura:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(said)) if said else "A página não tem falas."
    prose = ask(args.url, args.model, TEACHER_PROMPT.format(lines=listed), img.crop(record["crop"]), args.max_side)
    return {"id": record["id"], "lines": lines, "narration": prose}


def main():
    args = parse_args()
    out_path = DATA / "teacher.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    with open(DATA / "pages.jsonl", encoding="utf-8") as f:
        todo = [
            r
            for r in map(json.loads, f)
            if r["id"] not in done and len(r["boxes"]) <= args.max_boxes and args.split in ("all", r["split"])
        ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} páginas já rotuladas, {len(todo)} a fazer")
    with open(out_path, "a", encoding="utf-8") as out:
        for n, record in enumerate(todo, 1):
            t = time.time()
            try:
                result = label(args, record)
            except Exception as err:  # uma página ruim (timeout, imagem ilegível) não para a execução
                print(f"[{n}/{len(todo)}] {record['id']}: {type(err).__name__}: {err}")
                continue
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n}/{len(todo)}] {record['id']} {len(record['boxes'])} blocos, {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()

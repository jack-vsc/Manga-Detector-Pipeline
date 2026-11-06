"""Funde o adaptador treinado nos pesos do modelo base (bf16, na CPU) para o conversor do llama.cpp.

O adaptador foi treinado sobre o modelo em 4 bits, mas é fundido no modelo em precisão cheia: depois o
llama.cpp quantiza o conjunto do mesmo jeito que faz com os arquivos do Gemma.

Uso: python scripts/merge_adapter.py --base Qwen/Qwen3.5-2B --adapter runs/q35-2b/final --out runs/q35-2b/merged
"""
import argparse

import torch

from common import BASE_MODEL, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="runs/manga-translate/final")
    p.add_argument("--out", default="runs/manga-translate/merged")
    p.add_argument("--base", default=BASE_MODEL, help="modelo em que o adaptador foi treinado, ex.: Qwen/Qwen3.5-2B")
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter)).merge_and_unload()
    out = ROOT / args.out
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(out)
    print(f"modelo fundido salvo em {out}")


if __name__ == "__main__":
    main()

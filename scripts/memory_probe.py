"""Mede até que tamanho de pedido o treino cabe na placa.

Faz uma passada de treino (forward e backward) por pedido, do mais longo para o mais curto, com o PyTorch
limitado à memória da placa (train.cap_memory), para que um pedido grande demais falhe em vez de
transbordar para a RAM. Imprime, por pedido: tokens, imagens, pico de memória, segundos, se coube, e a
RAM do sistema emprestada à GPU ("memória compartilhada" do Windows).

Uso: python scripts/memory_probe.py [--count 40]
"""
import argparse
import subprocess
import time

import torch

from common import BASE_MODEL
from train import Collator, Requests, build_model, cap_memory, load_examples
from model import load_processor


def shared_mb():
    """RAM do sistema que este processo usa como memória de GPU, segundo o Windows."""
    import os

    cmd = (f"(Get-Counter '\\GPU Process Memory(pid_{os.getpid()}*)\\Shared Usage').CounterSamples"
           " | Measure-Object CookedValue -Sum | ForEach-Object { [int]($_.Sum / 1MB) }")
    try:
        return int(subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout.strip() or 0)
    except (ValueError, subprocess.TimeoutExpired):
        return -1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=40, help="pedidos testados, espalhados do mais longo para baixo")
    p.add_argument("--from-tokens", type=int, default=2800, help="menor tamanho testado")
    args = p.parse_args()
    args.base, args.val_chapters, args.val_max, args.lora_r, args.empty_followups = BASE_MODEL, 0.03, 0, 16, 1.0

    cap_memory()
    train, _ = load_examples(args)
    processor = load_processor(args.base)
    collator = Collator(processor)
    data = Requests(train)
    # O tamanho do texto separa os longos sem custo; a contagem exata vem do collator.
    order = sorted(range(len(train)), key=lambda i: -(len(train[i]["prompt"]) + len(train[i]["answer"]) + 900 * len(train[i]["images"])))
    sized = []
    for i in order[:600]:
        sized.append((collator([data[i]])["input_ids"].shape[1], i))
    sized.sort(reverse=True)
    sized = [s for s in sized if s[0] >= args.from_tokens]
    step = max(1, len(sized) // args.count)
    picks = sized[::step][: args.count]
    print(f"{len(picks)} pedidos, de {picks[0][0]} a {picks[-1][0]} tokens", flush=True)

    model = build_model(args)
    model.train()
    core = model.get_base_model()
    print(f"depois de carregar: compartilhada {shared_mb()} MiB", flush=True)
    print("tokens  imgs  pico_MiB  segundos  result  compart_MiB", flush=True)
    for tokens, i in picks:
        batch = collator([data[i]])
        labels = batch.pop("labels").cuda()
        batch = {k: v.cuda() for k, v in batch.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            hidden = core.model(**batch, use_cache=False, return_dict=True).last_hidden_state
            target = labels[:, 1:]
            learnt = target != -100
            loss = torch.nn.functional.cross_entropy(core.lm_head(hidden[:, :-1][learnt]).float(), target[learnt])
            loss.backward()
            torch.cuda.synchronize()
            result = "ok"
        except torch.OutOfMemoryError:
            result = "OOM"
        finally:
            hidden = loss = None
            model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_reserved() / 2**20
        torch.cuda.empty_cache()
        print(f"{tokens:6}  {len(train[i]['images']):4}  {peak:8.0f}  {time.time() - started:8.1f}  {result:6}  {shared_mb():11}", flush=True)


if __name__ == "__main__":
    main()

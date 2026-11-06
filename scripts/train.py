"""Ensina a tradução do Manga Index a um modelo pequeno (4 bits + LoRA, cabe numa placa de 6 GB).

Entrada: a página com as caixas numeradas, a imagem-chave com cada caixa recortada e o prompt do próprio
app. Saída: o JSON de blocos que o Claude respondeu. O dataset é o que o Manga Index grava em
%APPDATA%/Manga Index/training-dataset (veja dataset_stats.py), ou outra pasta com --dataset: pedidos de
página e repescagens das caixas que ficaram de fora. Alguns capítulos inteiros ficam de fora do treino
para medir o modelo em páginas que ele nunca viu. O adaptador vai para runs/<nome>/final.

Uso: python scripts/train.py --name q35-2b --base Qwen/Qwen3.5-2B --dataset dataset-pt/final --max-tokens 2600
"""
import argparse
import hashlib
import json
import random
import time

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from common import BASE_MODEL, MAX_TOKENS, PAGE_TOKENS, PAGE_TOKENS_SHRUNK, ROOT
import dataset_stats
from model import load_model, load_processor, messages, request_images
import model as model_module

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# O Qwen3.5 tem também camadas de atenção linear (in_proj_qkv, in_proj_z, out_proj). Só o modelo de
# linguagem aprende.
LORA_TARGETS_QWEN35 = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"
DATASET = dataset_stats.DATASET  # trocado por --dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="manga-translate")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="pedido mais longo treinado (as páginas encolhem para caber)")
    p.add_argument("--dataset", help="pasta do dataset (padrão: training-dataset do Manga Index), ex.: dataset-pt/final")
    p.add_argument("--val-chapters", type=float, default=0.03, help="fração dos capítulos fora do treino")
    p.add_argument("--val-max", type=int, default=150, help="máximo de pedidos de validação medidos")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="para depois de N passos (vale mais que --epochs)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accum", type=int, default=8, help="acúmulo de gradiente (o batch é 1)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--resume", action="store_true", help="continua do último checkpoint de --name")
    p.add_argument("--pause", type=float, default=0.6, help="segundos de descanso da GPU após cada pedido;"
                   " runs/<nome>/pause (um número) muda o valor durante o treino")
    p.add_argument("--heaviest", type=int, default=0, help="treina só nos N pedidos mais longos (teste de memória)")
    return p.parse_args()


def held_out(chapter, share):
    """Os mesmos capítulos em toda execução, não importa o que entrar depois no dataset."""
    return int(hashlib.sha1(chapter.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


class Requests(Dataset):
    """Pedidos do Manga Index; as imagens só são abertas quando pedidas."""

    def __init__(self, records, root=None):
        self.records = records
        # Guardados aqui porque no Windows os workers do DataLoader reimportam o módulo com os valores padrão.
        self.root = root or DATASET
        self.unit = model_module.UNIT

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        model_module.UNIT = self.unit
        images = request_images([Image.open(self.root / "images" / img["file"]) for img in r["images"]], r.get("page_tokens", PAGE_TOKENS))
        return images, r["prompt"], r["answer"]


def load_examples(args):
    train, val = [], []
    for file in sorted((DATASET / "annotations").glob("*.jsonl")):
        with open(file, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        (val if held_out(file.stem, args.val_chapters) else train).extend(records)
    random.Random(0).shuffle(val)
    val = val[: args.val_max]
    print(f"treino: {len(train)} pedidos, validação: {len(val)} (capítulos fora do treino)")
    if not train:
        raise SystemExit(f"nada para treinar em {DATASET}")
    return train, val


class Collator:
    """Tokeniza o batch; só a resposta entra na loss (prompt, imagens e padding ficam mascarados)."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        # A resposta começa depois da abertura do turno do assistente. No Qwen3.5 sem raciocínio essa
        # abertura inclui um <think></think> vazio, igual ao que o llama-server envia no uso.
        opened = processor.apply_chat_template(messages("x"), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.opening = opened[opened.rindex("<|im_start|>assistant\n"):]
        self.answer_starts = processor.tokenizer.encode(self.opening, add_special_tokens=False)

    def text(self, prompt, answer, images):
        asked = self.processor.apply_chat_template(messages(prompt, images=images), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return asked + answer + "<|im_end|>\n"

    def __call__(self, batch):
        p = self.processor
        full = [self.text(prompt, answer, len(imgs)) for imgs, prompt, answer in batch]
        enc = p(text=full, images=[img for imgs, _, _ in batch for img in imgs], padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        n = len(self.answer_starts)
        for i, ids in enumerate(enc["input_ids"].tolist()):
            start = max(j + n for j in range(len(ids) - n + 1) if ids[j : j + n] == self.answer_starts)
            labels[i, :start] = -100
        enc["labels"] = labels
        return enc


def fits(processor, collator, dataset, name, heaviest=0):
    """Tira os pedidos longos demais para a placa, contados uma vez com as imagens já cortadas.

    heaviest: mantém só os N mais longos, para conferir se ainda cabem.
    """
    keep, shrunk = [], 0
    for i, r in enumerate(dataset.records):
        n = collator([dataset[i]])["input_ids"].shape[1]
        for budget in PAGE_TOKENS_SHRUNK:
            if n <= MAX_TOKENS or len(r["images"]) == 1:
                break
            r = {**r, "page_tokens": budget}
            n = collator([Requests([r], dataset.root)[0]])["input_ids"].shape[1]
        if n <= MAX_TOKENS:
            keep.append((n, r))
            shrunk += "page_tokens" in r
    print(f"{name}: {shrunk} pedidos acima de {MAX_TOKENS} tokens com páginas menores, {len(dataset) - len(keep)} de fora")
    if heaviest:
        keep = sorted(keep, key=lambda k: -k[0])[:heaviest]
        print(f"{name}: os {len(keep)} mais longos, de {keep[-1][0]} a {keep[0][0]} tokens")
    return Requests([r for _, r in keep], dataset.root)


class AnswerTrainer(Trainer):
    """Calcula os logits só nas posições da resposta.

    O vocabulário tem 151 mil tokens; os logits de um pedido de 4k tokens passam de 2,5 GB, mais do que
    sobra na placa, e só as posições da resposta entram na loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = model.get_base_model()
        hidden = core.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        target = labels[:, 1:]
        learnt = target != -100
        logits = core.lm_head(hidden[:, :-1][learnt]).float()
        loss = torch.nn.functional.cross_entropy(logits, target[learnt])
        return (loss, {"loss": loss}) if return_outputs else loss

    pause, pause_file = 0.0, None

    def rest(self):
        """Pausa curta após cada pedido para a GPU esfriar, ao custo de tempo."""
        try:
            self.pause = float(self.pause_file.read_text().strip())
        except (OSError, ValueError, AttributeError):
            pass
        if self.pause > 0:
            torch.cuda.synchronize()
            time.sleep(self.pause)

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return self._step(model, inputs, num_items_in_batch)
        finally:
            self.rest()

    def _step(self, model, inputs, num_items_in_batch):
        # Com cap_memory, um pedido que não cabe falha em vez de transbordar para a RAM: é pulado.
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.skipped = getattr(self, "skipped", 0) + 1
            print(f"sem memória: pedido pulado ({self.skipped} até agora)", flush=True)
            return torch.zeros((), device=self.args.device)


def cap_memory():
    """Limita o PyTorch à memória livre da placa.

    Acima disso o Windows empresta RAM do sistema à GPU: o treino fica 3x mais lento e numa vez derrubou o
    PC. Com o limite, um pedido grande demais levanta OutOfMemoryError.
    """
    free, total = torch.cuda.mem_get_info()
    # O que o contexto CUDA, o cuBLAS e a área de trabalho usam fora do pool do PyTorch.
    budget = free - 700 * 2**20
    torch.cuda.set_per_process_memory_fraction(budget / total)
    print(f"PyTorch pode usar {budget / 2**20:.0f} de {total / 2**20:.0f} MiB")


def build_model(args):
    from peft import LoraConfig, get_peft_model

    # Sem isso a atenção cai no kernel "math", que monta a matriz inteira: 1,2 GB a mais num pedido de 3,5k
    # tokens. O Qwen3.5 precisa dele, porque os outros kernels recusam os formatos das suas 6 camadas de
    # atenção completa.
    if "qwen3.5" not in str(args.base).lower():
        torch.backends.cuda.enable_math_sdp(False)
    model = load_model(args.base)
    # Só o modelo de linguagem aprende (as camadas de visão se chamam qkv/proj/fc1/fc2).
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = LORA_TARGETS_QWEN35 if model.config.model_type.startswith("qwen3_5") else LORA_TARGETS
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    global DATASET, MAX_TOKENS
    MAX_TOKENS = args.max_tokens
    if args.dataset:
        DATASET = (ROOT / args.dataset).resolve()
    print(f"dataset: {DATASET}")
    cap_memory()
    train, val = load_examples(args)
    processor = load_processor(args.base)
    collator = Collator(processor)
    train, val = fits(processor, collator, Requests(train), "treino", args.heaviest), fits(processor, collator, Requests(val), "validação")
    model = build_model(args)

    out = ROOT / "runs" / args.name
    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # O transformers 5 tirou warmup_ratio; lá um warmup_steps menor que 1 é a fração dos passos.
        **({"warmup_ratio": 0.03} if "warmup_ratio" in TrainingArguments.__dataclass_fields__ else {"warmup_steps": 0.03}),
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps" if len(val) else "no",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        prediction_loss_only=True,
    )
    trainer = AnswerTrainer(model=model, args=targs, train_dataset=train, eval_dataset=val if len(val) else None, data_collator=collator)
    # A loss acima já é a média de um pedido; o Trainer tira a média entre os acumulados.
    trainer.model_accepts_loss_kwargs = False
    trainer.pause, trainer.pause_file = args.pause, out / "pause"
    trainer.train(resume_from_checkpoint=args.resume or None)
    model.save_pretrained(out / "final")
    processor.save_pretrained(out / "final")
    print(f"adaptador salvo em {out / 'final'}")


if __name__ == "__main__":
    main()

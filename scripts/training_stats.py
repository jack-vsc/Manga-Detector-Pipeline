"""Andamento de um treino lido do log (saída de train.py redirecionada para runs-<nome>.log).

Extrai o passo atual, o tempo que o tqdm conta e as losses que o Trainer imprime. Usado pelo painel.
"""
import ast
import re
import time

from common import ROOT

# Linha do tqdm: " 12%|█▏   | 154/1282 [1:23:45<10:11:12, 32.5s/it]"
PROGRESS = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),")
# Dicionários do Trainer, às vezes depois dos códigos de escape das barras do tqdm.
DICT = re.compile(r"\{'[^{}]*\}$")
# Sem nenhuma linha por esse tempo, com um passo levando ~2 min, o treino parou.
SILENT_FOR = 15 * 60
# Fim do treino e contagem de pedidos, nas versões em inglês e em português do log.
SAVED = ("adapter saved to", "adaptador salvo em")
COUNTS = (("train:", "val:"), ("treino:", "validação:"))


def seconds(clock):
    if "?" in clock:
        return None
    total = 0
    for part in clock.split(":"):
        total = total * 60 + int(part)
    return total


def compute(name="manga-translate"):
    log = ROOT / f"runs-{name}.log"
    if not log.exists():
        return {"state": "absent", "name": name}
    text = log.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in re.split(r"[\r\n]+", text) if l.strip()]
    bars = [(int(m[1]), int(m[2]), m[3], m[4]) for l in lines if "shards" not in l
            for m in [PROGRESS.search(l)] if m]
    # A avaliação desenha uma barra própria, mais curta; a do treino é a mais longa.
    total = max((b[1] for b in bars), default=0)
    step = [b for b in bars if b[1] == total]
    losses, evals = [], []
    for l in lines:
        if m := DICT.search(l):
            try:
                d = ast.literal_eval(m[0])
            except (ValueError, SyntaxError):
                continue
            # O transformers 5 imprime os números como texto ({'loss': '0.2557'}).
            for k, v in d.items():
                try:
                    d[k] = float(v) if isinstance(v, str) else v
                except ValueError:
                    pass
            if "loss" in d:
                losses.append({"epoch": d.get("epoch"), "loss": d["loss"], "lr": d.get("learning_rate")})
            if "eval_loss" in d:
                evals.append({"epoch": d.get("epoch"), "loss": d["eval_loss"]})
    quiet = time.time() - log.stat().st_mtime
    if any(s in text for s in SAVED):
        state = "done"
    elif "Traceback (most recent call last)" in text:
        state = "failed"
    elif (ROOT / "runs" / "paused").exists():
        state = "paused"  # system_stats.set_paused
    elif quiet > SILENT_FOR:
        state = "stopped"
    elif not step:
        state = "preparing"
    else:
        state = "training"
    out = {"state": state, "name": name, "quiet_s": round(quiet), "losses": losses, "evals": evals,
           "last_line": lines[-1][-300:] if lines else ""}
    if step:
        done, total, elapsed, left = step[-1]
        out |= {"step": done, "steps": total, "elapsed_s": seconds(elapsed), "left_s": seconds(left)}
        # Um treino retomado reinicia o relógio do tqdm: o tempo soma as execuções, o ritmo é o desta.
        runs, prev = [], None
        for b in step:
            t = seconds(b[2]) or 0
            if prev is None or t < prev:
                runs.append([None, 0, b[0], t])
            # A barra retomada mostra 0 e depois pula para o passo do checkpoint: conta a partir dali.
            if runs[-1][0] is None and b[0]:
                runs[-1][:2] = [b[0], t]
            runs[-1][2:] = [b[0], t]
            prev = t
        first, first_t, last, last_t = runs[-1]
        out["trained_s"] = sum(r[3] for r in runs)
        if first is not None and last > first:
            out["pace_s"] = (last_t - first_t) / (last - first)
            # A estimativa do tqdm usa só os últimos passos, e uma pausa a desvia em horas.
            out["left_s"] = round((total - done) * out["pace_s"])
    for l in lines:
        if any(l.startswith(a) and b in l for a, b in COUNTS):
            out["requests"] = l
    return out

"""Estado da máquina durante o treino, para o painel, e a pausa do treino.

Lê a GPU pelo nvidia-smi, a RAM que o Windows empresta ao treino como memória de GPU (o transbordo que o
deixava 3x mais lento), CPU e RAM. A pausa suspende e retoma os processos do treino inteiros, que seguem
exatamente de onde pararam; a memória da placa continua ocupada enquanto isso.

A temperatura da CPU não aparece: o Windows só a entrega a um administrador.
"""
import subprocess

import psutil

from common import ROOT

PAUSED = ROOT / "runs" / "paused"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def training_processes():
    """train.py: o lançador do venv, o interpretador e os workers do DataLoader."""
    out = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any(a.endswith("train.py") for a in p.info["cmdline"] or []):
                out += [p, *p.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return list({p.pid: p for p in out}.values())


def number(text):
    try:
        return float(text)
    except ValueError:
        return None


def gpu():
    fields = "name,utilization.gpu,temperature.gpu,clocks.gr,clocks.max.gr,power.draw,memory.used,memory.total"
    try:
        line = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    name, *values = [v.strip() for v in line.splitlines()[0].split(",")] if line else [None]
    if not name:
        return None
    util, temp, clock, clock_max, power, used, total = map(number, values)
    return {"name": name, "util": util, "temp": temp, "clock": clock, "clock_max": clock_max,
            "power": power, "mem_used": used, "mem_total": total}


def shared_mb(pids):
    """RAM do sistema emprestada a estes processos como memória de GPU."""
    if not pids:
        return None
    names = ",".join(f"'\\GPU Process Memory(pid_{pid}*)\\Shared Usage'" for pid in pids)
    cmd = (f"(Get-Counter {names} -ErrorAction SilentlyContinue).CounterSamples"
           " | Measure-Object CookedValue -Sum | ForEach-Object { [int]($_.Sum / 1MB) }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=20, creationflags=NO_WINDOW).stdout.strip()
        return int(out or 0)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None


def compute():
    procs = training_processes()
    mem, swap = psutil.virtual_memory(), psutil.swap_memory()
    freq = psutil.cpu_freq()
    train_ram = 0
    for p in procs:
        try:
            train_ram += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        "gpu": gpu(),
        "shared_mb": shared_mb([p.pid for p in procs]),
        "cpu": {"util": psutil.cpu_percent(interval=0.5), "ghz": freq.current / 1000 if freq else None,
                "cores": psutil.cpu_count()},
        "ram": {"used": mem.used, "total": mem.total, "swap_used": swap.used, "train": train_ram},
        "training": {"running": bool(procs), "paused": bool(procs) and PAUSED.exists()},
    }


def set_paused(paused):
    """Suspende ou retoma o treino; False se nenhum estiver rodando."""
    procs = training_processes()
    if not procs:
        PAUSED.unlink(missing_ok=True)
        return False
    for p in procs:
        try:
            p.suspend() if paused else p.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if paused:
        PAUSED.parent.mkdir(exist_ok=True)
        PAUSED.write_text(",".join(str(p.pid) for p in procs))
    else:
        PAUSED.unlink(missing_ok=True)
    return True

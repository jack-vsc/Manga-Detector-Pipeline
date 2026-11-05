"""Grava o estado da GPU, da CPU e da RAM uma vez por segundo num CSV, até ser interrompido.

Colunas: time, gpu_util %, gpu_mem MB, gpu_power W, gpu_temp C, gpu_clock MHz, cpu %, ram_used MB.

Uso: python scripts/gpu_monitor.py saida.csv
"""
import csv
import subprocess
import sys
import time

import psutil

QUERY = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"


def main():
    with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["time", "gpu_util", "gpu_mem", "gpu_power", "gpu_temp", "gpu_clock", "cpu", "ram_used"])
        psutil.cpu_percent()
        while True:
            started = time.time()
            try:
                line = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
            except (OSError, subprocess.TimeoutExpired):
                line = [""] * 5
            out.writerow([round(started, 1), *line, psutil.cpu_percent(), psutil.virtual_memory().used // 2**20])
            f.flush()
            time.sleep(max(0, 1 - (time.time() - started)))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Testes dos modelos locais nos capítulos 96 e 97, um depois do outro (a GPU roda um de cada vez).
# Cada etapa grava um traço de GPU, CPU e RAM em metrics/<etapa>.csv e o início e o fim em metrics/steps.tsv.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
T="$P/test-soccer"
PY="$P/.venv/Scripts/python.exe"
MANGA_INDEX="${MANGA_INDEX:-/d/Dev/Manga Index}"
mkdir -p "$T/metrics"
cd "$P"

step() {  # step <nome> <comando...>
  local name=$1; shift
  "$PY" scripts/gpu_monitor.py "$T/metrics/$name.csv" & local mon=$!
  local start; start=$(date +%s.%N)
  echo "== $name começou $(date +%T)"
  "$@"
  local rc=$?
  echo -e "$name\t$start\t$(date +%s.%N)\t$rc" >> "$T/metrics/steps.tsv"
  kill $mon 2>/dev/null
  echo "== $name terminou ($rc) $(date +%T)"
}

render() {  # render <nome> <url-base|modelo>: os dois capítulos pelo app, como um leitor receberia
  local name=$1 model=$2
  for c in 96 97; do
    (cd "$MANGA_INDEX" && MI_USER_DATA="$T/userdata" MI_DATASET_DIR="$T/dataset-render-$name" \
      MI_EVAL_MODEL="$model" MI_EVAL_PDF="$T/I'm Not a Soccer Genius! Chapter $c.pdf" \
      MI_EVAL_OUT="$T/render/$name/ch$c" npx electron-vite dev > "$T/render-$name-ch$c.log" 2>&1)
  done
}

unload() { ollama stop gemma4-e4b-16k >/dev/null 2>&1; ollama stop gemma4-e2b-16k >/dev/null 2>&1; sleep 5; }

# 1. E2B nos mesmos 111 pedidos que o Claude respondeu.
unload
step bench-gemma-e2b "$PY" scripts/compare_models.py gemma --ollama-model gemma4-e2b-16k --name gemma-e2b
unload

# 2. Capítulos inteiros no app: E4B, E2B e o Qwen treinado (Q4_0 no llama.cpp, como os arquivos do Gemma).
step render-gemma-e4b render gemma-e4b "http://localhost:11434/v1|gemma4-e4b-16k"
unload
step render-gemma-e2b render gemma-e2b "http://localhost:11434/v1|gemma4-e2b-16k"
unload
"$P/tools/llama-bin/llama-server.exe" -m runs/manga-translate/gguf/qwen-manga-q4_0.gguf \
  --mmproj runs/manga-translate/gguf/qwen-manga-mmproj-f16.gguf -c 16384 -ngl 99 --temp 0 \
  --host 127.0.0.1 --port 8771 > "$T/llama-server-render.log" 2>&1 & SERVER=$!
until curl -s localhost:8771/health | grep -q ok; do sleep 2; done
step render-qwen-q4 render qwen-q4 "http://localhost:8771/v1|qwen-manga-q4"
kill $SERVER
cd "$P/scripts" && "$PY" full_report.py
echo "FIM"

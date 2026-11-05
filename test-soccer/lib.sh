#!/usr/bin/env bash
# Funções comuns aos testes dos capítulos 96 e 97. Os scripts de teste fazem `source` deste arquivo.
#
# Cada etapa grava um traço de GPU, CPU e RAM em metrics/<etapa>.csv e o início e o fim em
# metrics/steps.tsv, que scripts/full_report.py lê.

P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$P/test-soccer"
PY="$P/.venv/Scripts/python.exe"
LLAMA="$P/tools/llama-bin/llama-server.exe"
MANGA_INDEX="${MANGA_INDEX:-/d/Dev/Manga Index}"
# Build congelado do Manga Index (electron-vite build --outDir test-soccer/app-out). O servidor de
# desenvolvimento recarrega o app a cada arquivo salvo e interrompia os testes no meio.
APP="${APP:-$T/app-out}"
CHAPTERS="96 97"
mkdir -p "$T/metrics"

step() {  # step <nome> <comando...>
  local name=$1; shift
  "$PY" "$P/scripts/gpu_monitor.py" "$T/metrics/$name.csv" & local mon=$!
  local start; start=$(date +%s.%N); echo "== $name começou $(date +%T)"
  "$@"; local rc=$?
  sed -i "/^$name\t/d" "$T/metrics/steps.tsv" 2>/dev/null
  echo -e "$name\t$start\t$(date +%s.%N)\t$rc" >> "$T/metrics/steps.tsv"
  kill $mon 2>/dev/null; echo "== $name terminou ($rc) $(date +%T)"
}

render() {  # render <nome> <url-base|modelo>: os capítulos pelo app, como um leitor receberia
  local name=$1 model=$2 data="$T/userdata-$1"
  # Pasta do app vazia a cada execução: sem respostas guardadas de tentativas anteriores.
  rm -rf "$data"; mkdir -p "$data" && cp "$T/userdata/settings.json" "$T/userdata/Local State" "$data/"
  mkdir -p "$T/old-cached"; [ -d "$T/render/$name" ] && mv "$T/render/$name" "$T/old-cached/render-$name-$(date +%s)"
  for c in $CHAPTERS; do
    (cd "$MANGA_INDEX" && MI_USER_DATA="$data" MI_DATASET_DIR="$T/dataset-render-$name" \
      MI_EVAL_MODEL="$model" MI_EVAL_PDF="$T/I'm Not a Soccer Genius! Chapter $c.pdf" MI_EVAL_OUT="$T/render/$name/ch$c" \
      npx electron-vite preview --skipBuild --outDir "$APP" --entry "$APP/main/index.js" > "$T/render-$name-ch$c.log" 2>&1)
  done
}

serve() {  # serve <modelo.gguf> <mmproj.gguf> [opções do llama-server]: sobe na porta 8771 e espera carregar
  local model=$1 mmproj=$2; shift 2
  "$LLAMA" -m "$model" --mmproj "$mmproj" -c 16384 -np 1 -ngl 99 --temp 0 "$@" \
    --host 127.0.0.1 --port 8771 > "$T/llama-server.log" 2>&1 & SERVER=$!
  until curl -s localhost:8771/health | grep -q ok; do
    kill -0 $SERVER 2>/dev/null || { echo "llama-server não abriu (veja test-soccer/llama-server.log)"; return 1; }
    sleep 2
  done
}

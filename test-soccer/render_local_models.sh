#!/usr/bin/env bash
# Capítulos 96 e 97 pelo Manga Index com os modelos locais, um de cada vez: Gemma 4 E4B e E2B no Ollama e o
# Qwen2-VL treinado em Q4_0 no llama.cpp. No fim, gera o relatório.
set -u
source "$(dirname "$0")/lib.sh"
cd "$P"

step render-gemma-e4b render gemma-e4b "http://localhost:11434/v1|gemma4-e4b-16k"
ollama stop gemma4-e4b-16k >/dev/null 2>&1; sleep 5
step render-gemma-e2b render gemma-e2b "http://localhost:11434/v1|gemma4-e2b-16k"
ollama stop gemma4-e2b-16k >/dev/null 2>&1; sleep 5
# Um pedido por vez com os 16k inteiros, como o Ollama dá ao Gemma (com 4 em paralelo as páginas longas não cabiam).
serve runs/manga-translate/gguf/qwen-manga-q4_0.gguf runs/manga-translate/gguf/qwen-manga-mmproj-f16.gguf || exit 1
step render-qwen-q4 render qwen-q4 "http://localhost:8771/v1|qwen-manga-q4"
kill $SERVER
cd "$P/scripts" && "$PY" full_report.py
echo "FIM"

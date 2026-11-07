#!/usr/bin/env bash
# Capítulos 96 e 97 pelo Manga Index com o Qwen3.5-2B treinado (runs/q35-2b, Q8_0, sem raciocínio) num
# llama-server à parte, como render_local_models.sh faz com os outros modelos.
set -u
source "$(dirname "$0")/lib.sh"
cd "$P"
G="$P/runs/q35-2b/gguf"
serve "$G/q35-manga-q8_0.gguf" "$G/q35-manga-mmproj-f16.gguf" --chat-template-kwargs '{"enable_thinking":false}' || exit 1
step render-qwen35-2b render qwen35-2b "http://localhost:8771/v1|qwen35-manga"
kill $SERVER
echo "FIM"

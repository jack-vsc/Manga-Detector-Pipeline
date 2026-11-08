#!/usr/bin/env bash
# Capítulos 96 e 97 com o Qwen3.5-2B treinado pelo perfil "Neste PC" do Manga Index: o próprio app sobe o
# llama-server (contexto de 16k para este modelo, cache em 8 bits, --fit, --reasoning-budget 0).
#
# A pasta do app recebe o motor (junction para tools/llama-bin) e um source.json apontando para os GGUF de
# runs/q35-2b, como o app faria depois de instalar o modelo.
set -u
source "$(dirname "$0")/lib.sh"
APP="${APP_Q35:-$T/app-q35}"
cd "$P"
name=qwen35-2b-app; data="$T/userdata-$name"
rm -rf "$data"; mkdir -p "$data/local-ai/engine" "$data/local-ai/models/qwen35-manga-2b"
cp "$T/userdata/settings.json" "$T/userdata/Local State" "$data/"
cmd //c mklink //J "$(cygpath -w "$data/local-ai/engine/cuda")" "$(cygpath -w "$P/tools/llama-bin")" >/dev/null
G="$P/runs/q35-2b/gguf"
printf '{"model":"%s","mmproj":"%s","fromOllama":false}' "$G/q35-manga-q8_0.gguf" "$G/q35-manga-mmproj-f16.gguf" \
  > "$data/local-ai/models/qwen35-manga-2b/source.json"
mkdir -p "$T/old-cached"; [ -d "$T/render/$name" ] && mv "$T/render/$name" "$T/old-cached/render-$name-$(date +%s)"

chapters() {
  for c in $CHAPTERS; do
    (cd "$MANGA_INDEX" && MI_USER_DATA="$data" MI_DATASET_DIR="$T/dataset-render-$name" \
      MI_EVAL_MODEL="local|qwen35-manga-2b" MI_EVAL_PDF="$T/I'm Not a Soccer Genius! Chapter $c.pdf" MI_EVAL_OUT="$T/render/$name/ch$c" \
      npx electron-vite preview --skipBuild --outDir "$APP" --entry "$APP/main/index.js" > "$T/render-$name-ch$c.log" 2>&1)
  done
}
step "render-$name" chapters
echo "FIM"

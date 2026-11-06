#!/usr/bin/env bash
# Os modelos Qwen originais (sem treino) e os treinados nos mesmos 111 pedidos dos outros, cada um em GGUF
# no llama.cpp, para separar o ganho do treino do ganho das gerações mais novas.
#
# Os arquivos dos modelos base ficam em models/base-gguf/ (baixados do Hugging Face).
set -u
source "$(dirname "$0")/lib.sh"
M="$P/models/base-gguf"
cd "$P"

bench() {  # bench <nome> <modelo.gguf> <mmproj.gguf> [opções do llama-server]
  local name=$1 model=$2 mmproj=$3; shift 3
  # Um modelo que já respondeu tudo mantém os resultados e o traço da GPU (rodar de novo sobrescreveria o traço).
  if [ -f "$T/results/$name.jsonl" ] && [ "$(grep -c . "$T/results/$name.jsonl")" -ge 111 ]; then echo "$name: já feito"; return 0; fi
  serve "$M/$model" "$M/$mmproj" "$@" || return 1
  step "bench-$name" "$PY" scripts/compare_models.py server --name "$name"
  kill $SERVER; sleep 5
}

bench qwen2-vl-2b Qwen2-VL-2B-Instruct-Q8_0.gguf mmproj-Qwen2-VL-2B-Instruct-f16.gguf
# O codificador de imagem do Qwen2.5-VL responde "GGGG..." na GPU em pedidos com muitas imagens grandes: roda na CPU.
bench qwen25-vl-3b Qwen2.5-VL-3B-Instruct-Q8_0.gguf mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf --no-mmproj-offload
bench qwen3-vl-2b Qwen3VL-2B-Instruct-Q8_0.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf
# O 4B em Q8_0 transborda a placa de 6 GB (uns 4 minutos por pedido); em Q4_K_M cabe.
bench qwen3-vl-4b Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-F16.gguf
# Qwen3.5 (visão embutida) sem raciocínio, porque o app precisa da resposta direto.
NOTHINK=(--chat-template-kwargs '{"enable_thinking":false}')
bench qwen35-2b Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf Qwen3.5-2B-GGUF/mmproj-F16.gguf "${NOTHINK[@]}"
bench qwen35-4b Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf Qwen3.5-4B-GGUF/mmproj-F16.gguf "${NOTHINK[@]}"
# Qwen3.5-2B treinado no dataset-pt (runs/q35-2b), Q8_0, sem raciocínio como no treino.
bench qwen35-2b-trained ../../runs/q35-2b/gguf/q35-manga-q8_0.gguf ../../runs/q35-2b/gguf/q35-manga-mmproj-f16.gguf "${NOTHINK[@]}"
echo "FIM"

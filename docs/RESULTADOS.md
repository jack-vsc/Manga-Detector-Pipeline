# Resultados

Todos os testes usam os capítulos 96 e 97 de *I'm Not a Soccer Genius!*, que nunca entraram em nenhum
treino. Máquina: Windows 11, RTX 4050 Laptop 6 GB, temperatura 0, um pedido por vez, contexto de 16k
(exceto onde indicado). Números brutos: `test-soccer/results/metricas.json`, gerado por
`scripts/full_report.py`.

## 1. Os mesmos 111 pedidos

Cada modelo recebe exatamente as imagens e o prompt que o Claude recebeu. Das 346 falas que o Claude
traduziu, conta-se quantas o modelo perdeu, quantas escreveu quase igual (similaridade ≥ 70%) e quantas
escreveu bem diferente (< 35%: outro balão ou outro sentido). Uma resposta "quebrada" é a que não tem JSON
legível.

| modelo | arquivo | quebradas | perdidas | quase iguais | bem diferentes | s/pedido | pico de VRAM |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B | Ollama Q4 | 4 | 50 | 145 | 73 | 34,2 | 4,9 GB |
| Gemma 4 E2B | Ollama Q4 | 14 | 103 | 111 | 74 | 19,6 | 4,5 GB |
| Qwen2-VL-2B original | Q8_0 | 11 | 204 | 31 | 73 | 8,5 | 4,0 GB |
| Qwen2.5-VL-3B original (visão na CPU) | Q8_0 | 12 | 98 | 81 | 78 | 57,0 | 4,3 GB |
| Qwen3-VL-2B original | Q8_0 | 8 | 47 | 120 | 71 | 12,3 | 5,3 GB |
| Qwen3-VL-4B original | Q4_K_M | 2 | 27 | 177 | 45 | 27,5 | 5,5 GB |
| Qwen3.5-2B original | Q8_0 | 8 | 30 | 169 | 29 | 8,1 | 3,5 GB |
| Qwen3.5-4B original | Q4_K_M | 4 | 24 | 217 | 13 | 12,9 | 4,5 GB |
| Qwen2-VL-2B treinado | PyTorch 4 bits | 3 | 72 | 184 | 11 | 31,2 | 5,8 GB |
| Qwen2-VL-2B treinado | Q8_0 | 6 | 69 | 191 | 13 | 6,4 | 4,2 GB |
| Qwen2-VL-2B treinado | Q4_0 | 12 | 103 | 172 | 11 | 7,2 | 3,3 GB |
| **Qwen3.5-2B treinado** | **Q8_0** | **1** | **45** | **232** | **6** | **6,0** | **3,7 GB** |
| Qwen3.5-2B treinado v2 (menos repescagens vazias) | Q8_0 | 2 | 56 | 227 | 7 | 6,3 | 3,5 GB |

- O treino quase zera as respostas "bem diferentes" em todas as gerações: o modelo aprende a juntar as
  caixas de cada balão como o Claude junta.
- Entre os modelos sem treino, o Qwen3.5 é o melhor por tamanho. O 2B original já perde menos falas que o
  Gemma 4 E4B, em um quarto do tempo.
- Energia da GPU nos 111 pedidos: Qwen3.5-2B treinado 9,1 Wh, Qwen3.5-2B original 13,4 Wh,
  Gemma 4 E2B 25,9 Wh, Qwen3-VL-4B 33,9 Wh, Qwen2.5-VL-3B 42,4 Wh.

## 2. Capítulos inteiros no Manga Index

O app faz tudo como para um leitor: detecta, pede, repesca, apaga e desenha. As 330 falas do Claude nas
28 páginas são pareadas com as do modelo pela posição na página. "Texto deixado em inglês" conta as áreas
que o detector achou e que ficaram sem tradução (inclui créditos e anúncios, que o Claude também deixa:
118 áreas).

| modelo | perdidas | quase iguais | bem diferentes | deixado em inglês | pedidos | s/pedido | cortados no limite | tempo | energia | pico de VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E4B (Q4) | 33 | 202 | 29 | 105 | 99 | 9,5 | 0 | 25,0 min | 20,7 Wh | 5,6 GB |
| Gemma 4 E2B (Q4) | 110 | 127 | 35 | 186 | 126 | 4,9 | 1 | 19,3 min | 14,6 Wh | 4,7 GB |
| Qwen2-VL-2B treinado (Q4_0) | 54 | 186 | 18 | 150 | 132 | 10,7 | 28 | 32,4 min | 30,1 Wh | 5,6 GB |
| Qwen3.5-2B treinado, llama-server à parte (Q8_0) | 38 | 232 | 5 | 182 | 121 | 4,5 | 0 | 18,8 min | 14,3 Wh | 5,7 GB |
| **Qwen3.5-2B treinado, perfil "Neste PC" (Q8_0)** | **31** | **234** | **5** | 187 | 120 | **4,4** | **0** | **17,9 min** | **14,0 Wh** | **4,9 GB** |

- A repescagem do app recupera boa parte das caixas que o modelo pula nos pedidos de página: de 45 falas
  perdidas no teste isolado para 31-38 dentro do app.
- No perfil "Neste PC" o app guarda o contexto em 8 bits e usa `--fit`; o pico de VRAM cai 0,8 GB em
  relação ao llama-server à parte, sem piorar as respostas.
- "Cortados no limite" são respostas que chegaram ao máximo de tokens, quase sempre por repetição. O
  Qwen2-VL treinado em Q4_0 entrou em repetição 28 vezes; o Qwen3.5, nenhuma.

## 3. Treinos

| treino | base | dataset | pedidos de treino | passos | tempo | loss de treino | loss de validação |
|---|---|---|---|---|---|---|---|
| `manga-translate` | Qwen2-VL-2B-Instruct | replay do app (171 capítulos) | 10.256 | 1.282 | 9,3 h | 0,225 | 0,155 |
| `q35-2b` | Qwen3.5-2B | dataset-pt (174 capítulos) | 12.295 | 1.519 | 9,5 h | 0,152 | 0,157 |
| `q35-2b-v2` | Qwen3.5-2B | dataset-pt, 20% das repescagens vazias | 10.616 | 1.309 | 8,8 h | 0,148 | 0,158 |

Uma época, batch 1 com acúmulo de 8, lr 2e-4 com cosseno, LoRA r=16. A loss de validação vem de 150
pedidos de capítulos que ficam sempre fora do treino. Pedidos pulados por falta de memória: 31 no
Qwen2-VL e 134 e 163 nos treinos do Qwen3.5.

## 4. Datasets

| dataset | origem | capítulos | páginas | pedidos | falas traduzidas | custo |
|---|---|---|---|---|---|---|
| `training-dataset` do app | cache de traduções feitas antes pelo app | 171 | 5.239 | 10.833 | 23.803 | 0 |
| `dataset-pt/final` | capítulos novos pela API do Claude Haiku | 174 (9 obras) | 8.640 pedidos de página | 12.473 | 22.587 | US$ 7,60 |

No `dataset-pt`, 1.121 respostas tinham uma nota antes do JSON (removida) e 5 não tinham JSON legível
(descartadas). 2.122 das 3.838 repescagens voltaram vazias.

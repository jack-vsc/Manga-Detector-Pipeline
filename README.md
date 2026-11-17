# Manga Detector Pipeline

Destilação do tradutor de balões do [Manga Index](#integração-com-o-manga-index) para um modelo pequeno que
roda numa placa de vídeo de 6 GB. O Manga Index detecta o texto de cada página de mangá ou webtoon em
inglês e pede a tradução para português do Brasil a um modelo com visão. Com o Claude Haiku pela API a
tradução é boa; localmente, o Gemma 4 E4B ocupava quase a placa inteira e o E2B perdia falas demais.

Este repositório tem tudo o que levou de um ao outro: a gravação dos pedidos que o app faz ao Claude como
dataset, o treino LoRA em 4 bits, a conversão para GGUF, o benchmark contra o Claude e os modelos locais e
o painel que acompanha dataset, treino e máquina.

```
página em inglês ──► Manga Index: detector ──► caixas numeradas + imagem-chave + prompt
                                                    │
               Claude Haiku (professor) ◄──────────┤  dataset: pedido + resposta JSON
                                                    │
     Qwen3.5-2B + LoRA (aluno, 4 bits) ◄── treino ──┘
                    │
       fusão ──► GGUF Q8_0 ──► llama-server dentro do Manga Index ("Neste PC")
```

## Resultado principal

Capítulos 96 e 97 de *I'm Not a Soccer Genius!*, fora de todos os treinos, traduzidos inteiros pelo app
(detecção, pedidos, repescagem, apagamento e desenho), comparados fala a fala com o Claude. RTX 4050
Laptop 6 GB:

| modelo | falas perdidas (de 330) | quase iguais às do Claude | bem diferentes | tempo (2 caps) | energia da GPU | pico de VRAM |
|---|---|---|---|---|---|---|
| Gemma 4 E4B (Ollama, Q4) | 33 | 202 | 29 | 25,0 min | 20,7 Wh | 5,6 GB |
| Gemma 4 E2B (Ollama, Q4) | 110 | 127 | 35 | 19,3 min | 14,6 Wh | 4,7 GB |
| Qwen2-VL-2B treinado (Q4_0) | 54 | 186 | 18 | 32,4 min | 30,1 Wh | 5,6 GB |
| **Qwen3.5-2B treinado, integrado ao app (Q8_0)** | **31** | **234** | **5** | **17,9 min** | **14,0 Wh** | **4,9 GB** |

- "Quase iguais" = similaridade ≥ 70% com a frase do Claude; "bem diferentes" = < 35% (outro balão ou
  outro sentido). O pico de VRAM inclui o detector e o inpainting do próprio app.
- O Qwen3.5-2B treinado ganha do Gemma 4 E4B em todas as colunas, com metade do tamanho
  (2,0 GB + 0,7 GB de visão) e 30% menos tempo.
- No benchmark de 111 pedidos isolados (sem a repescagem do app), o treino levou o Qwen3.5-2B de
  169 para 232 falas quase iguais e de 29 para 6 bem diferentes, mas de 30 para 45 falas perdidas. As
  perdas vêm de caixas que o modelo marca como "ignorar" nos pedidos de página; o app pergunta de novo
  por elas, e por isso dentro do app o número cai para 31.

Tabelas completas, incluindo Qwen2-VL, Qwen2.5-VL, Qwen3-VL e Qwen3.5 sem treino, estão em
[docs/RESULTADOS.md](docs/RESULTADOS.md). O caminho até aqui, inclusive o que não deu certo, está no
[diário](docs/ITERACOES.md).

### Limites

- Um par de capítulos de uma obra só serve de teste; os números servem para comparar modelos entre si,
  não para prometer a qualidade em qualquer mangá.
- O aluno copia o professor: a qualidade fica no máximo no nível do Claude Haiku, e os erros dele
  (falas puladas, créditos traduzidos) passam junto.
- Tudo foi medido numa máquina só (Windows 11, RTX 4050 Laptop 6 GB, CUDA 12).

## Instalação

Windows, Python 3.11 e GPU NVIDIA. São dois ambientes, porque o Qwen3.5 só existe no transformers 5 e o
Qwen2-VL foi treinado no 4:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe --index-strategy unsafe-best-match -r requirements.txt

uv venv .venv-q35 --python 3.11
uv pip install --python .venv-q35/Scripts/python.exe --index-strategy unsafe-best-match -r requirements-qwen35.txt
```

Também são usados:

- o [Manga Index](#integração-com-o-manga-index) com o modo de avaliação (`MI_EVAL_*`), para gravar o
  dataset e traduzir os capítulos de teste;
- o [llama.cpp](https://github.com/ggml-org/llama.cpp) build b11870 em `tools/` (`llama-server`,
  `llama-quantize` e `convert_hf_to_gguf.py`);
- o [Ollama](https://ollama.com), só para comparar com o Gemma (`test-soccer/Modelfile.*`).

## Uso

```bash
# 1. Dataset: os pedidos que o Manga Index fez ao Claude, gravados pelo próprio app
powershell -File scripts/replay_manga_index.ps1 -Pages 5000      # do cache de tradução, sem custo
python scripts/build_pt_dataset.py --workers 3                     # capítulos novos, com a API do Claude
python scripts/finish_pt_dataset.py                                # só o JSON de cada resposta

# 2. Treino (Qwen3.5-2B, ~9,5 h numa RTX 4050 6 GB) e painel ao vivo
.venv-q35/Scripts/python scripts/train.py --name q35-2b --base Qwen/Qwen3.5-2B \
    --dataset dataset-pt/final --max-tokens 2600 > runs-q35-2b.log 2>&1
python scripts/dataset_panel.py --dataset dataset-pt/final --run q35-2b

# 3. GGUF para o llama.cpp
.venv-q35/Scripts/python scripts/merge_adapter.py --base Qwen/Qwen3.5-2B \
    --adapter runs/q35-2b/final --out runs/q35-2b/merged
python tools/llama.cpp-b11870/convert_hf_to_gguf.py runs/q35-2b/merged --outfile runs/q35-2b/gguf/q35-manga-f16.gguf
python tools/llama.cpp-b11870/convert_hf_to_gguf.py runs/q35-2b/merged --mmproj --outfile runs/q35-2b/gguf/q35-manga-mmproj-f16.gguf
tools/llama-bin/llama-quantize runs/q35-2b/gguf/q35-manga-f16.gguf runs/q35-2b/gguf/q35-manga-q8_0.gguf Q8_0

# 4. Avaliação
bash test-soccer/run_base_bench.sh          # 111 pedidos, cada modelo num llama-server
bash test-soccer/render_qwen35_app.sh       # capítulos inteiros pelo perfil "Neste PC" do app
python scripts/full_report.py               # relatório, métricas e PDFs lado a lado
```

O Qwen3.5 precisa rodar sem raciocínio: no llama-server,
`--chat-template-kwargs '{"enable_thinking":false}'` (ou `--reasoning-budget 0`), que é como ele foi
treinado.

## Integração com o Manga Index

O Manga Index (projeto separado, em `D:\Dev\Manga Index` ou na variável `MANGA_INDEX`) traz o próprio
llama-server e uma lista de modelos locais em `src/main/localAi.ts`. O modelo treinado entra nessa lista
como `qwen35-manga-2b`, com contexto de 16k (as páginas com várias fatias chegam a ~9,3 mil tokens, e o
padrão de 8k do Gemma não basta). Os arquivos ficam em
`%APPDATA%\Manga Index\local-ai\models\qwen35-manga-2b\`. Detalhes em
[docs/ARQUITETURA.md](docs/ARQUITETURA.md#integração-com-o-manga-index).

## Estrutura

```
scripts/        dataset, treino, servidor, comparação, relatório e painel (detalhes em docs/ARQUITETURA.md)
test-soccer/    scripts de teste dos capítulos 96 e 97 e Modelfiles do Ollama
docs/           arquitetura, diário de iterações e resultados
requirements*.txt
```

Ficam fora do git, por tamanho ou por direitos de terceiros: `data/`, `dataset-pt/`, `models/`, `runs/`,
`tools/` e o conteúdo gerado em `test-soccer/` (capítulos, páginas traduzidas, respostas e builds do app).

## Licenças e dados

O código deste repositório está sob a [licença MIT](LICENSE).

Nada além do código é distribuído aqui:

- **Páginas de mangá e webtoon** são de seus autores e editoras. Datasets, capítulos de teste e páginas
  traduzidas ficam só na máquina local.
- **Respostas do Claude** usadas como dataset vêm da API da Anthropic e estão sujeitas aos termos de uso
  da Anthropic. Leia os termos antes de publicar um dataset ou pesos treinados com elas.
- **Modelos de terceiros** são baixados de suas fontes e seguem as próprias licenças, indicadas no cartão
  de cada modelo: Qwen2-VL-2B, Qwen3-VL e Qwen3.5 (Alibaba, Apache 2.0), Qwen2.5-VL-3B (licença Qwen
  Research, só para pesquisa), Gemma 4 (Google). O llama.cpp é MIT. O detector de texto usado pelo Manga
  Index tem licença própria e não faz parte deste repositório.
- **Pesos treinados** não são publicados aqui. Quem treinar com estes scripts herda a licença do modelo
  base e as condições dos dados usados.

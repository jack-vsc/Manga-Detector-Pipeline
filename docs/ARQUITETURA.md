# Arquitetura

Como as peças se encaixam, do pedido que o Manga Index faz ao Claude até o modelo treinado rodando dentro
do app. Os resultados estão em [RESULTADOS.md](RESULTADOS.md); as decisões e o que deu errado, em
[ITERACOES.md](ITERACOES.md).

## Fluxo

```
Manga Index (modo de avaliação ou replay)
  detector ──► página com caixas numeradas + imagem-chave + prompt ──► Claude Haiku
                                    │                                     │
                                    └──── grava pedido + resposta ◄───────┘
                                          training-dataset/  ou  dataset-pt/data/
                                                    │
                    finish_pt_dataset.py ──► dataset-pt/final/  (só o JSON das respostas)
                                                    │
                            train.py ──► runs/<nome>/final  (adaptador LoRA)
                                                    │
                   merge_adapter.py ──► runs/<nome>/merged  (bf16)
                                                    │
   convert_hf_to_gguf.py + llama-quantize ──► runs/<nome>/gguf/*.gguf  (Q8_0 + mmproj f16)
                                                    │
        llama-server ──► compare_models.py (111 pedidos)  /  Manga Index (capítulos inteiros)
                                                    │
                     compare_report.py, full_report.py ──► test-soccer/results/
```

## O pedido de tradução

O modelo aprende exatamente o que o Manga Index envia, para que o aluno troque o professor sem mudança no
app. Um pedido de página tem:

- **as imagens da página** (uma ou várias fatias, no caso de webtoon) com cada caixa de texto marcada em
  magenta e numerada;
- **a imagem-chave**, com cada caixa recortada e numerada, para as letras ficarem legíveis;
- **o prompt do app**, com as regras de tradução e o formato da resposta.

A resposta é um JSON `{"blocks": [{"ids": [...], "kind": ..., "original": ..., "translation": ...}],
"skip": [...]}`: cada bloco junta as caixas de um balão. `skip` lista as caixas sem nada a traduzir
(onomatopeias, créditos, marcas d'água). Quando sobram caixas sem bloco nem `skip`, o app faz uma
**repescagem**: um pedido só com as caixas que faltaram.

O app grava cada pedido do Claude em `training-dataset/` num formato parecido com o do Manga109:
`images/` com as imagens enviadas e `annotations/<capítulo>.jsonl` com uma linha por pedido (`key`,
`chapter`, `page`, `kind` = `page` ou `followup`, `images`, `prompt`, `answer`, `at`).

## Scripts

| script | função |
|---|---|
| `common.py` | caminhos, modelo base padrão e orçamentos de tokens por imagem |
| `model.py` | carrega o modelo em 4 bits (nf4) com o adaptador; corta as imagens no tamanho do treino |
| `replay_manga_index.ps1` | refaz no app os capítulos já traduzidos, a partir do cache, e grava o dataset sem chamar modelo |
| `dataset_stats.py`, `dataset_remove.py` | resumo do dataset e remoção de capítulos |
| `library_pairs.py` | obras da biblioteca do app com capítulos em inglês e em português |
| `pt_align.py` | localiza a página em inglês na edição em português, pela imagem |
| `build_pt_dataset.py` | roda o app em modo de avaliação em cada capítulo, com um proxy local até a API da Anthropic |
| `finish_pt_dataset.py` | deixa só o objeto JSON de cada resposta e grava `dataset-pt/final` |
| `train.py` | treino LoRA com loss só na resposta, corte de pedidos longos, pausa da GPU e retomada |
| `memory_probe.py` | mede o maior pedido que cabe no treino |
| `merge_adapter.py` | funde o adaptador no modelo base para a conversão em GGUF |
| `qwen_server.py` | modelo treinado (PyTorch) atrás de um endpoint no formato da OpenAI |
| `compare_models.py` | roda Qwen (PyTorch), Gemma (Ollama) ou qualquer llama-server nos pedidos do teste |
| `compare_report.py` | comparativo caixa a caixa em HTML e as métricas por modelo |
| `full_report.py` | relatório dos capítulos inteiros, traços de GPU e PDFs lado a lado |
| `gpu_monitor.py` | traço de GPU, CPU e RAM por segundo |
| `dataset_panel.py`, `.html` | painel ao vivo do dataset, do treino e da máquina, com pausa |
| `training_stats.py`, `system_stats.py` | leitura do log do treino e dos contadores da máquina |

`test-soccer/lib.sh` concentra o que os testes compartilham: `step` (roda uma etapa com traço de GPU),
`render` (traduz os capítulos pelo app numa pasta de dados vazia) e `serve` (sobe um llama-server).

## Treino

- **Base em 4 bits (nf4, duplo) + LoRA r=16, alfa=32** no modelo de linguagem: atenção, MLP e, no
  Qwen3.5, as camadas de atenção linear (`in_proj_qkv`, `in_proj_z`, `out_proj`). A parte de visão fica
  congelada.
- **Loss só na resposta.** O collator mascara prompt, imagens e padding. `AnswerTrainer` calcula os
  logits só nas posições da resposta: com 151 mil tokens no vocabulário, os logits de um pedido inteiro
  passam de 2,5 GB.
- **Orçamento de imagem.** A chave fica com até 640 tokens e as páginas dividem 1.024. Um pedido acima do
  limite recebe páginas menores (768, 512, 256) antes de ficar de fora. No Qwen3.5 o limite é 2.600
  tokens, porque acima disso estoura a memória da placa.
- **Memória.** `cap_memory` limita o PyTorch à memória livre da placa. Sem isso o Windows empresta RAM à
  GPU, o treino fica 3x mais lento e numa vez derrubou o PC. Um pedido que não cabe é pulado.
- **Calor.** `--pause` dá um descanso à GPU depois de cada pedido; `runs/<nome>/pause` muda o valor com o
  treino rodando, e o painel suspende e retoma os processos.
- **Validação.** 3% dos capítulos, escolhidos por hash do nome, ficam sempre fora do treino; até 150
  pedidos deles são medidos a cada 200 passos.
- **Qwen3.5 sem raciocínio.** A resposta é treinada depois de `<think>\n\n</think>\n\n`, a mesma
  abertura que o llama-server monta com `enable_thinking=false`.

## Avaliação

- **111 pedidos.** O Claude traduziu os capítulos 96 e 97 pelo app; os mesmos pedidos (mesmas imagens e
  mesmo prompt) vão para cada modelo. Uma fala do Claude conta como perdida quando nenhum bloco do modelo
  cobre as mesmas caixas.
- **Capítulos inteiros.** O app traduz os capítulos com cada modelo como faria para um leitor, com
  repescagem, apagamento e desenho. Como o detector é o mesmo, os blocos são pareados com os do Claude
  pela posição na página (IoU > 0,3).
- **Similaridade.** `difflib.SequenceMatcher` entre as traduções em minúsculas: ≥ 0,7 é "quase igual" e
  < 0,35 é "bem diferente".

## Integração com o Manga Index

O Manga Index baixa o llama.cpp e roda o `llama-server` sozinho no perfil "Neste PC"
(`src/main/localAi.ts`). Para usar o modelo treinado:

1. Na lista `MODELS` de `localAi.ts`, uma entrada `qwen35-manga-2b` com o tamanho e o SHA-256 dos dois
   arquivos e `ctx: 16384`. Hoje a entrada não tem URL de download, porque os pesos não foram publicados.
2. O servidor usa `--ctx-size` do modelo (8.192 quando não há valor), cache em 8 bits, `--fit on` e
   `--reasoning-budget 0`, que desliga o raciocínio do Qwen3.5.
3. Os arquivos vão para `%APPDATA%\Manga Index\local-ai\models\qwen35-manga-2b\` (`model.gguf`,
   `mmproj.gguf` e um `source.json` com os caminhos).

`test-soccer/render_qwen35_app.sh` reproduz essa instalação numa pasta de dados de teste e traduz os
capítulos 96 e 97 pelo caminho real do app.

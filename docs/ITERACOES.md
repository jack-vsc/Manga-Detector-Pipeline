# Diário de iterações

Cada etapa com o que se queria, o que foi feito e o que saiu, inclusive o que não deu certo. Datas de
novembro de 2026.

## 1. Página em prosa de light novel (3/11, abandonada no mesmo dia)

**Ideia inicial.** Uma página de mangá entra e sai como trecho de light novel: quem aparece, como está e
o que fala. Primeiro com o Manga109 (transcrições dos balões e personagens marcados); depois com páginas
em inglês e os blocos achados pelo detector do próprio Manga Index (`prepare_pages.py`). Um modelo maior,
o Gemma no Ollama, escrevia a prosa uma vez (`narrate_teacher.py`), e o modelo pequeno aprenderia a
copiá-lo (`pipeline.py`).

**Por que mudou.** O problema real era outro. O Manga Index já traduzia os balões, e o Gemma 4 E4B, o
melhor modelo local, ocupava a placa inteira. O objetivo passou a ser destilar a tradução do Claude num
modelo de ~2B. Os scripts de prosa saíram do repositório no commit que fez essa troca.

## 2. Dataset gravado pelo próprio app (3/11)

O Manga Index passou a gravar cada pedido que faz ao Claude e a resposta, no formato de
[ARQUITETURA.md](ARQUITETURA.md#o-pedido-de-tradução). Para não pagar de novo pelo que já tinha sido
traduzido, `replay_manga_index.ps1` refaz os capítulos da biblioteca a partir do cache de tradução, sem
chamar modelo nenhum.

Resultado: 171 capítulos, 5.239 páginas e 10.833 pedidos, a custo zero. Uma novel que tinha entrado por
engano foi tirada com `dataset_remove.py`. O painel (`dataset_panel.py`) acompanhava o replay ao vivo.

## 3. Primeiro treino: Qwen2-VL-2B em 4 bits (3–4/11)

**Memória.** A primeira tentativa transbordou para a "memória compartilhada" do Windows: o treino ficou 3x
mais lento e uma vez derrubou o PC. A partir daí:

- o PyTorch fica limitado à memória livre da placa (`cap_memory`), e um pedido que não cabe é pulado;
- `memory_probe.py` mediu o limite em ~3,9 mil tokens. Os 54 pedidos acima de 3,5 mil recebem páginas
  menores; 2 ficaram de fora;
- os logits são calculados só nas posições da resposta, o que poupa ~2,5 GB num pedido longo;
- o kernel "math" de atenção foi desligado, o que poupa 1,2 GB num pedido de 3,5 mil tokens.

**Calor.** Com a placa a 100% por horas, entrou uma pausa curta depois de cada pedido, ajustável com o
treino rodando, e o botão de pausa no painel, que suspende os processos. O painel também mostra GPU, CPU,
RAM e a memória compartilhada.

**Resultado.** Uma época (1.282 passos, 9,3 h), loss de validação 0,155.

## 4. Comparação com o Claude e o Gemma (4/11)

O Claude traduziu pelo app os capítulos 96 e 97 de *I'm Not a Soccer Genius!*. Os 111 pedidos foram
para cada modelo (`compare_models.py`). Para comparar em igualdade com o Gemma, que roda em GGUF no
Ollama, o adaptador foi fundido (`merge_adapter.py`) e convertido para GGUF.

| modelo | perdidas | quase iguais | bem diferentes | s/pedido |
|---|---|---|---|---|
| Gemma 4 E4B | 50 | 145 | 73 | 34,2 |
| Qwen2-VL treinado Q8_0 | 69 | 191 | 13 | 6,4 |

O Qwen treinado acerta muito mais o balão e o sentido, mas perde mais falas.

## 5. Capítulos inteiros dentro do app (4–5/11)

Rodar os capítulos pelo app, como um leitor, revelou três problemas de método, todos corrigidos antes dos
números finais:

1. **O servidor de desenvolvimento recarregava o app** a cada arquivo salvo no Manga Index e interrompia
   os testes. Os testes passaram a usar um build congelado (`electron-vite build --outDir`).
2. **Respostas em cache** de tentativas interrompidas deixavam os tempos curtos demais. Cada execução
   passou a ter uma pasta de dados vazia.
3. **O llama-server dividia 16k de contexto entre 4 pedidos paralelos**, e as páginas longas não cabiam.
   Passou a atender um pedido por vez (`-np 1`), como o Ollama faz com o Gemma.

Resultado: Gemma 4 E4B perdeu 33 de 330 falas; E2B, 110; Qwen2-VL treinado em Q4_0, 54, com 28
respostas presas em repetição até o limite de tokens.

## 6. Segundo dataset e a recusa do Claude (5/11)

**Ideia.** O Claude às vezes junta caixas de balões diferentes ou separa um balão em dois. Várias obras da
biblioteca têm o mesmo capítulo numa edição em português feita por fãs. `pt_align.py` acha a página
correspondente pela imagem (a arte é igual, só o letreiramento muda), e o plano era mostrar esse trecho ao
Claude como referência de agrupamento.

**O que aconteceu.** O Claude recusou usar a tradução de terceiros, por questão de direitos autorais,
mesmo com a referência num turno anterior e com o pedido restrito ao agrupamento. A recusa foi
respeitada: o trecho em português não é enviado, só localizado e registrado. Os dados das tentativas
foram apagados.

**Resultado.** 174 capítulos de 9 obras traduzidos pela API do Claude Haiku, 12.473 pedidos e 22.587
falas, por US$ 7,60. `finish_pt_dataset.py` tirou as notas que o Claude escrevia antes do JSON em 1.121
respostas. Capítulos em português que baixaram quebrados (uma imagem pequena no lugar da tira) ficaram de
fora.

## 7. Modelos originais lado a lado (5–6/11)

Para separar o ganho do treino do ganho de uma geração mais nova, os Qwen sem treino rodaram nos mesmos
111 pedidos (`run_base_bench.sh`):

- o **Qwen2-VL-2B original** perde 204 falas, o que mostra quanto o treino ajudou;
- o **Qwen2.5-VL-3B** gera "GGGG..." com o codificador de imagem na GPU e só funciona com a visão na CPU
  (57 s por pedido);
- o **Qwen3-VL-4B** em Q8_0 transborda a placa (4 minutos por pedido) e em Q4_K_M perde só 27;
- o **Qwen3.5** (visão embutida, atenção linear Gated DeltaNet em 18 de 24 camadas) é o melhor por
  tamanho. O 2B original perde 30 falas em 8 s por pedido, e o 4B, 24.

## 8. Treino do Qwen3.5-2B (6/11)

O Qwen3.5 só existe no transformers 5, que pedia um ambiente à parte (`.venv-q35`, com
flash-linear-attention e Triton). Ajustes no treino:

- tokens de imagem de 32 px, em vez de 28: `model.UNIT` vem do próprio processor;
- as camadas de atenção linear também recebem LoRA;
- o kernel "math" de atenção precisa ficar ligado, porque os outros recusam os formatos das 6 camadas de
  atenção completa;
- a resposta é treinada depois do `<think></think>` vazio que o template insere sem raciocínio;
- pedidos de até 2.600 tokens, porque acima disso estoura a memória (134 pedidos pulados);
- os workers do DataLoader no Windows reimportam os módulos, então a pasta do dataset e o `UNIT` ficam
  guardados no próprio `Requests`.

**Resultado.** 1.519 passos (9,5 h), loss de validação 0,157. Nos 111 pedidos: 1 resposta quebrada, 45
falas perdidas, 232 quase iguais e 6 bem diferentes, em 6 s por pedido.

## 9. Menos repescagens vazias (7/11, descartado)

**Hipótese.** 2.122 das 3.838 repescagens do dataset voltaram vazias, o que ensinaria o modelo a deixar
caixas de fora. O treino v2 manteve só 20% delas (`--empty-followups 0.2`).

**Resultado.** Pior: 56 falas perdidas contra 45, e loss de validação 0,158 contra 0,157. A ideia foi
descartada. O painel ganhou a curva do treino anterior por baixo da atual (`--compare`), para comparar no
mesmo ponto da época. O log do transformers 5 imprime as losses como texto, o que deixava o gráfico
vazio; isso também foi corrigido.

## 10. De onde vêm as falas perdidas (7/11)

- As perdas são caixas que o modelo lista em `skip` nos pedidos de página (27 no v1, 38 no v2, 2 no
  original), não nas repescagens.
- ~12 falas vêm de uma única resposta presa em repetição (capítulo 96, página 10: uma lista de `skip`
  contando até 780).
- 6 a 8 são da página de anúncio com QR code no fim do capítulo.
- O resto são páginas longas, de 7 a 8 imagens.
- O tamanho das imagens não é a causa: original, ajustada ao treino ou maior deram o mesmo resultado.
- O próprio Claude pula 39% das caixas no dataset, contra 33% nos capítulos de teste.
- Em 36 casos o modelo põe em `skip` uma caixa que ele mesmo traduziu.

Dentro do app, a repescagem pergunta de novo pelas caixas que sobram, e o Qwen3.5-2B treinado perde 38
falas, contra 33 do Gemma 4 E4B.

## 11. Integração ao Manga Index (8/11)

O modelo entrou na lista de modelos locais do app (`localAi.ts`). No primeiro teste pelo perfil "Neste
PC", um pedido de 9.300 tokens estourou o contexto de 8.192 que o app dava ao Gemma. Agora cada modelo
tem o próprio contexto, e o Qwen3.5 usa 16.384.

**Resultado final:** 31 falas perdidas (Gemma 4 E4B: 33), 234 quase iguais (202), 5 bem diferentes
(29), 17,9 min (25,0), 14,0 Wh (20,7 Wh) e 4,9 GB de pico de VRAM (5,6 GB).

## Próximos passos possíveis

- Treinar o Qwen3.5-4B, que sem treino já é o melhor modelo local nos 111 pedidos.
- Limpar do dataset as contradições de `skip` (caixas puladas e traduzidas na mesma resposta).
- Baixar de novo os capítulos em português quebrados e ampliar o dataset.
- Publicar os pesos, se os termos dos dados permitirem, para o app poder baixar o modelo.

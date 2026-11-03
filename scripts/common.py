"""Constantes compartilhadas: caminhos, modelo base e orçamentos de imagem."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# Tamanhos de imagem em pixels. No Qwen2-VL cada bloco de 28x28 vira um token.
OCR_PIXELS = (28 * 28 * 16, 28 * 28 * 192)
PAGE_PIXELS = (28 * 28 * 256, 28 * 28 * 640)

# Pedidos de tradução do Manga Index (scripts/train.py). A imagem-chave (as caixas recortadas) fica com até
# KEY_TOKENS; a página e suas fatias dividem PAGE_TOKENS. Um pedido acima de MAX_TOKENS recebe páginas
# menores (o primeiro de PAGE_TOKENS_SHRUNK que couber) e só fica de fora se nenhum couber. Numa placa de
# 6 GB o treino estoura perto de 3,9k tokens (scripts/memory_probe.py).
KEY_TOKENS = 640
PAGE_TOKENS = 1024
PAGE_TOKENS_SHRUNK = (768, 512, 256)
MAX_TOKENS = 3500

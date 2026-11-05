"""Acha, na edição em português de um capítulo, o trecho que corresponde a uma página da edição em inglês.

A arte é a mesma nas duas edições e só o letreiramento muda. Cada edição é montada como uma tira contínua
numa largura comum, e a página em inglês é procurada na tira em português pela imagem.
"""
import cv2
import numpy as np
import pymupdf

WIDTH = 720      # largura comum das duas tiras
SMALL = 160      # largura usada na busca


def render_pages(pdf):
    """Páginas do PDF como arrays RGB, cada uma com WIDTH de largura."""
    pages = []
    for page in pymupdf.open(pdf):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(WIDTH / page.rect.width, WIDTH / page.rect.width), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        if img.shape[1] != WIDTH:
            img = cv2.resize(img, (WIDTH, round(img.shape[0] * WIDTH / img.shape[1])))
        pages.append(img)
    return pages


def small(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (SMALL, max(1, round(g.shape[0] * SMALL / g.shape[1]))), interpolation=cv2.INTER_AREA)


def unmark(rgb):
    """Apaga as caixas e etiquetas magenta que o app desenha na página, para não atrapalharem a busca."""
    r, g, b = (rgb[:, :, i].astype(int) for i in range(3))
    mask = ((r > 180) & (b > 180) & (g < 110)).astype(np.uint8)
    if mask.any():
        rgb = cv2.inpaint(rgb, cv2.dilate(mask, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    return rgb


class Strip:
    """Um capítulo montado como tira contínua."""

    def __init__(self, pdf):
        self.full = np.vstack(render_pages(pdf))
        self.small = small(self.full)
        self.scale = WIDTH / SMALL

    def find(self, page_rgb):
        """(topo, base, nota) do trecho da tira que corresponde à página, em pixels da tira cheia.

        As edições podem cortar a arte um pouco diferente nas laterais, então a página também é testada um
        pouco menor e um pouco maior (só o miolo).
        """
        q = small(unmark(page_rgb))
        best = (0, self.full.shape[0], 0.0)
        for f in (1.0, 0.94, 0.88, 1.06, 1.12):
            w = round(SMALL * f)
            qs = cv2.resize(q, (w, max(8, round(q.shape[0] * f))), interpolation=cv2.INTER_AREA)
            if w > SMALL:  # mais larga que a tira: fica o miolo
                x = (w - SMALL) // 2
                qs = qs[:, x:x + SMALL]
            if qs.shape[0] >= self.small.shape[0]:
                continue
            res = cv2.matchTemplate(self.small, qs, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score > best[2]:
                top = round(y * self.scale)
                best = (top, top + round(qs.shape[0] * self.scale), float(score))
        return best

    def find_slices(self, slices_rgb):
        """(topo, base, nota) cobrindo as fatias consecutivas de uma página longa.

        Cada fatia é procurada sozinha (elas se sobrepõem e podem ter tamanhos diferentes) e as que caem
        longe da melhor correspondência são descartadas.
        """
        found = sorted((self.find(s) for s in slices_rgb), key=lambda f: -f[2])
        anchor = found[0]
        reach = sum(s.shape[0] * WIDTH / s.shape[1] for s in slices_rgb)
        near = [f for f in found if abs(f[0] - anchor[0]) <= reach * 1.2 and f[2] >= anchor[2] - 0.25]
        return min(f[0] for f in near), max(f[1] for f in near), anchor[2]

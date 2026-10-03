"""Desenho dos achados sobre a radiografia (mapas de calor, contornos, rótulos e ICT)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .patologias import cor_bgr
from .resultado import MOTOR_NUVEM, POSITIVO, Achado, ResultadoAnalise

COR_ICT_CORACAO = "#EC407A"
COR_ICT_TORAX = "#4DD0E1"

_FONTES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)


@lru_cache(maxsize=16)
def fonte(tamanho: int) -> ImageFont.ImageFont:
    for caminho in _FONTES:
        if Path(caminho).exists():
            return ImageFont.truetype(caminho, tamanho)
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


@dataclass
class OpcoesVisualizacao:
    mapas_calor: bool = True
    contornos: bool = True
    rotulos: bool = True
    ict: bool = True
    anatomia: bool = False
    indeterminados: bool = True
    legenda: bool = True
    opacidade: float = 0.45


def achados_exibidos(resultado: ResultadoAnalise, incluir_indeterminados: bool = True) -> list[Achado]:
    """Achados desenhados na imagem, em ordem de escore decrescente."""
    status = ("positivo", "indeterminado") if incluir_indeterminados else ("positivo",)
    return resultado.com_status(*status)


def _hex_rgb(cor: str) -> tuple[int, int, int]:
    b, g, r = cor_bgr(cor)
    return (r, g, b)


def _linha_tracejada(img: np.ndarray, p0, p1, cor, espessura: int, traco: int = 10) -> None:
    p0, p1 = np.array(p0, float), np.array(p1, float)
    comprimento = float(np.linalg.norm(p1 - p0))
    if comprimento < 1:
        return
    direcao = (p1 - p0) / comprimento
    for inicio in np.arange(0, comprimento, 2 * traco):
        a = p0 + direcao * inicio
        b = p0 + direcao * min(inicio + traco, comprimento)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), cor, espessura, cv2.LINE_AA)


def _poligono(img: np.ndarray, pontos, cor, espessura: int, tracejado: bool) -> None:
    if len(pontos) < 2:
        return
    if not tracejado:
        cv2.polylines(img, [np.array(pontos, np.int32)], True, cor, espessura, cv2.LINE_AA)
        return
    for i in range(len(pontos)):
        _linha_tracejada(img, pontos[i], pontos[(i + 1) % len(pontos)], cor, espessura)


def _sobrepor_mapa(img: np.ndarray, mapa: np.ndarray, cor_hex: str, opacidade: float) -> None:
    """Mistura uma camada colorida com transparência proporcional ao mapa."""
    m = mapa.astype(np.float32) / 255.0
    alfa = np.clip((m - 0.25) / 0.75, 0.0, 1.0) ** 1.2 * opacidade
    cor = np.array(cor_bgr(cor_hex), dtype=np.float32)
    img[:] = (img.astype(np.float32) * (1 - alfa[..., None]) + cor * alfa[..., None]).astype(np.uint8)


def desenhar_achados(resultado: ResultadoAnalise, opcoes: OpcoesVisualizacao | None = None) -> np.ndarray:
    """Retorna a imagem anotada em RGB (uint8), com legenda lateral opcional."""
    op = opcoes or OpcoesVisualizacao()
    base = cv2.cvtColor(resultado.imagem, cv2.COLOR_GRAY2BGR)
    altura, largura = resultado.imagem.shape
    espessura = max(2, round(max(altura, largura) / 400))
    achados = achados_exibidos(resultado, op.indeterminados)

    if op.anatomia:
        camada = base.copy()
        for contornos in resultado.contornos_anatomicos.values():
            for c in contornos:
                _poligono(camada, c, (255, 255, 255), 1, False)
        base = cv2.addWeighted(camada, 0.5, base, 0.5, 0)

    if op.mapas_calor:
        for achado in reversed(achados):
            if achado.mapa is not None:
                _sobrepor_mapa(base, achado.mapa, achado.cor, op.opacidade)

    if op.contornos:
        for achado in reversed(achados):
            for regiao in achado.regioes:
                _poligono(base, regiao.contorno, cor_bgr(achado.cor), espessura,
                          tracejado=achado.status != POSITIVO)

    if op.ict and resultado.ict:
        for (p0, p1), cor in ((resultado.ict.linha_torax, COR_ICT_TORAX),
                              (resultado.ict.linha_coracao, COR_ICT_CORACAO)):
            bgr = cor_bgr(cor)
            cv2.line(base, p0, p1, bgr, espessura, cv2.LINE_AA)
            for p in (p0, p1):
                cv2.line(base, (p[0], p[1] - 4 * espessura), (p[0], p[1] + 4 * espessura), bgr,
                         espessura, cv2.LINE_AA)

    imagem = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    if op.rotulos:
        _desenhar_rotulos(imagem, resultado, achados)
    if op.legenda:
        imagem = _adicionar_legenda(imagem, resultado, achados)
    return np.asarray(imagem)


def _desenhar_rotulos(imagem: Image.Image, resultado: ResultadoAnalise, achados: list[Achado]) -> None:
    draw = ImageDraw.Draw(imagem)
    tamanho = max(12, round(max(imagem.size) / 55))
    f = fonte(tamanho)
    ocupados: list[tuple[int, int, int, int]] = []
    for numero, achado in enumerate(achados, start=1):
        for regiao in achado.regioes[:1]:
            texto = f"{numero}  {achado.nome} {achado.rotulo_confianca}".strip()
            x0, y0, x1, y1 = draw.textbbox((0, 0), texto, font=f)
            w, h = x1 - x0 + 10, y1 - y0 + 8
            x, y, _, _ = regiao.caixa
            y = y - h - 2 if y - h - 2 >= 0 else y + 2
            x = min(max(0, x), imagem.width - w)
            # Evita sobreposição entre rótulos deslocando-os para baixo.
            while any(x < o[2] and x + w > o[0] and y < o[3] and y + h > o[1] for o in ocupados):
                y += h + 2
            y = min(y, imagem.height - h)
            ocupados.append((x, y, x + w, y + h))
            rgb = _hex_rgb(achado.cor)
            draw.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=rgb)
            luminancia = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            draw.text((x + 5, y + 4 - y0), texto, font=f, fill=(0, 0, 0) if luminancia > 140 else (255, 255, 255))
    if resultado.ict:
        (x0, y), (x1, _) = resultado.ict.linha_coracao
        texto = f"ICT {resultado.ict.indice:.2f}".replace(".", ",")
        bx = draw.textbbox((0, 0), texto, font=f)
        draw.text(((x0 + x1) // 2 - (bx[2] - bx[0]) // 2, y + 6), texto, font=f,
                  fill=_hex_rgb(COR_ICT_CORACAO), stroke_width=2, stroke_fill=(0, 0, 0))


def _adicionar_legenda(imagem: Image.Image, resultado: ResultadoAnalise, achados: list[Achado]) -> Image.Image:
    largura_painel = max(280, imagem.width // 3)
    painel = Image.new("RGB", (largura_painel, imagem.height), (20, 24, 31))
    draw = ImageDraw.Draw(painel)
    escala = max(imagem.height / 1024, 0.75)
    f_titulo, f_texto, f_menor = fonte(round(26 * escala)), fonte(round(19 * escala)), fonte(round(15 * escala))
    margem = round(16 * escala)
    y = margem

    def escrever(texto: str, f, cor=(230, 233, 238), recuo: int = 0) -> None:
        nonlocal y
        for linha in _quebrar(draw, texto, f, largura_painel - 2 * margem - recuo):
            draw.text((margem + recuo, y), linha, font=f, fill=cor)
            y += f.size + round(5 * escala)

    nuvem = resultado.motor == MOTOR_NUVEM
    escrever("Achados (IA)", f_titulo)
    escrever(resultado.regiao_nome, f_menor, (170, 176, 186))
    y += round(6 * escala)
    if not achados:
        escrever("Nenhum achado acima dos limiares de detecção.", f_texto, (160, 200, 160))
    for numero, achado in enumerate(achados, start=1):
        quadrado = round(16 * escala)
        draw.rectangle((margem, y + 2, margem + quadrado, y + 2 + quadrado), fill=_hex_rgb(achado.cor))
        estado = "" if achado.status == POSITIVO or achado.escore is None else " (indeterminado)"
        escrever(f"{numero}. {achado.nome} — {achado.rotulo_confianca}{estado}", f_texto, recuo=quadrado + 8)
        if achado.local:
            escrever(achado.local[0].upper() + achado.local[1:], f_menor, (170, 176, 186), recuo=quadrado + 8)
        y += round(4 * escala)
    if resultado.ict:
        y += round(8 * escala)
        escrever(f"Índice cardiotorácico: {resultado.ict.indice:.2f}".replace(".", ","), f_texto,
                 _hex_rgb(COR_ICT_CORACAO))
    rodape = "Apoio à decisão — requer revisão por médico(a) radiologista."
    if nuvem:
        rodape = "IA em nuvem: localização aproximada. " + rodape
    linhas = _quebrar(draw, rodape, f_menor, largura_painel - 2 * margem)
    y_rodape = imagem.height - margem - len(linhas) * (f_menor.size + 4)
    for linha in linhas:
        draw.text((margem, y_rodape), linha, font=f_menor, fill=(150, 150, 150))
        y_rodape += f_menor.size + 4

    combinada = Image.new("RGB", (imagem.width + largura_painel, imagem.height))
    combinada.paste(imagem, (0, 0))
    combinada.paste(painel, (imagem.width, 0))
    return combinada


def _quebrar(draw: ImageDraw.ImageDraw, texto: str, f, largura: int) -> list[str]:
    linhas, atual = [], ""
    for palavra in texto.split():
        candidato = f"{atual} {palavra}".strip()
        if draw.textlength(candidato, font=f) <= largura or not atual:
            atual = candidato
        else:
            linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas


# --------------------------------------------------------------------------- #
# Codificação
# --------------------------------------------------------------------------- #

def png_bytes(imagem: np.ndarray) -> bytes:
    """Codifica uma imagem (cinza, RGB ou RGBA) em PNG."""
    buffer = io.BytesIO()
    Image.fromarray(imagem).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def mapa_rgba(mapa: np.ndarray, cor_hex: str, reducao: int = 4) -> np.ndarray:
    """Mapa de calor colorido com transparência, reduzido para envio ao navegador."""
    if reducao > 1:
        mapa = cv2.resize(mapa, (max(1, mapa.shape[1] // reducao), max(1, mapa.shape[0] // reducao)),
                          interpolation=cv2.INTER_AREA)
    m = mapa.astype(np.float32) / 255.0
    alfa = (np.clip((m - 0.25) / 0.75, 0.0, 1.0) ** 1.2 * 255).astype(np.uint8)
    rgba = np.zeros(mapa.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = _hex_rgb(cor_hex)
    rgba[..., 3] = alfa
    return rgba

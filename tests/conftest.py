"""Fixtures: DICOMs sintéticos e modelos falsos (testes rápidos, sem pesos reais)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from interpretador_radiologico.analisador import AnalisadorTorax
from interpretador_radiologico.config import Configuracao
from interpretador_radiologico.exemplo import imagem_para_dicom
from interpretador_radiologico.patologias import CATALOGO

# Geometria do "tórax" sintético, em coordenadas relativas (x, y).
PULMAO_IMAGEM_ESQUERDA = ((0.30, 0.45), (0.15, 0.28))  # pulmão DIREITO do paciente
PULMAO_IMAGEM_DIREITA = ((0.70, 0.45), (0.15, 0.28))   # pulmão ESQUERDO do paciente
CORACAO = ((0.56, 0.62), (0.14, 0.10))


def _elipse(tamanho: int, centro, eixos) -> np.ndarray:
    m = np.zeros((tamanho, tamanho), dtype=np.uint8)
    cv2.ellipse(m, (int(centro[0] * tamanho), int(centro[1] * tamanho)),
                (int(eixos[0] * tamanho), int(eixos[1] * tamanho)), 0, 0, 360, 1, -1)
    return m.astype(np.float32)


def torax_sintetico(tamanho: int = 400) -> np.ndarray:
    """Imagem uint8 grosseiramente parecida com um tórax (pulmões escuros)."""
    img = np.full((tamanho, tamanho), 170, dtype=np.float32)
    img -= 110 * _elipse(tamanho, *PULMAO_IMAGEM_ESQUERDA)
    img -= 110 * _elipse(tamanho, *PULMAO_IMAGEM_DIREITA)
    img += 50 * _elipse(tamanho, *CORACAO)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    return np.clip(img, 0, 255).astype(np.uint8)


class ModelosFalsos:
    """Substitui os modelos do TorchXRayVision com saídas controladas.

    ``focos`` mapeia patologia -> (x, y) relativos onde o mapa de ativação tem pico.
    """

    def __init__(self, escores: dict[str, float] | None = None,
                 focos: dict[str, tuple[float, float]] | None = None,
                 segmentacao: bool = True, coracao=CORACAO):
        self.escores = escores or {}
        self.focos = focos or {}
        self.segmentacao = segmentacao
        self.coracao = coracao
        self.chamadas = 0

    def classificar(self, entrada, refinar):
        assert entrada.shape == (224, 224)
        self.chamadas += 1
        nomes = list(CATALOGO)
        escores = np.array([self.escores.get(n, 0.2) for n in nomes], dtype=np.float32)
        mapas = np.zeros((len(nomes), 224, 224), dtype=np.float32)
        yy, xx = np.mgrid[0:224, 0:224]
        for i, nome in enumerate(nomes):
            if nome in self.focos:
                fx, fy = self.focos[nome]
                mapas[i] = np.exp(-((xx - fx * 224) ** 2 + (yy - fy * 224) ** 2) / (2 * 14.0 ** 2))
        return nomes, escores, mapas

    def segmentar(self, entrada):
        if not self.segmentacao:
            raise RuntimeError("pesos indisponíveis")
        assert entrada.shape == (512, 512)
        coluna = np.zeros((512, 512), dtype=np.float32)
        coluna[:, 246:266] = 1
        return {
            "pulmao_direito": _elipse(512, *PULMAO_IMAGEM_ESQUERDA),
            "pulmao_esquerdo": _elipse(512, *PULMAO_IMAGEM_DIREITA),
            "coracao": _elipse(512, *self.coracao),
            "coluna": coluna,
        }

    def descricao(self):
        return {"classificador": "modelo falso", "segmentacao": "segmentação falsa"}


@pytest.fixture
def caminho_dicom(tmp_path):
    """Fábrica de arquivos DICOM sintéticos."""
    def fabricar(nome: str = "torax.dcm", **kwargs):
        ds = imagem_para_dicom(torax_sintetico(), **kwargs)
        caminho = tmp_path / nome
        ds.save_as(caminho, enforce_file_format=True)
        return caminho
    return fabricar


@pytest.fixture
def analisador_falso():
    """Fábrica de analisadores com modelos falsos."""
    def fabricar(escores=None, focos=None, segmentacao=True, **config):
        return AnalisadorTorax(Configuracao(**config), ModelosFalsos(escores, focos, segmentacao))
    return fabricar

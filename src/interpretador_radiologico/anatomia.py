"""Referências anatômicas: máscaras, lateralidade, zonas pulmonares e ICT."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .localizacao import DIREITO, ESQUERDO, ZONAS
from .patologias import REGIAO_CORACAO, REGIAO_DIAFRAGMA, REGIAO_MEDIASTINO, REGIAO_PULMAO

# Estruturas do modelo ChestX-Det (PSPNet) mapeadas para nomes internos.
# Os rótulos "Left"/"Right" seguem a lateralidade do paciente numa imagem na
# orientação radiológica padrão (lado direito do paciente à esquerda da tela).
ESTRUTURAS_PSPNET = {
    "Right Lung": "pulmao_direito",
    "Left Lung": "pulmao_esquerdo",
    "Heart": "coracao",
    "Mediastinum": "mediastino",
    "Aorta": "aorta",
    "Weasand": "traqueia",
    "Facies Diaphragmatica": "diafragma",
    "Spine": "coluna",
    "Left Clavicle": "clavicula_esquerda",
    "Right Clavicle": "clavicula_direita",
}


@dataclass
class MedidaICT:
    """Índice cardiotorácico estimado a partir da segmentação."""

    indice: float
    largura_coracao_px: int
    largura_torax_px: int
    linha_coracao: tuple[tuple[int, int], tuple[int, int]]
    linha_torax: tuple[tuple[int, int], tuple[int, int]]
    largura_coracao_cm: float | None = None
    largura_torax_cm: float | None = None

    def para_dict(self) -> dict:
        return {
            "indice": round(self.indice, 3),
            "largura_coracao_px": self.largura_coracao_px,
            "largura_torax_px": self.largura_torax_px,
            "largura_coracao_cm": None if self.largura_coracao_cm is None else round(self.largura_coracao_cm, 1),
            "largura_torax_cm": None if self.largura_torax_cm is None else round(self.largura_torax_cm, 1),
            "linha_coracao": [list(p) for p in self.linha_coracao],
            "linha_torax": [list(p) for p in self.linha_torax],
        }


def _dilatar(mascara: np.ndarray, raio: int) -> np.ndarray:
    if raio <= 0:
        return mascara.astype(bool)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))
    return cv2.dilate(mascara.astype(np.uint8), nucleo).astype(bool)


def _suavizar(mascara: np.ndarray, sigma: float) -> np.ndarray:
    m = mascara.astype(np.float32)
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


def _maior_componente(mascara: np.ndarray) -> np.ndarray:
    mascara = mascara.astype(np.uint8)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return mascara.astype(bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return rotulos == i


def _envoltoria(mascara: np.ndarray) -> np.ndarray:
    pontos = cv2.findNonZero(mascara.astype(np.uint8))
    saida = np.zeros(mascara.shape, dtype=np.uint8)
    if pontos is not None and len(pontos) >= 3:
        cv2.fillConvexPoly(saida, cv2.convexHull(pontos), 1)
    return saida.astype(bool)


def _elipse(forma: tuple[int, int], centro: tuple[float, float], eixos: tuple[float, float]) -> np.ndarray:
    altura, largura = forma
    m = np.zeros(forma, dtype=np.uint8)
    cv2.ellipse(m, (int(centro[0] * largura), int(centro[1] * altura)),
                (int(eixos[0] * largura), int(eixos[1] * altura)), 0, 0, 360, 1, -1)
    return m.astype(bool)


class Anatomia:
    """Máscaras anatômicas na resolução de exibição.

    Quando a segmentação não está disponível, usa aproximações geométricas
    (``segmentada = False``).

    ``invertida`` indica imagem espelhada em relação à orientação radiológica
    padrão (PatientOrientation iniciando por "R").
    """

    def __init__(self, forma: tuple[int, int], mascaras: dict[str, np.ndarray] | None = None,
                 invertida: bool = False):
        self.forma = forma
        self.mascaras = {k: v.astype(bool) for k, v in (mascaras or {}).items()}
        self.invertida = invertida
        self.segmentada = bool(self.mascaras)
        if self.segmentada:
            self.mascaras["coracao"] = _maior_componente(self.mascara("coracao"))
        self._priors: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def mascara(self, nome: str) -> np.ndarray:
        return self.mascaras.get(nome, np.zeros(self.forma, dtype=bool))

    @property
    def pulmoes(self) -> np.ndarray:
        return self.mascara("pulmao_direito") | self.mascara("pulmao_esquerdo")

    @property
    def _largura(self) -> int:
        return self.forma[1]

    def linha_media(self) -> float:
        """Coordenada x da linha média (coluna vertebral, se segmentada)."""
        for nome in ("coluna", "traqueia", "mediastino"):
            m = self.mascara(nome)
            if m.sum() > 0.002 * m.size:
                return float(np.nonzero(m)[1].mean())
        d, e = self.mascara("pulmao_direito"), self.mascara("pulmao_esquerdo")
        if d.any() and e.any():
            return (np.nonzero(d)[1].mean() + np.nonzero(e)[1].mean()) / 2.0
        return self._largura / 2.0

    # ------------------------------------------------------------------ #
    def prior(self, regiao: str) -> np.ndarray:
        """Máscara suave [0, 1] da região anatômica plausível para o achado."""
        if regiao not in self._priors:
            self._priors[regiao] = self._calcular_prior(regiao)
        return self._priors[regiao]

    def _calcular_prior(self, regiao: str) -> np.ndarray:
        w = self._largura
        sigma = 0.02 * w
        if not self.segmentada or self.pulmoes.sum() < 0.03 * self.pulmoes.size:
            if regiao in (REGIAO_CORACAO, REGIAO_MEDIASTINO):
                base = _elipse(self.forma, (0.52, 0.55), (0.25, 0.30))
            else:
                base = _elipse(self.forma, (0.5, 0.5), (0.47, 0.46))
            return _suavizar(base, 0.04 * w)

        pulmoes = self.pulmoes
        coracao = self.mascara("coracao")
        if regiao == REGIAO_PULMAO:
            base = _dilatar(pulmoes, int(0.025 * w))
        elif regiao == REGIAO_CORACAO:
            base = _dilatar(coracao, int(0.03 * w))
        elif regiao == REGIAO_MEDIASTINO:
            base = _dilatar(coracao | self.mascara("mediastino") | self.mascara("aorta")
                            | self.mascara("traqueia"), int(0.04 * w))
        elif regiao == REGIAO_DIAFRAGMA:
            base = _dilatar(_envoltoria(pulmoes | coracao), int(0.06 * w))
            ys = np.nonzero(pulmoes)[0]
            base[: int(np.percentile(ys, 40))] = False  # metade inferior do tórax
        else:  # tórax (padrão)
            base = _dilatar(_envoltoria(pulmoes | coracao), int(0.08 * w))
        return _suavizar(base, sigma)

    # ------------------------------------------------------------------ #
    def lado_e_zonas(self, mascara: np.ndarray) -> tuple[str | None, list[str]]:
        """Lado do paciente e terços pulmonares ocupados por uma região."""
        if not mascara.any():
            return None, []
        ys, xs = np.nonzero(mascara)
        lado_imagem = None  # "esquerda"/"direita" da imagem

        if self.segmentada:
            raio = int(0.03 * self._largura)
            d = _dilatar(self.mascara("pulmao_direito"), raio)  # à esquerda na imagem padrão
            e = _dilatar(self.mascara("pulmao_esquerdo"), raio)
            n_d, n_e = int(d[ys, xs].sum()), int(e[ys, xs].sum())
            if max(n_d, n_e) > 0.15 * len(xs):
                lado_imagem = "esquerda" if n_d >= n_e else "direita"
        if lado_imagem is None:
            lado_imagem = "esquerda" if xs.mean() < self.linha_media() else "direita"

        lado_paciente = DIREITO if lado_imagem == "esquerda" else ESQUERDO
        if self.invertida:
            lado_paciente = ESQUERDO if lado_paciente == DIREITO else DIREITO

        topo, base = self._extensao_vertical(lado_imagem)
        altura = max(base - topo, 1)
        limites = (topo + altura / 3.0, topo + 2 * altura / 3.0)
        contagens = [
            int((ys < limites[0]).sum()),
            int(((ys >= limites[0]) & (ys < limites[1])).sum()),
            int((ys >= limites[1]).sum()),
        ]
        total = max(sum(contagens), 1)
        zonas = [ZONAS[i] for i, c in enumerate(contagens) if c / total >= 0.25]
        if not zonas:
            zonas = [ZONAS[int(np.argmax(contagens))]]
        return lado_paciente, zonas

    def _extensao_vertical(self, lado_imagem: str) -> tuple[float, float]:
        if self.segmentada:
            nome = "pulmao_direito" if lado_imagem == "esquerda" else "pulmao_esquerdo"
            m = self.mascara(nome)
            if m.sum() < 0.01 * m.size:
                m = self.pulmoes
            if m.any():
                ys = np.nonzero(m)[0]
                return float(ys.min()), float(ys.max())
        altura = self.forma[0]
        return 0.10 * altura, 0.85 * altura

    # ------------------------------------------------------------------ #
    def indice_cardiotoracico(self, espacamento_mm: tuple[float, float] | None = None,
                              escala: float = 1.0) -> MedidaICT | None:
        """ICT = maior diâmetro transverso cardíaco / maior diâmetro torácico interno."""
        if not self.segmentada:
            return None
        coracao, pulmoes = self.mascara("coracao"), self.pulmoes
        if coracao.sum() < 0.01 * coracao.size or pulmoes.sum() < 0.05 * pulmoes.size:
            return None

        ys_c, xs_c = np.nonzero(coracao)
        x0_c, x1_c = int(xs_c.min()), int(xs_c.max())
        y_c = int(np.median(ys_c))
        largura_c = x1_c - x0_c + 1

        linhas = np.nonzero(pulmoes.any(axis=1))[0]
        melhor = (0, 0, 0, 0)  # largura, y, x0, x1
        for y in linhas:
            xs = np.nonzero(pulmoes[y])[0]
            largura = int(xs.max() - xs.min() + 1)
            if largura > melhor[0]:
                melhor = (largura, int(y), int(xs.min()), int(xs.max()))
        largura_t, y_t, x0_t, x1_t = melhor
        if largura_t <= 0:
            return None
        indice = largura_c / largura_t
        if not 0.2 <= indice <= 0.9:
            return None

        cm_c = cm_t = None
        if espacamento_mm:
            mm_por_px = espacamento_mm[1] * escala
            cm_c, cm_t = largura_c * mm_por_px / 10.0, largura_t * mm_por_px / 10.0
        return MedidaICT(
            indice=float(indice),
            largura_coracao_px=largura_c,
            largura_torax_px=largura_t,
            linha_coracao=((x0_c, y_c), (x1_c, y_c)),
            linha_torax=((x0_t, y_t), (x1_t, y_t)),
            largura_coracao_cm=cm_c,
            largura_torax_cm=cm_t,
        )

    def parece_torax(self) -> bool:
        """Dois campos pulmonares de tamanho plausível foram segmentados."""
        if not self.segmentada:
            return False
        direito = float(self.mascara("pulmao_direito").mean())
        esquerdo = float(self.mascara("pulmao_esquerdo").mean())
        return min(direito, esquerdo) >= 0.04 and direito + esquerdo >= 0.12

    # ------------------------------------------------------------------ #
    def avisos_qualidade(self) -> list[str]:
        """Verificações de plausibilidade da imagem com base na anatomia."""
        if not self.segmentada:
            return []
        avisos = []
        fracao_pulmoes = float(self.pulmoes.mean())
        if fracao_pulmoes < 0.08:
            avisos.append(
                "Campos pulmonares pouco identificados na imagem: verifique se trata-se de "
                "radiografia de tórax em incidência frontal; os resultados podem não ser confiáveis."
            )
            return avisos
        coracao = self.mascara("coracao")
        if coracao.sum() > 0.01 * coracao.size:
            desvio = (np.nonzero(coracao)[1].mean() - self.linha_media()) / self._largura
            if (desvio < -0.03 and not self.invertida) or (desvio > 0.03 and self.invertida):
                avisos.append(
                    "Silhueta cardíaca predominantemente no hemitórax direito do paciente: "
                    "verificar a marcação de lateralidade/orientação da imagem (imagem espelhada?) "
                    "ou possibilidade de dextrocardia. A lateralidade descrita pode estar invertida."
                )
        return avisos

    def contornos(self) -> dict[str, list[list[tuple[int, int]]]]:
        """Contornos dos pulmões e do coração para exibição."""
        saida: dict[str, list[list[tuple[int, int]]]] = {}
        if not self.segmentada:
            return saida
        for nome in ("pulmao_direito", "pulmao_esquerdo", "coracao"):
            m = self.mascara(nome).astype(np.uint8)
            if not m.any():
                continue
            contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            grandes = [c for c in contornos if cv2.contourArea(c) > 0.002 * m.size]
            saida[nome] = [
                [(int(p[0][0]), int(p[0][1]))
                 for p in cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True)]
                for c in grandes
            ]
        return saida

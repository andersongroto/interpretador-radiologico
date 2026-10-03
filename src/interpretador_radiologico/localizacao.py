"""Conversão de mapas de ativação em regiões (contornos/caixas) e descrição anatômica."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .patologias import LOCAL_HEMITORAX, LOCAL_LADO, LOCAL_NENHUM

DIREITO = "direito"
ESQUERDO = "esquerdo"
ZONAS = ("superior", "médio", "inferior")


@dataclass
class Regiao:
    """Região suspeita, em coordenadas da imagem de exibição."""

    contorno: list[tuple[int, int]]
    caixa: tuple[int, int, int, int]  # x, y, largura, altura
    area: int
    pico: float
    centroide: tuple[float, float]
    lado: str | None = None  # lado do PACIENTE: "direito"/"esquerdo"
    zonas: list[str] = field(default_factory=list)

    def para_dict(self, escala: float = 1.0) -> dict:
        return {
            "contorno": [[int(x), int(y)] for x, y in self.contorno],
            "caixa": [int(v) for v in self.caixa],
            "caixa_original": [int(round(v * escala)) for v in self.caixa],
            "area_pixels": int(self.area),
            "pico": round(float(self.pico), 3),
            "centroide": [round(float(c), 1) for c in self.centroide],
            "lado": self.lado,
            "zonas": list(self.zonas),
        }


def normalizar_mapa(mapa: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """Remove valores negativos, aplica a máscara anatômica e normaliza para [0, 1]."""
    mapa = np.maximum(mapa.astype(np.float32), 0.0)
    if prior is not None:
        mapa = mapa * prior.astype(np.float32)
    maximo = float(mapa.max())
    if maximo <= 1e-8:
        return np.zeros_like(mapa, dtype=np.float32)
    return mapa / maximo


def extrair_regioes(
    mapa: np.ndarray,
    limiar_relativo: float = 0.5,
    area_minima_fracao: float = 0.003,
    max_regioes: int = 3,
    pico_minimo: float = 0.6,
) -> list[Regiao]:
    """Segmenta um mapa normalizado [0, 1] em até ``max_regioes`` regiões conexas."""
    if mapa.size == 0 or float(mapa.max()) <= 0:
        return []
    binario = (mapa >= limiar_relativo).astype(np.uint8)
    n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(binario, connectivity=8)
    area_minima = max(4, int(area_minima_fracao * mapa.size))
    candidatas = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_minima:
            continue
        mascara = rotulos == i
        pico = float(mapa[mascara].max())
        if pico < pico_minimo:
            continue
        candidatas.append((pico, i, area))
    candidatas.sort(reverse=True)

    regioes: list[Regiao] = []
    for pico, i, area in candidatas[:max_regioes]:
        mascara = (rotulos == i).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            continue
        contorno = max(contornos, key=cv2.contourArea)
        epsilon = 0.004 * cv2.arcLength(contorno, True)
        contorno = cv2.approxPolyDP(contorno, epsilon, True)
        x, y, w, h = (int(v) for v in stats[i, :4])
        regioes.append(
            Regiao(
                contorno=[(int(p[0][0]), int(p[0][1])) for p in contorno],
                caixa=(x, y, w, h),
                area=area,
                pico=pico,
                centroide=(float(centroides[i][0]), float(centroides[i][1])),
            )
        )
    return regioes


def regiao_de_mascara(mascara: np.ndarray) -> Regiao | None:
    """Cria uma região a partir de uma máscara binária (maior componente)."""
    mascara = mascara.astype(np.uint8)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contorno = max(contornos, key=cv2.contourArea)
    area = int(cv2.contourArea(contorno))
    if area <= 0:
        return None
    momento = cv2.moments(contorno)
    cx, cy = momento["m10"] / momento["m00"], momento["m01"] / momento["m00"]
    epsilon = 0.003 * cv2.arcLength(contorno, True)
    contorno = cv2.approxPolyDP(contorno, epsilon, True)
    return Regiao(
        contorno=[(int(p[0][0]), int(p[0][1])) for p in contorno],
        caixa=tuple(int(v) for v in cv2.boundingRect(contorno)),
        area=area,
        pico=1.0,
        centroide=(float(cx), float(cy)),
    )


def mascara_da_regiao(regiao: Regiao, forma: tuple[int, int]) -> np.ndarray:
    mascara = np.zeros(forma, dtype=np.uint8)
    if len(regiao.contorno) >= 3:
        cv2.fillPoly(mascara, [np.array(regiao.contorno, dtype=np.int32)], 1)
    else:
        x, y, w, h = regiao.caixa
        mascara[y:y + h, x:x + w] = 1
    return mascara.astype(bool)


# --------------------------------------------------------------------------- #
# Descrição textual
# --------------------------------------------------------------------------- #

def _juntar(itens: list[str]) -> str:
    if len(itens) <= 1:
        return "".join(itens)
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def _ordenar_zonas(zonas) -> list[str]:
    return [z for z in ZONAS if z in set(zonas)]


def descrever_local(regioes: list[Regiao], tipo_local: str) -> str:
    """Gera a expressão de localização em português para um conjunto de regiões."""
    if tipo_local == LOCAL_NENHUM or not regioes:
        return ""
    lados = {r.lado for r in regioes if r.lado}
    if not lados:
        return ""

    if tipo_local == LOCAL_LADO:
        if len(lados) == 2:
            return "bilateral"
        return "à direita" if DIREITO in lados else "à esquerda"

    if tipo_local == LOCAL_HEMITORAX:
        if len(lados) == 2:
            return "em ambos os hemitóraces"
        return f"no hemitórax {lados.pop()}"

    # Pulmonar: agrupa zonas por lado.
    zonas_por_lado: dict[str, set[str]] = {}
    for r in regioes:
        if r.lado:
            zonas_por_lado.setdefault(r.lado, set()).update(r.zonas)
    if len(zonas_por_lado) == 2 and all(len(z) == 3 for z in zonas_por_lado.values()):
        return "difusamente em ambos os pulmões"

    partes = []
    for lado in (DIREITO, ESQUERDO):
        if lado not in zonas_por_lado:
            continue
        zonas = _ordenar_zonas(zonas_por_lado[lado])
        if not zonas or len(zonas) == 3:
            partes.append(f"no pulmão {lado}")
        elif len(zonas) == 1:
            partes.append(f"no terço {zonas[0]} do pulmão {lado}")
        else:
            partes.append(f"nos terços {_juntar(zonas)} do pulmão {lado}")
    return _juntar(partes)

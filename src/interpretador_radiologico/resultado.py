"""Estruturas de dados do resultado da análise."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .anatomia import MedidaICT
from .config import Configuracao
from .dicom_io import Metadados
from .localizacao import Regiao

POSITIVO = "positivo"
INDETERMINADO = "indeterminado"
NEGATIVO = "negativo"


@dataclass
class Achado:
    chave: str
    nome: str
    escore: float
    status: str
    cor: str
    gravidade: int
    regioes: list[Regiao] = field(default_factory=list)
    local: str = ""
    suprimido: bool = False  # redundante frente a um achado mais específico
    mapa: np.ndarray | None = None  # uint8 [0, 255] na resolução de exibição

    @property
    def relevante(self) -> bool:
        return self.status != NEGATIVO

    def para_dict(self, escala: float = 1.0) -> dict:
        return {
            "chave": self.chave,
            "nome": self.nome,
            "escore": round(float(self.escore), 4),
            "status": self.status,
            "cor": self.cor,
            "gravidade": self.gravidade,
            "local": self.local,
            "suprimido": self.suprimido,
            "regioes": [r.para_dict(escala) for r in self.regioes],
        }


@dataclass
class ResultadoAnalise:
    metadados: Metadados
    nome_arquivo: str
    imagem: np.ndarray  # uint8, resolução de exibição
    escala: float  # fator exibição -> resolução original
    achados: list[Achado]
    config: Configuracao
    ict: MedidaICT | None = None
    contornos_anatomicos: dict[str, list[list[tuple[int, int]]]] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    modelo: dict = field(default_factory=dict)
    data_analise: str = ""
    tempo_s: float = 0.0

    def com_status(self, *status: str, incluir_suprimidos: bool = False) -> list[Achado]:
        return [a for a in self.achados
                if a.status in status and (incluir_suprimidos or not a.suprimido)]

    @property
    def positivos(self) -> list[Achado]:
        return self.com_status(POSITIVO)

    @property
    def indeterminados(self) -> list[Achado]:
        return self.com_status(INDETERMINADO)

    def achado(self, chave: str) -> Achado | None:
        return next((a for a in self.achados if a.chave == chave), None)

    def para_dict(self) -> dict:
        return {
            "arquivo": self.nome_arquivo,
            "data_analise": self.data_analise,
            "tempo_s": round(self.tempo_s, 2),
            "metadados": self.metadados.para_dict(),
            "dimensoes_exibicao": [int(self.imagem.shape[1]), int(self.imagem.shape[0])],
            "escala_para_original": round(self.escala, 6),
            "ict": self.ict.para_dict() if self.ict else None,
            "achados": [a.para_dict(self.escala) for a in self.achados],
            "contornos_anatomicos": {
                k: [[list(p) for p in c] for c in v] for k, v in self.contornos_anatomicos.items()
            },
            "avisos": list(self.avisos),
            "modelo": dict(self.modelo),
            "limiares": {
                "positivo": self.config.limiar_positivo,
                "indeterminado": self.config.limiar_indeterminado,
            },
        }

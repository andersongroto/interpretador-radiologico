"""Orquestração: análise -> laudo -> imagem anotada -> arquivos de saída."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .analisador import Analisador
from .config import Configuracao
from .dicom_io import Exame
from .exportacao_dicom import captura_secundaria, dicom_bytes, pdf_encapsulado
from .laudo import Laudo, gerar_laudo
from .pdf import gerar_pdf
from .resultado import ResultadoAnalise
from .visualizacao import OpcoesVisualizacao, desenhar_achados, png_bytes

FORMATOS = ("pdf", "png", "json", "txt", "dcm")


@dataclass
class Saidas:
    """Produtos de uma análise, gerados sob demanda."""

    resultado: ResultadoAnalise
    laudo: Laudo
    exame: Exame
    opcoes: OpcoesVisualizacao = field(default_factory=OpcoesVisualizacao)

    @cached_property
    def imagem_anotada(self) -> np.ndarray:
        return desenhar_achados(self.resultado, self.opcoes)

    def pdf(self, laudo: Laudo | None = None) -> bytes:
        return gerar_pdf(laudo or self.laudo, self.imagem_anotada)

    def png(self) -> bytes:
        return png_bytes(self.imagem_anotada)

    def dados(self) -> dict:
        return {"resultado": self.resultado.para_dict(), "laudo": self.laudo.para_dict()}

    def json(self) -> str:
        return json.dumps(self.dados(), ensure_ascii=False, indent=2)

    def dicom_captura(self) -> bytes:
        return dicom_bytes(captura_secundaria(self.imagem_anotada, self.exame.dataset,
                                              self.resultado.config.anonimizar))

    def dicom_pdf(self, laudo: Laudo | None = None) -> bytes:
        return dicom_bytes(pdf_encapsulado(self.pdf(laudo), self.exame.dataset,
                                           self.resultado.config.anonimizar))


def processar(exame: Exame, analisador: Analisador,
              opcoes: OpcoesVisualizacao | None = None,
              config: Configuracao | None = None) -> Saidas:
    resultado = analisador.analisar(exame, config)
    return Saidas(resultado=resultado, laudo=gerar_laudo(resultado), exame=exame,
                  opcoes=opcoes or OpcoesVisualizacao())


def nome_base(nome_arquivo: str) -> str:
    base = Path(nome_arquivo).stem if nome_arquivo else "exame"
    return re.sub(r"[^\w.-]+", "_", base).strip("._") or "exame"


def salvar(saidas: Saidas, pasta: str | Path, formatos=FORMATOS) -> list[Path]:
    """Grava os formatos pedidos em ``pasta`` e retorna os caminhos criados."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    base = nome_base(saidas.resultado.nome_arquivo)
    criados: list[Path] = []

    def gravar(sufixo: str, conteudo: bytes | str) -> None:
        caminho = pasta / f"{base}{sufixo}"
        if isinstance(conteudo, str):
            caminho.write_text(conteudo, encoding="utf-8")
        else:
            caminho.write_bytes(conteudo)
        criados.append(caminho)

    if "pdf" in formatos:
        gravar("_laudo.pdf", saidas.pdf())
    if "png" in formatos:
        gravar("_anotada.png", saidas.png())
    if "json" in formatos:
        gravar("_resultado.json", saidas.json())
    if "txt" in formatos:
        gravar("_laudo.txt", saidas.laudo.texto())
    if "dcm" in formatos:
        gravar("_anotada_sc.dcm", saidas.dicom_captura())
        gravar("_laudo_pdf.dcm", saidas.dicom_pdf())
    return criados

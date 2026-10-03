"""Leitura de arquivos DICOM (e imagens comuns) e extração de metadados."""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut

_INCIDENCIAS_PERFIL = {"LL", "RL", "LAT", "LATERAL", "LLD", "RLD", "PERFIL"}

_SEXO = {"M": "Masculino", "F": "Feminino", "O": "Outro"}
_UNIDADES_IDADE = {"Y": ("ano", "anos"), "M": ("mês", "meses"),
                   "W": ("semana", "semanas"), "D": ("dia", "dias")}


class ErroLeitura(Exception):
    """O arquivo não pôde ser lido como DICOM nem como imagem."""


@dataclass
class Metadados:
    paciente_nome: str | None = None
    paciente_id: str | None = None
    sexo: str | None = None
    idade: str | None = None
    idade_anos: float | None = None
    data_nascimento: str | None = None
    data_exame: str | None = None
    hora_exame: str | None = None
    instituicao: str | None = None
    medico_solicitante: str | None = None
    numero_acesso: str | None = None
    modalidade: str | None = None
    regiao: str | None = None
    incidencia: str | None = None
    lateralidade: str | None = None  # "direito", "esquerdo" ou "bilateral"
    descricao_estudo: str | None = None
    descricao_serie: str | None = None
    fabricante: str | None = None
    orientacao: list[str] = field(default_factory=list)
    espacamento_mm: tuple[float, float] | None = None
    fotometria: str | None = None
    linhas: int | None = None
    colunas: int | None = None
    bits: int | None = None
    study_uid: str | None = None
    series_uid: str | None = None
    sop_uid: str | None = None

    def para_dict(self) -> dict:
        return asdict(self)

    def anonimizado(self) -> "Metadados":
        return replace(
            self,
            paciente_nome="ANÔNIMO",
            paciente_id=None,
            data_nascimento=None,
            medico_solicitante=None,
            numero_acesso=None,
        )


@dataclass
class Exame:
    """Imagem radiográfica pronta para análise.

    ``imagem`` é float32 em [0, 1], na resolução original, com a convenção de
    exibição usual (estruturas densas em branco).
    """

    imagem: np.ndarray
    metadados: Metadados
    nome_arquivo: str = "exame"
    dataset: Dataset | None = None
    avisos: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Formatação de campos DICOM
# --------------------------------------------------------------------------- #

def _texto(valor) -> str | None:
    if valor is None:
        return None
    if isinstance(valor, (list, tuple, pydicom.multival.MultiValue)):
        valor = "\\".join(str(v) for v in valor)
    texto = str(valor).strip()
    return texto or None


def formatar_nome(valor) -> str | None:
    """Converte o formato DICOM "SOBRENOME^NOME^MEIO" para "NOME MEIO SOBRENOME"."""
    texto = _texto(valor)
    if not texto:
        return None
    partes = [p.strip() for p in texto.split("^")]
    if len(partes) == 1:
        return partes[0]
    familia, *demais = partes
    nome = " ".join(p for p in demais[:2] + [familia] if p)
    return nome or None


def formatar_data(valor) -> str | None:
    texto = _texto(valor)
    if not texto or not re.fullmatch(r"\d{8}", texto):
        return texto
    return f"{texto[6:8]}/{texto[4:6]}/{texto[0:4]}"


def formatar_hora(valor) -> str | None:
    texto = _texto(valor)
    if not texto or not re.match(r"\d{4}", texto):
        return texto
    return f"{texto[0:2]}:{texto[2:4]}"


def _para_data(valor) -> date | None:
    texto = _texto(valor)
    if not texto or not re.fullmatch(r"\d{8}", texto):
        return None
    try:
        return date(int(texto[0:4]), int(texto[4:6]), int(texto[6:8]))
    except ValueError:
        return None


def formatar_idade(idade_dicom, nascimento=None, data_exame=None) -> tuple[str | None, float | None]:
    """Retorna (texto, idade em anos) a partir de PatientAge ou das datas."""
    texto = _texto(idade_dicom)
    if texto:
        m = re.fullmatch(r"(\d{1,3})\s*([YMWD])", texto.upper())
        if m:
            n, unidade = int(m.group(1)), m.group(2)
            singular, plural = _UNIDADES_IDADE[unidade]
            anos = {"Y": n, "M": n / 12, "W": n / 52.18, "D": n / 365.25}[unidade]
            return f"{n} {singular if n == 1 else plural}", anos
    nasc, exame = _para_data(nascimento), _para_data(data_exame)
    if nasc and exame and exame >= nasc:
        anos = exame.year - nasc.year - ((exame.month, exame.day) < (nasc.month, nasc.day))
        if anos >= 2:
            return f"{anos} anos", float(anos)
        meses = (exame.year - nasc.year) * 12 + exame.month - nasc.month - (exame.day < nasc.day)
        return f"{meses} {'mês' if meses == 1 else 'meses'}", meses / 12
    return None, None


def _normalizar_texto(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.upper()


def _palavras(*textos: str | None) -> set[str]:
    palavras: set[str] = set()
    for texto in textos:
        if texto:
            palavras.update(re.split(r"[^A-Z0-9]+", _normalizar_texto(texto)))
    palavras.discard("")
    return palavras


def _incidencia(ds: Dataset) -> str | None:
    vista = _normalizar_texto(_texto(ds.get("ViewPosition")) or "")
    if vista in ("PA", "AP"):
        return vista
    if vista in _INCIDENCIAS_PERFIL:
        return "PERFIL"
    palavras = _palavras(_texto(ds.get("SeriesDescription")), _texto(ds.get("StudyDescription")),
                         _texto(ds.get("ProtocolName")))
    if palavras & _INCIDENCIAS_PERFIL:
        return "PERFIL"
    for vista in ("PA", "AP"):
        if vista in palavras:
            return vista
    return None


_DIREITA = {"DIREITO", "DIREITA", "DIR", "RIGHT"}
_ESQUERDA = {"ESQUERDO", "ESQUERDA", "ESQ", "LEFT"}
_BILATERAL = {"BILATERAL", "AMBOS", "AMBAS", "BOTH"}


def lateralidade(codigo: str | None, *descricoes: str | None) -> str | None:
    """Lado do exame a partir de ImageLaterality/Laterality (R/L/B) ou das descrições."""
    codigo = (codigo or "").strip().upper()
    if codigo in ("R", "L", "B"):
        return {"R": "direito", "L": "esquerdo", "B": "bilateral"}[codigo]
    palavras = _palavras(*descricoes)
    if palavras & _BILATERAL or (palavras & _DIREITA and palavras & _ESQUERDA):
        return "bilateral"
    if palavras & _DIREITA:
        return "direito"
    if palavras & _ESQUERDA:
        return "esquerdo"
    return None


def _espacamento(ds: Dataset) -> tuple[float, float] | None:
    for chave in ("PixelSpacing", "ImagerPixelSpacing"):
        valor = ds.get(chave)
        if valor is not None and len(valor) == 2:
            try:
                linha, coluna = float(valor[0]), float(valor[1])
            except (TypeError, ValueError):
                continue
            if linha > 0 and coluna > 0:
                return (linha, coluna)
    return None


def extrair_metadados(ds: Dataset) -> Metadados:
    data_exame = ds.get("StudyDate") or ds.get("AcquisitionDate") or ds.get("ContentDate")
    idade, idade_anos = formatar_idade(ds.get("PatientAge"), ds.get("PatientBirthDate"), data_exame)
    orientacao = [str(v) for v in (ds.get("PatientOrientation") or [])]
    return Metadados(
        paciente_nome=formatar_nome(ds.get("PatientName")),
        paciente_id=_texto(ds.get("PatientID")),
        sexo=_SEXO.get((_texto(ds.get("PatientSex")) or "").upper()),
        idade=idade,
        idade_anos=idade_anos,
        data_nascimento=formatar_data(ds.get("PatientBirthDate")),
        data_exame=formatar_data(data_exame),
        hora_exame=formatar_hora(ds.get("StudyTime") or ds.get("AcquisitionTime")),
        instituicao=_texto(ds.get("InstitutionName")),
        medico_solicitante=formatar_nome(ds.get("ReferringPhysicianName")),
        numero_acesso=_texto(ds.get("AccessionNumber")),
        modalidade=(_texto(ds.get("Modality")) or "").upper() or None,
        regiao=_texto(ds.get("BodyPartExamined")),
        incidencia=_incidencia(ds),
        lateralidade=lateralidade(_texto(ds.get("ImageLaterality")) or _texto(ds.get("Laterality")),
                                  _texto(ds.get("SeriesDescription")), _texto(ds.get("StudyDescription")),
                                  _texto(ds.get("ProtocolName"))),
        descricao_estudo=_texto(ds.get("StudyDescription")),
        descricao_serie=_texto(ds.get("SeriesDescription")),
        fabricante=_texto(ds.get("Manufacturer")),
        orientacao=orientacao,
        espacamento_mm=_espacamento(ds),
        fotometria=_texto(ds.get("PhotometricInterpretation")),
        linhas=int(ds.Rows) if "Rows" in ds else None,
        colunas=int(ds.Columns) if "Columns" in ds else None,
        bits=int(ds.BitsStored) if "BitsStored" in ds else None,
        study_uid=_texto(ds.get("StudyInstanceUID")),
        series_uid=_texto(ds.get("SeriesInstanceUID")),
        sop_uid=_texto(ds.get("SOPInstanceUID")),
    )


# --------------------------------------------------------------------------- #
# Pixels
# --------------------------------------------------------------------------- #

def _normalizar_percentis(arr: np.ndarray, inferior: float = 0.5, superior: float = 99.5) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [inferior, superior])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _primeiro(valor) -> float | None:
    if valor is None:
        return None
    if isinstance(valor, (list, tuple, pydicom.multival.MultiValue)):
        valor = valor[0] if len(valor) else None
    try:
        return float(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None


def janelar(arr: np.ndarray, centro: float, largura: float, funcao: str = "LINEAR") -> np.ndarray:
    """Aplica a janela VOI (PS3.3 C.11.2.1.2) e retorna valores em [0, 1]."""
    arr = arr.astype(np.float64)
    funcao = (funcao or "LINEAR").upper()
    if funcao == "SIGMOID":
        saida = 1.0 / (1.0 + np.exp(-4.0 * (arr - centro) / max(largura, 1e-6)))
    elif funcao == "LINEAR_EXACT":
        saida = (arr - (centro - largura / 2.0)) / max(largura, 1e-6)
    else:
        largura = max(largura, 1.0)
        saida = (arr - (centro - 0.5)) / max(largura - 1.0, 1e-6) + 0.5
    return np.clip(saida, 0.0, 1.0).astype(np.float32)


def pixels_normalizados(ds: Dataset, avisos: list[str] | None = None) -> np.ndarray:
    """Converte os pixels do DICOM em escala de cinza float32 [0, 1] (branco = denso)."""
    avisos = avisos if avisos is not None else []
    try:
        arr = ds.pixel_array
    except Exception as exc:  # pragma: no cover - depende de codecs instalados
        raise ErroLeitura(
            "Não foi possível decodificar os pixels do DICOM "
            f"(sintaxe de transferência {getattr(ds.file_meta, 'TransferSyntaxUID', '?')}): {exc}"
        ) from exc

    fotometria = (_texto(ds.get("PhotometricInterpretation")) or "MONOCHROME2").upper()
    quadros = int(_primeiro(ds.get("NumberOfFrames")) or 1)
    amostras = int(_primeiro(ds.get("SamplesPerPixel")) or 1)

    if quadros > 1 and arr.ndim >= 3 and (amostras == 1 or arr.ndim == 4):
        avisos.append(f"DICOM com {quadros} quadros: apenas o primeiro foi analisado.")
        arr = arr[0]

    if amostras > 1 or arr.ndim == 3:
        # Imagem colorida (RGB/YBR já convertida pelo pydicom): converte para cinza.
        return _normalizar_percentis(arr[..., :3].astype(np.float32).mean(axis=-1), 0.0, 100.0)

    arr = apply_modality_lut(arr, ds)

    if "VOILUTSequence" in ds:
        img = _normalizar_percentis(apply_voi_lut(arr, ds, prefer_lut=True), 0.0, 100.0)
    else:
        centro, largura = _primeiro(ds.get("WindowCenter")), _primeiro(ds.get("WindowWidth"))
        if centro is not None and largura is not None and largura > 0:
            img = janelar(arr, centro, largura, _texto(ds.get("VOILUTFunction")) or "LINEAR")
            if float(img.std()) < 0.02:
                avisos.append("Janela VOI do DICOM inadequada; usada normalização automática.")
                img = _normalizar_percentis(arr)
        else:
            img = _normalizar_percentis(arr)

    if fotometria == "MONOCHROME1":
        img = 1.0 - img
    return img.astype(np.float32)


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #

def _eh_dicom(cabecalho: bytes) -> bool:
    return len(cabecalho) >= 132 and cabecalho[128:132] == b"DICM"


def carregar_exame(origem: str | Path | bytes | BinaryIO, nome_arquivo: str | None = None) -> Exame:
    """Carrega um DICOM (ou PNG/JPEG, para demonstração) a partir de caminho ou bytes."""
    if isinstance(origem, (str, Path)):
        caminho = Path(origem)
        nome_arquivo = nome_arquivo or caminho.name
        dados = caminho.read_bytes()
    elif isinstance(origem, (bytes, bytearray)):
        dados = bytes(origem)
    else:
        dados = origem.read()
    nome_arquivo = nome_arquivo or "exame"

    if not dados:
        raise ErroLeitura("Arquivo vazio.")

    ds = None
    erro_dicom: Exception | None = None
    if _eh_dicom(dados) or not _parece_imagem_comum(dados):
        try:
            ds = pydicom.dcmread(io.BytesIO(dados), force=True)
            if "PixelData" not in ds:
                if _eh_dicom(dados) or len(ds) > 3:
                    raise ErroLeitura(
                        "O arquivo DICOM não contém imagem (PixelData): pode ser um laudo "
                        "estruturado, DICOMDIR ou outro objeto não-imagem."
                    )
                raise ErroLeitura("Formato de arquivo não reconhecido (esperado DICOM, PNG ou JPEG).")
        except ErroLeitura:
            raise
        except Exception as exc:
            erro_dicom = exc
            ds = None

    avisos: list[str] = []
    if ds is not None:
        imagem = pixels_normalizados(ds, avisos)
        metadados = extrair_metadados(ds)
    else:
        try:
            imagem = _carregar_imagem_comum(dados)
        except Exception as exc:
            detalhe = f" (DICOM: {erro_dicom})" if erro_dicom else ""
            raise ErroLeitura(f"Formato de arquivo não reconhecido{detalhe}.") from exc
        metadados = Metadados(linhas=imagem.shape[0], colunas=imagem.shape[1])
        avisos.append(
            "Arquivo não-DICOM: sem metadados do paciente/exame; região anatômica "
            "e incidência foram assumidas (tórax, frontal)."
        )

    if min(imagem.shape) < 256:
        avisos.append(
            f"Resolução baixa ({imagem.shape[1]}x{imagem.shape[0]} pixels): a confiabilidade "
            "da análise pode estar reduzida."
        )
    return Exame(imagem=imagem, metadados=metadados, nome_arquivo=nome_arquivo,
                 dataset=ds, avisos=avisos)


def _parece_imagem_comum(dados: bytes) -> bool:
    assinaturas = (b"\x89PNG", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM")
    return dados.startswith(assinaturas)


def _carregar_imagem_comum(dados: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(dados)) as img:
        img.load()
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            return _normalizar_percentis(np.asarray(img, dtype=np.float32), 0.0, 100.0)
        cinza = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return cinza

"""Geração de DICOM de exemplo a partir de imagens comuns (para demonstração/testes)."""

from __future__ import annotations

import io
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import DigitalXRayImageStorageForPresentation, ExplicitVRLittleEndian, generate_uid

from . import __version__

# Radiografias públicas usadas nos testes do TorchXRayVision.
# NIH ChestX-ray14 (uso irrestrito; NIH Clinical Center, Wang et al., CVPR 2017).
URL_EXEMPLO = "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png"
UID_RAIZ = "1.2.826.0.1.3680043.10.1561."  # prefixo de UIDs de exemplo


def imagem_para_dicom(
    imagem: np.ndarray,
    *,
    paciente_nome: str = "PACIENTE^EXEMPLO",
    paciente_id: str = "EX0001",
    sexo: str = "M",
    idade: str = "058Y",
    regiao: str = "CHEST",
    incidencia: str = "PA",
    modalidade: str = "DX",
    largura_campo_mm: float = 360.0,
    monochrome1: bool = False,
    bits: int = 12,
    instituicao: str = "HOSPITAL DE DEMONSTRACAO",
) -> Dataset:
    """Converte uma imagem em escala de cinza (uint8 ou float [0, 1]) em DICOM DX."""
    if imagem.ndim == 3:
        imagem = imagem[..., :3].mean(axis=-1)
    img = imagem.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    maximo = 2 ** bits - 1
    pixels = np.round(img * maximo).astype(np.uint16)
    if monochrome1:
        pixels = (maximo - pixels).astype(np.uint16)

    agora = datetime.now()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid(prefix=UID_RAIZ)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid(prefix=UID_RAIZ)

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid(prefix=UID_RAIZ)
    ds.SeriesInstanceUID = generate_uid(prefix=UID_RAIZ)
    ds.PatientName = paciente_nome
    ds.PatientID = paciente_id
    ds.PatientSex = sexo
    ds.PatientAge = idade
    ds.StudyDate = agora.strftime("%Y%m%d")
    ds.StudyTime = agora.strftime("%H%M%S")
    ds.AccessionNumber = "A" + agora.strftime("%H%M%S")
    ds.InstitutionName = instituicao
    ds.ReferringPhysicianName = "SOLICITANTE^DR"
    ds.Modality = modalidade
    ds.BodyPartExamined = regiao
    ds.ViewPosition = incidencia
    ds.StudyDescription = "RX TORAX PA" if regiao == "CHEST" else f"RX {regiao}"
    ds.SeriesDescription = incidencia
    ds.Manufacturer = f"Interpretador Radiologico {__version__} (exemplo)"
    ds.PatientOrientation = ["L", "F"]
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.PresentationIntentType = "FOR PRESENTATION"

    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RescaleType = "US"
    ds.WindowCenter = (maximo + 1) // 2
    ds.WindowWidth = maximo + 1
    espacamento = round(largura_campo_mm / pixels.shape[1], 4)
    ds.ImagerPixelSpacing = [espacamento, espacamento]
    ds.PixelData = pixels.tobytes()
    return ds


def baixar_exemplo(destino: str | Path = ".", url: str = URL_EXEMPLO, timeout: float = 60.0) -> Path:
    """Baixa uma radiografia pública de tórax e a salva como DICOM de exemplo."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resposta:  # noqa: S310 - URL fixa
        dados = resposta.read()
    with Image.open(io.BytesIO(dados)) as img:
        imagem = np.asarray(img.convert("L"))
    ds = imagem_para_dicom(imagem)
    caminho = destino / "exemplo_torax_pa.dcm"
    ds.save_as(caminho, enforce_file_format=True)
    return caminho

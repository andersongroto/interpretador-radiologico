"""Exportação para PACS: captura secundária anotada e PDF encapsulado em DICOM."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (EncapsulatedPDFStorage, ExplicitVRLittleEndian,
                         SecondaryCaptureImageStorage, generate_uid)

from . import __version__

# Atributos de paciente/estudo copiados do exame original para manter o vínculo no PACS.
_ATRIBUTOS_HERDADOS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
    "StudyInstanceUID", "StudyDate", "StudyTime", "StudyID", "AccessionNumber",
    "ReferringPhysicianName", "StudyDescription", "InstitutionName",
)
_ATRIBUTOS_IDENTIFICAVEIS = ("PatientName", "PatientID", "PatientBirthDate", "AccessionNumber",
                             "ReferringPhysicianName")


def _base(origem: Dataset | None, classe_sop: str, modalidade: str, descricao_serie: str,
          numero_serie: int, anonimizar: bool) -> Dataset:
    agora = datetime.now()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = classe_sop
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = meta
    for atributo in _ATRIBUTOS_HERDADOS:
        if origem is not None and atributo in origem:
            setattr(ds, atributo, origem.data_element(atributo).value)
    if anonimizar:
        for atributo in _ATRIBUTOS_IDENTIFICAVEIS:
            if atributo in ds:
                delattr(ds, atributo)
        ds.PatientName = "ANONIMO"
        ds.PatientIdentityRemoved = "YES"
    if "StudyInstanceUID" not in ds:
        ds.StudyInstanceUID = generate_uid()
    ds.setdefault("PatientName", "")
    ds.setdefault("PatientID", "")
    ds.SOPClassUID = classe_sop
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = modalidade
    ds.SeriesDescription = descricao_serie
    ds.SeriesNumber = numero_serie
    ds.InstanceNumber = 1
    ds.ContentDate = agora.strftime("%Y%m%d")
    ds.ContentTime = agora.strftime("%H%M%S")
    ds.Manufacturer = "Interpretador Radiologico"
    ds.SoftwareVersions = __version__
    return ds


def captura_secundaria(imagem_rgb: np.ndarray, origem: Dataset | None = None,
                       anonimizar: bool = False) -> Dataset:
    """Cria um DICOM Secondary Capture (RGB) com as anotações gravadas na imagem."""
    ds = _base(origem, SecondaryCaptureImageStorage, "OT", "IA - Achados anotados", 9901, anonimizar)
    ds.ConversionType = "WSD"
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.BurnedInAnnotation = "YES"
    ds.ImageComments = "Anotações geradas por IA para apoio à decisão; requer revisão médica."
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = imagem_rgb.shape[:2]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.ascontiguousarray(imagem_rgb[..., :3], dtype=np.uint8).tobytes()
    return ds


def pdf_encapsulado(pdf: bytes, origem: Dataset | None = None, anonimizar: bool = False,
                    titulo: str = "Laudo de radiografia de tórax (IA)") -> Dataset:
    """Cria um DICOM Encapsulated PDF com o laudo."""
    ds = _base(origem, EncapsulatedPDFStorage, "DOC", "IA - Laudo", 9902, anonimizar)
    ds.ConversionType = "WSD"
    ds.BurnedInAnnotation = "YES"
    ds.DocumentTitle = titulo
    ds.MIMETypeOfEncapsulatedDocument = "application/pdf"
    ds.ConceptNameCodeSequence = []
    if len(pdf) % 2:
        pdf += b"\x00"
    ds.EncapsulatedDocument = pdf
    return ds


def dicom_bytes(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()

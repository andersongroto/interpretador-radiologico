import io
import json

import numpy as np
import pydicom
from PIL import Image

from interpretador_radiologico.dicom_io import carregar_exame
from interpretador_radiologico.pipeline import nome_base, processar, salvar
from interpretador_radiologico.visualizacao import OpcoesVisualizacao, desenhar_achados, mapa_rgba


def _saidas(caminho_dicom, analisador_falso, **kwargs):
    analisador = analisador_falso(escores={"Nodule": 0.7, "Effusion": 0.57},
                                  focos={"Nodule": (0.3, 0.5), "Effusion": (0.7, 0.7)}, **kwargs)
    return processar(carregar_exame(caminho_dicom()), analisador)


def test_imagem_anotada_desenha_achados(caminho_dicom, analisador_falso):
    saidas = _saidas(caminho_dicom, analisador_falso)
    anotada = saidas.imagem_anotada
    assert anotada.ndim == 3 and anotada.shape[0] == 400 and anotada.shape[1] > 400  # legenda lateral
    sem_legenda = desenhar_achados(saidas.resultado, OpcoesVisualizacao(legenda=False))
    assert sem_legenda.shape == (400, 400, 3)
    limpa = desenhar_achados(saidas.resultado, OpcoesVisualizacao(
        legenda=False, mapas_calor=False, contornos=False, rotulos=False, ict=False))
    original = np.stack([saidas.resultado.imagem] * 3, axis=-1)
    assert np.array_equal(limpa, original)
    assert not np.array_equal(sem_legenda, original)
    # A cor amarela do nódulo deve aparecer sobre o pulmão direito (lado esquerdo da imagem).
    regiao = sem_legenda[150:250, 80:160].astype(int)
    assert ((regiao[..., 0] - regiao[..., 2]) > 60).any()


def test_mapa_rgba():
    mapa = np.zeros((40, 40), np.uint8)
    mapa[10:30, 10:30] = 255
    rgba = mapa_rgba(mapa, "#FF0000", reducao=4)
    assert rgba.shape == (10, 10, 4)
    assert rgba[5, 5, 3] == 255 and rgba[0, 0, 3] == 0 and tuple(rgba[5, 5, :3]) == (255, 0, 0)


def test_pdf_png_json(caminho_dicom, analisador_falso):
    saidas = _saidas(caminho_dicom, analisador_falso)
    pdf = saidas.pdf()
    assert pdf.startswith(b"%PDF") and len(pdf) > 10_000
    assert Image.open(io.BytesIO(saidas.png())).format == "PNG"
    dados = json.loads(saidas.json())
    assert dados["laudo"]["impressao"] and dados["resultado"]["achados"]
    editado = saidas.pdf(saidas.laudo.aplicar_edicao({"impressao": "Texto revisado.", "revisor": "Dr. X"}))
    assert editado.startswith(b"%PDF") and editado != pdf


def test_exportacao_dicom(caminho_dicom, analisador_falso):
    saidas = _saidas(caminho_dicom, analisador_falso)
    original = saidas.exame.dataset
    sc = pydicom.dcmread(io.BytesIO(saidas.dicom_captura()))
    assert sc.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"
    assert sc.StudyInstanceUID == original.StudyInstanceUID
    assert sc.SeriesInstanceUID != original.SeriesInstanceUID
    assert sc.PatientName == original.PatientName
    assert sc.pixel_array.shape == saidas.imagem_anotada.shape
    doc = pydicom.dcmread(io.BytesIO(saidas.dicom_pdf()))
    assert doc.MIMETypeOfEncapsulatedDocument == "application/pdf"
    assert bytes(doc.EncapsulatedDocument).startswith(b"%PDF")


def test_exportacao_dicom_anonimizada(caminho_dicom, analisador_falso):
    saidas = _saidas(caminho_dicom, analisador_falso, anonimizar=True)
    sc = pydicom.dcmread(io.BytesIO(saidas.dicom_captura()))
    assert sc.PatientName == "ANONIMO" and "AccessionNumber" not in sc
    assert sc.PatientIdentityRemoved == "YES"


def test_salvar(tmp_path, caminho_dicom, analisador_falso):
    saidas = _saidas(caminho_dicom, analisador_falso)
    criados = salvar(saidas, tmp_path / "saida", ["pdf", "png", "json", "txt", "dcm"])
    nomes = sorted(p.name for p in criados)
    assert nomes == ["torax_anotada.png", "torax_anotada_sc.dcm", "torax_laudo.pdf",
                     "torax_laudo.txt", "torax_laudo_pdf.dcm", "torax_resultado.json"]
    assert all(p.stat().st_size > 0 for p in criados)
    assert "IMPRESSÃO DIAGNÓSTICA" in (tmp_path / "saida" / "torax_laudo.txt").read_text(encoding="utf-8")


def test_nome_base():
    assert nome_base("exame 01/2024.dcm") == "2024"
    assert nome_base("Raio X tórax.dcm") == "Raio_X_tórax"
    assert nome_base("") == "exame"

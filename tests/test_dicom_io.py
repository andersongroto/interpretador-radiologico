import io

import numpy as np
import pytest
from PIL import Image

from interpretador_radiologico.dicom_io import (ErroLeitura, Metadados, carregar_exame,
                                                formatar_data, formatar_idade, formatar_nome,
                                                janelar, lateralidade)
from interpretador_radiologico.exemplo import imagem_para_dicom

from .conftest import torax_sintetico


def test_formatacao_de_campos():
    assert formatar_nome("SILVA^JOAO^CARLOS") == "JOAO CARLOS SILVA"
    assert formatar_nome("Maria Souza") == "Maria Souza"
    assert formatar_nome("") is None
    assert formatar_data("20240115") == "15/01/2024"
    assert formatar_idade("045Y") == ("45 anos", 45)
    assert formatar_idade("001Y")[0] == "1 ano"
    texto, anos = formatar_idade("006M")
    assert texto == "6 meses" and anos == pytest.approx(0.5)
    assert formatar_idade(None, "19800610", "20240609") == ("43 anos", 43.0)
    assert formatar_idade(None, "20230101", "20240301")[0] == "14 meses"


def test_janela_linear():
    arr = np.array([0, 1000, 2000, 3000, 4000], dtype=float)
    saida = janelar(arr, centro=2000, largura=2000)
    assert saida[0] == 0 and saida[-1] == 1
    assert saida[2] == pytest.approx(0.5, abs=0.01)


def test_carrega_dicom_e_metadados(caminho_dicom):
    exame = carregar_exame(caminho_dicom(paciente_nome="SOUZA^ANA", sexo="F", idade="030Y"))
    m = exame.metadados
    assert m.paciente_nome == "ANA SOUZA"
    assert m.sexo == "Feminino" and m.idade == "30 anos"
    assert m.modalidade == "DX" and m.incidencia == "PA"
    assert m.espacamento_mm and m.espacamento_mm[0] == pytest.approx(0.9)
    assert exame.imagem.dtype == np.float32
    assert 0.0 <= exame.imagem.min() and exame.imagem.max() <= 1.0
    assert exame.avisos == []


def test_monochrome1_e_invertido(caminho_dicom):
    normal = carregar_exame(caminho_dicom("m2.dcm")).imagem
    invertido = carregar_exame(caminho_dicom("m1.dcm", monochrome1=True)).imagem
    # Após a leitura, MONOCHROME1 deve ser apresentado como MONOCHROME2.
    assert np.abs(normal - invertido).mean() < 0.01


def test_rescale_e_sem_janela(tmp_path):
    ds = imagem_para_dicom(torax_sintetico())
    del ds.WindowCenter
    del ds.WindowWidth
    ds.RescaleSlope = 2
    ds.RescaleIntercept = -100
    caminho = tmp_path / "x.dcm"
    ds.save_as(caminho, enforce_file_format=True)
    img = carregar_exame(caminho).imagem
    assert img.min() == pytest.approx(0, abs=1e-6) and img.max() == pytest.approx(1, abs=1e-6)


def test_multiplos_quadros(tmp_path):
    ds = imagem_para_dicom(torax_sintetico(64))
    quadro = np.frombuffer(ds.PixelData, dtype=np.uint16).reshape(64, 64)
    ds.NumberOfFrames = 2
    ds.PixelData = np.stack([quadro, quadro]).tobytes()
    caminho = tmp_path / "mf.dcm"
    ds.save_as(caminho, enforce_file_format=True)
    exame = carregar_exame(caminho)
    assert exame.imagem.shape == (64, 64)
    assert any("quadros" in a for a in exame.avisos)
    assert any("Resolução baixa" in a for a in exame.avisos)


def test_imagem_png_e_bytes():
    buffer = io.BytesIO()
    Image.fromarray(torax_sintetico()).save(buffer, format="PNG")
    exame = carregar_exame(buffer.getvalue(), "foto.png")
    assert exame.dataset is None and exame.nome_arquivo == "foto.png"
    assert exame.imagem.shape == (400, 400)
    assert any("não-DICOM" in a for a in exame.avisos)


def test_arquivos_invalidos(tmp_path):
    with pytest.raises(ErroLeitura, match="não reconhecido"):
        carregar_exame(b"isto nao e um dicom", "lixo.dcm")
    with pytest.raises(ErroLeitura, match="vazio"):
        carregar_exame(b"", "vazio.dcm")
    ds = imagem_para_dicom(torax_sintetico(32))
    del ds.PixelData
    caminho = tmp_path / "sem_pixels.dcm"
    ds.save_as(caminho, enforce_file_format=True)
    with pytest.raises(ErroLeitura, match="PixelData"):
        carregar_exame(caminho)


@pytest.mark.parametrize("codigo, descricoes, esperado", [
    ("R", (), "direito"),
    ("L", ("JOELHO DIREITO",), "esquerdo"),
    (None, ("RX JOELHO ESQUERDO AP",), "esquerdo"),
    (None, ("RX TORAX PA E PERFIL",), None),  # "E" (conjunção) não é lateralidade
    (None, ("Mãos direita e esquerda",), "bilateral"),
    ("", ("RX PE DIREITO",), "direito"),
])
def test_lateralidade(codigo, descricoes, esperado):
    assert lateralidade(codigo, *descricoes) == esperado


def test_lateralidade_nos_metadados(tmp_path):
    ds = imagem_para_dicom(torax_sintetico(64), regiao="KNEE")
    ds.ImageLaterality = "R"
    caminho = tmp_path / "joelho.dcm"
    ds.save_as(caminho, enforce_file_format=True)
    assert carregar_exame(caminho).metadados.lateralidade == "direito"


def test_anonimizacao():
    m = Metadados(paciente_nome="ANA", paciente_id="123", numero_acesso="A1", sexo="Feminino")
    a = m.anonimizado()
    assert a.paciente_nome == "ANÔNIMO" and a.paciente_id is None and a.numero_acesso is None
    assert a.sexo == "Feminino"

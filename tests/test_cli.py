import pytest

from interpretador_radiologico import analisador as modulo_analisador
from interpretador_radiologico.cli import coletar_arquivos, main

from .conftest import ModelosFalsos


@pytest.fixture(autouse=True)
def modelos_falsos(monkeypatch):
    monkeypatch.setattr(modulo_analisador, "ModelosTorchXRayVision",
                        lambda config: ModelosFalsos({"Effusion": 0.7}, {"Effusion": (0.3, 0.7)}))


def test_analisar_arquivo(tmp_path, caminho_dicom, capsys):
    caminho = caminho_dicom()
    codigo = main(["analisar", str(caminho), "-o", str(tmp_path / "saida"), "-f", "pdf,txt"])
    assert codigo == 0
    saida = capsys.readouterr()
    assert "Derrame pleural à direita." in saida.out
    assert sorted(p.name for p in (tmp_path / "saida").iterdir()) == ["torax_laudo.pdf", "torax_laudo.txt"]


def test_analisar_pasta_com_incompativel(tmp_path, caminho_dicom, capsys):
    caminho_dicom("a.dcm")
    caminho_dicom("b.dcm", regiao="SKULL")
    sem_extensao = caminho_dicom("c.dcm")
    sem_extensao.rename(tmp_path / "IM0001")
    (tmp_path / "notas.txt").write_text("ignorar")
    assert len(coletar_arquivos([str(tmp_path)])) == 3
    codigo = main(["analisar", str(tmp_path), "-o", str(tmp_path / "saida"), "-f", "json", "-q"])
    assert codigo == 0
    erros = capsys.readouterr().err
    assert "SKULL" in erros and "1 de 3" in erros
    assert len(list((tmp_path / "saida").glob("*_resultado.json"))) == 2


def test_limiares_invalidos(caminho_dicom, capsys):
    codigo = main(["analisar", str(caminho_dicom()), "--limiar-positivo", "0.5",
                   "--limiar-indeterminado", "0.7"])
    assert codigo == 2
    assert "limiares" in capsys.readouterr().err.lower()


def test_nenhum_arquivo(tmp_path):
    assert main(["analisar", str(tmp_path / "nao_existe.dcm")]) == 1

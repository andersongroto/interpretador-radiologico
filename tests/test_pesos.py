import io

import pytest

from interpretador_radiologico import analisador
from interpretador_radiologico.analisador import baixar_pesos, carregar_com_pesos


class _Resposta(io.BytesIO):
    def __init__(self, dados: bytes, tamanho_declarado: int | None = None):
        super().__init__(dados)
        self.headers = {"Content-Length": str(tamanho_declarado if tamanho_declarado is not None else len(dados))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def servidor(monkeypatch):
    estado = {"dados": b"x" * 3_000_000, "declarado": None, "chamadas": 0}

    def urlopen(url, timeout=None):
        estado["chamadas"] += 1
        return _Resposta(estado["dados"], estado["declarado"])

    monkeypatch.setattr(analisador.urllib.request, "urlopen", urlopen)
    return estado


def test_baixa_de_forma_atomica(tmp_path, servidor):
    destino = baixar_pesos("https://exemplo/pesos.pt", tmp_path / "sub" / "pesos.pt")
    assert destino.read_bytes() == servidor["dados"]
    assert not (tmp_path / "sub" / "pesos.pt.parcial").exists()


def test_download_incompleto_nao_deixa_arquivo(tmp_path, servidor):
    servidor["declarado"] = 5_000_000  # servidor anuncia mais bytes do que entrega
    with pytest.raises(OSError, match="incompleto"):
        baixar_pesos("https://exemplo/pesos.pt", tmp_path / "pesos.pt")
    assert list(tmp_path.iterdir()) == []


def test_arquivo_corrompido_e_baixado_novamente(tmp_path, servidor):
    arquivo = tmp_path / "pesos.pt"
    arquivo.write_bytes(b"truncado")  # ex.: download interrompido anteriormente
    tentativas = []

    def construir():
        tentativas.append(arquivo.read_bytes()[:8])
        if arquivo.stat().st_size < 1000:
            raise RuntimeError("unexpected EOF, expected 38032 more bytes")
        return "modelo"

    assert carregar_com_pesos("https://exemplo/pesos.pt", tmp_path, construir) == "modelo"
    assert tentativas == [b"truncado", b"x" * 8] and servidor["chamadas"] == 1


def test_baixa_quando_ausente(tmp_path, servidor):
    assert carregar_com_pesos("https://exemplo/pesos.pt", tmp_path, lambda: "ok") == "ok"
    assert (tmp_path / "pesos.pt").stat().st_size == 3_000_000


def test_pasta_somente_leitura(tmp_path, servidor, monkeypatch):
    (tmp_path / "pesos.pt").write_bytes(b"ruim")
    monkeypatch.setattr(analisador.os, "access", lambda *a: False)

    def construir():
        raise RuntimeError("corrompido")

    with pytest.raises(RuntimeError, match="Reinstale"):
        carregar_com_pesos("https://exemplo/pesos.pt", tmp_path, construir)
    assert servidor["chamadas"] == 0

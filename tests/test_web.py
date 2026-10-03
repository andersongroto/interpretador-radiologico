import io

import pydicom
import pytest
from fastapi.testclient import TestClient

from interpretador_radiologico.web.app import criar_app


@pytest.fixture
def cliente(analisador_falso):
    analisador = analisador_falso(escores={"Nodule": 0.7}, focos={"Nodule": (0.3, 0.5)})
    return TestClient(criar_app(analisador, max_resultados=2))


def _enviar(cliente, caminho, **campos):
    with open(caminho, "rb") as f:
        return cliente.post("/api/analisar", files={"arquivo": (caminho.name, f, "application/dicom")},
                            data={k: str(v).lower() for k, v in campos.items()})


def test_pagina_e_status(cliente):
    pagina = cliente.get("/")
    assert pagina.status_code == 200 and "Interpretador Radiológico" in pagina.text
    assert cliente.get("/static/app.js").status_code == 200
    status = cliente.get("/api/status").json()
    assert status["limiares"] == {"positivo": 0.6, "indeterminado": 0.55}


def test_analisar_e_baixar(cliente, caminho_dicom):
    resposta = _enviar(cliente, caminho_dicom())
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["imagem"].startswith("data:image/png;base64,")
    nodulo = next(a for a in dados["achados"] if a["chave"] == "Nodule")
    assert nodulo["status"] == "positivo" and nodulo["mapa"].startswith("data:image/png")
    assert nodulo["regioes"][0]["contorno"]
    assert next(a for a in dados["achados"] if a["status"] == "negativo")["mapa"] is None
    assert dados["laudo"]["impressao"][0].startswith("Nódulo pulmonar")

    base = f"/api/resultados/{dados['id']}"
    png = cliente.get(f"{base}/anotada.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert "torax_anotada.png" in png.headers["content-disposition"]
    assert cliente.get(f"{base}/resultado.json").json()["laudo"]
    assert pydicom.dcmread(io.BytesIO(cliente.get(f"{base}/anotada.dcm").content)).Modality == "OT"
    assert cliente.get(f"{base}/laudo.pdf").content.startswith(b"%PDF")

    edicao = {"impressao": "1. Nódulo a esclarecer.", "revisor": "Dr. Y — CRM 2"}
    texto = cliente.post(f"{base}/laudo.txt", json=edicao).text
    assert "1. Nódulo a esclarecer." in texto and "Revisado por: Dr. Y — CRM 2" in texto
    assert cliente.post(f"{base}/laudo.pdf", json=edicao).content.startswith(b"%PDF")
    doc = pydicom.dcmread(io.BytesIO(cliente.post(f"{base}/laudo.dcm", json=edicao).content))
    assert doc.Modality == "DOC"


def test_erros(cliente, caminho_dicom, tmp_path):
    joelho = _enviar(cliente, caminho_dicom("joelho.dcm", regiao="KNEE"))
    assert joelho.status_code == 422 and "KNEE" in joelho.json()["detail"]
    assert _enviar(cliente, caminho_dicom("joelho.dcm", regiao="KNEE"), forcar=True).status_code == 200

    lixo = tmp_path / "lixo.dcm"
    lixo.write_bytes(b"0" * 300)
    resposta = _enviar(cliente, lixo)
    assert resposta.status_code == 400
    assert cliente.get("/api/resultados/inexistente/laudo.pdf").status_code == 404


def test_anonimizar_e_limite_de_resultados(cliente, caminho_dicom):
    ids = [_enviar(cliente, caminho_dicom(), anonimizar=True).json()["id"] for _ in range(3)]
    assert cliente.get(f"/api/resultados/{ids[0]}/anotada.png").status_code == 404  # descartado (LRU)
    dados = cliente.get(f"/api/resultados/{ids[-1]}/resultado.json").json()
    assert dados["resultado"]["metadados"]["paciente_nome"] == "ANÔNIMO"

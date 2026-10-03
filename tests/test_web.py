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
    assert joelho.status_code == 422 and "KNEE" in joelho.json()["detail"]["mensagem"]
    assert joelho.json()["detail"]["acao"] == "configurar_nuvem"
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


def test_status_lista_regioes(cliente):
    status = cliente.get("/api/status").json()
    assert {"chave": "joelho", "nome": "Joelho"} in status["regioes"]
    assert status["configuravel"] is False and status["exemplo"] is False
    assert cliente.get("/api/configuracoes").status_code == 404
    assert cliente.post("/api/exemplo").status_code == 404


def test_configuracoes_e_exemplo(tmp_path, analisador_falso, caminho_dicom, monkeypatch):
    from interpretador_radiologico.preferencias import carregar

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    analisador = analisador_falso()
    caminho = tmp_path / "prefs.json"
    app = TestClient(criar_app(analisador, exemplo=caminho_dicom(), preferencias_caminho=caminho))
    inicial = app.get("/api/configuracoes").json()
    assert inicial["usar_nuvem"] is False and inicial["chave_configurada"] is False
    assert "chave_api" not in inicial

    resposta = app.put("/api/configuracoes", json={"usar_nuvem": True, "chave_api": "sk-ant-teste-1234",
                                                    "nome_instituicao": "Clínica Y", "limiar_positivo": 0.7})
    dados = resposta.json()
    assert dados["chave_configurada"] and dados["chave_final"] == "1234" and "sk-ant" not in resposta.text
    assert analisador.config.usar_nuvem and analisador.config.chave_api == "sk-ant-teste-1234"
    assert analisador.config.limiar_positivo == 0.7
    assert carregar(caminho).nome_instituicao == "Clínica Y"
    # Chave vazia mantém a atual; remover_chave apaga.
    app.put("/api/configuracoes", json={"chave_api": ""})
    assert carregar(caminho).chave_api == "sk-ant-teste-1234"
    app.put("/api/configuracoes", json={"remover_chave": True})
    assert carregar(caminho).chave_api == ""
    assert app.put("/api/configuracoes", json={"limiar_positivo": 0.3}).status_code == 400

    exemplo = app.post("/api/exemplo")
    assert exemplo.status_code == 200 and exemplo.json()["arquivo"] == "torax.dcm"
    assert app.get("/api/status").json()["exemplo"] is True


def test_regiao_e_erro_da_nuvem(analisador_falso, caminho_dicom):
    from interpretador_radiologico.motor_nuvem import ErroNuvem

    from .conftest import MotorNuvemFalso

    motor = MotorNuvemFalso()
    cliente = TestClient(criar_app(analisador_falso(usar_nuvem=True, motor_nuvem=motor)))
    dados = _enviar(cliente, caminho_dicom(), regiao="joelho").json()  # DICOM de tórax, região forçada
    assert dados["motor"] == "nuvem" and dados["regiao"] == "joelho"
    assert dados["laudo"]["titulo"] == "LAUDO DE RADIOGRAFIA DO JOELHO DIREITO"
    assert dados["achados"][0]["rotulo_confianca"] == "confiança alta" and dados["achados"][0]["mapa"] is None
    assert _enviar(cliente, caminho_dicom(), regiao="inexistente").status_code == 400

    motor.erro = ErroNuvem("Sem conexão com a API da Anthropic.")
    erro = _enviar(cliente, caminho_dicom(regiao="HAND"))
    assert erro.status_code == 502 and "Sem conexão" in erro.json()["detail"]["mensagem"]

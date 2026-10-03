import base64
import io
import json

import anthropic
import httpx2
import numpy as np
import pytest
from PIL import Image

from interpretador_radiologico.config import Configuracao
from interpretador_radiologico.dicom_io import Metadados, carregar_exame
from interpretador_radiologico.laudo import gerar_laudo
from interpretador_radiologico.motor_nuvem import (ErroNuvem, InterpretacaoIA, MotorNuvem,
                                                   imagem_para_envio)
from interpretador_radiologico.pipeline import processar
from interpretador_radiologico.regioes import REGIOES
from interpretador_radiologico.resultado import INDETERMINADO, MOTOR_NUVEM, POSITIVO

from .conftest import MotorNuvemFalso, interpretacao_falsa


# --------------------------------------------------------------------------- #
# Requisição real do SDK com transporte simulado (sem rede)
# --------------------------------------------------------------------------- #

def _cliente_simulado(capturas: list, resposta: dict | None = None, status: int = 200):
    def tratar(requisicao: httpx2.Request) -> httpx2.Response:
        capturas.append(requisicao)
        if status != 200:
            return httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": "falha"}})
        corpo = {
            "id": "msg_teste", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(resposta)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }
        return httpx2.Response(200, json=corpo)

    return anthropic.Anthropic(
        api_key="chave-de-teste", base_url="http://api.teste", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(tratar)),
    )


def test_requisicao_e_resposta_estruturada():
    capturas: list = []
    esperado = interpretacao_falsa().model_dump()
    motor = MotorNuvem(Configuracao(usar_nuvem=True), _cliente_simulado(capturas, esperado))
    meta = Metadados(paciente_nome="FULANO DE TAL", paciente_id="123", sexo="Masculino", idade="40 anos",
                     incidencia="AP", lateralidade="direito")
    imagem = np.random.default_rng(0).random((2000, 1500)).astype(np.float32)

    resultado = motor.interpretar(imagem, meta, REGIOES["joelho"], "BodyPartExamined")

    assert isinstance(resultado, InterpretacaoIA) and resultado.achados[0].confianca == "alta"
    requisicao = capturas[0]
    assert requisicao.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in requisicao.headers.get("anthropic-beta", "")
    corpo = json.loads(requisicao.content)
    assert corpo["model"] == "claude-opus-5-5"
    assert corpo["fallbacks"] == "default"
    assert corpo["output_config"]["effort"] == "high"
    assert corpo["output_config"]["format"]["type"] == "json_schema"
    conteudo = corpo["messages"][0]["content"]
    assert conteudo[0]["type"] == "image" and conteudo[0]["source"]["media_type"] == "image/png"
    png = Image.open(io.BytesIO(base64.b64decode(conteudo[0]["source"]["data"])))
    assert max(png.size) == 1568  # reduzida para o tamanho útil do modelo
    texto = conteudo[1]["text"]
    assert "Joelho" in texto and "sexo masculino" in texto and "40 anos" in texto
    # Nenhum dado identificável do paciente é enviado.
    assert "FULANO" not in json.dumps(corpo) and "123" not in texto


def test_modelo_sem_fallback():
    capturas: list = []
    motor = MotorNuvem(Configuracao(modelo_nuvem="claude-haiku-4-5"),
                       _cliente_simulado(capturas, interpretacao_falsa().model_dump()))
    motor.interpretar(np.zeros((64, 64), np.float32), Metadados(), None)
    corpo = json.loads(capturas[0].content)
    assert "fallbacks" not in corpo
    assert "Região anatômica não informada" in corpo["messages"][0]["content"][1]["text"]


@pytest.mark.parametrize("status, trecho", [(401, "inválida"), (429, "Limite"), (529, "indisponível")])
def test_erros_da_api(status, trecho):
    motor = MotorNuvem(Configuracao(), _cliente_simulado([], status=status))
    with pytest.raises(ErroNuvem, match=trecho):
        motor.interpretar(np.zeros((64, 64), np.float32), Metadados(), None)


def test_sem_chave(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ErroNuvem, match="Chave de API"):
        MotorNuvem(Configuracao()).interpretar(np.zeros((8, 8), np.float32), Metadados(), None)


def test_recusa():
    class Resposta:
        stop_reason = "refusal"
        parsed_output = None

    class Cliente:
        class beta:  # noqa: N801
            class messages:  # noqa: N801
                @staticmethod
                def parse(**_):
                    return Resposta()

    with pytest.raises(ErroNuvem, match="recusou"):
        MotorNuvem(Configuracao(), Cliente()).interpretar(np.zeros((8, 8), np.float32), Metadados(), None)


def test_imagem_para_envio_preserva_pequenas():
    dados, largura, altura = imagem_para_envio(np.ones((300, 200), np.float32))
    assert (largura, altura) == (200, 300)
    assert Image.open(io.BytesIO(base64.b64decode(dados))).size == (200, 300)


# --------------------------------------------------------------------------- #
# Roteamento e integração com o analisador
# --------------------------------------------------------------------------- #

def test_joelho_vai_para_nuvem(caminho_dicom, analisador_falso):
    motor = MotorNuvemFalso()
    analisador = analisador_falso(usar_nuvem=True, motor_nuvem=motor)
    saidas = processar(carregar_exame(caminho_dicom(regiao="KNEE")), analisador)
    r = saidas.resultado
    assert motor.chamadas[0]["regiao"] == "joelho"
    assert analisador.modelos.chamadas == 0  # modelos de tórax não executados
    assert r.motor == MOTOR_NUVEM and r.regiao == "joelho" and r.regiao_nome == "Joelho direito"
    fratura, derrame = r.achados
    assert fratura.status == POSITIVO and fratura.gravidade == 3 and fratura.escore is None
    assert fratura.rotulo_confianca == "confiança alta"
    assert fratura.regioes[0].caixa == (80, 200, 80, 60)  # 200–400 x 500–650 de 1000 em 400 px
    assert derrame.status == INDETERMINADO and derrame.regioes == []

    laudo = saidas.laudo
    assert laudo.titulo == "LAUDO DE RADIOGRAFIA DO JOELHO DIREITO"
    assert laudo.impressao == ["Fratura do planalto tibial lateral direito."]
    assert ("Estruturas ósseas", ["Traço de fratura no planalto tibial lateral."]) in laudo.analise
    assert laudo.achados_indeterminados[0].startswith("Derrame articular")
    assert "Considerar tomografia para planejamento cirúrgico." in laudo.recomendacoes
    assert "Avaliação limitada da patela no perfil." in laudo.observacoes
    assert "IA multimodal em nuvem" in laudo.tecnica
    # Saídas completas também funcionam com achados sem escore.
    assert saidas.pdf().startswith(b"%PDF")
    assert saidas.imagem_anotada.shape[0] == 400
    assert json.loads(saidas.json())["resultado"]["interpretacao"]["regiao"] == "joelho"


def test_joelho_sem_nuvem_orienta_configuracao(caminho_dicom, analisador_falso):
    from interpretador_radiologico.analisador import ExameIncompativel

    with pytest.raises(ExameIncompativel) as erro:
        analisador_falso().analisar(carregar_exame(caminho_dicom(regiao="KNEE")))
    assert erro.value.acao == "configurar_nuvem" and "Configurações" in str(erro.value)


def test_regiao_desconhecida_com_pulmoes_vai_para_torax(caminho_dicom, analisador_falso):
    motor = MotorNuvemFalso()
    analisador = analisador_falso(usar_nuvem=True, motor_nuvem=motor)
    exame = carregar_exame(caminho_dicom(regiao=""))
    exame.metadados.descricao_estudo = exame.metadados.descricao_serie = None
    resultado = analisador.analisar(exame)
    assert resultado.motor == "local" and motor.chamadas == []
    assert any("identificada como tórax" in a for a in resultado.avisos)


def test_regiao_desconhecida_sem_pulmoes(caminho_dicom, analisador_falso):
    from interpretador_radiologico.analisador import ExameIncompativel

    exame = carregar_exame(caminho_dicom(regiao=""))
    exame.metadados.descricao_estudo = exame.metadados.descricao_serie = None
    with pytest.raises(ExameIncompativel, match="não parece uma radiografia de tórax"):
        analisador_falso(pulmoes=False).analisar(exame)
    motor = MotorNuvemFalso(interpretacao_falsa(regiao="mao", achados=False))
    resultado = analisador_falso(pulmoes=False, usar_nuvem=True, motor_nuvem=motor).analisar(exame)
    assert motor.chamadas[0]["regiao"] is None  # a IA identifica a região
    assert resultado.regiao == "mao" and resultado.achados == []
    assert gerar_laudo(resultado).impressao == ["Estudo radiográfico sem alterações significativas."]


def test_regiao_informada_pelo_usuario(caminho_dicom, analisador_falso):
    motor = MotorNuvemFalso()
    analisador = analisador_falso(usar_nuvem=True, motor_nuvem=motor, regiao="joelho")
    resultado = analisador.analisar(carregar_exame(caminho_dicom()))  # DICOM diz CHEST
    assert motor.chamadas[0]["fonte"] == "informada pelo usuário" and resultado.regiao == "joelho"


def test_divergencia_de_regiao_gera_aviso(caminho_dicom, analisador_falso):
    motor = MotorNuvemFalso(interpretacao_falsa(regiao="tornozelo"))
    resultado = analisador_falso(usar_nuvem=True, motor_nuvem=motor).analisar(
        carregar_exame(caminho_dicom(regiao="KNEE")))
    assert any("diferente da informada" in a for a in resultado.avisos)


def test_erro_da_nuvem_propaga(caminho_dicom, analisador_falso):
    motor = MotorNuvemFalso(erro=ErroNuvem("Sem conexão"))
    with pytest.raises(ErroNuvem):
        analisador_falso(usar_nuvem=True, motor_nuvem=motor).analisar(carregar_exame(caminho_dicom(regiao="HAND")))


def test_modalidade_nao_radiografica(caminho_dicom, analisador_falso):
    from interpretador_radiologico.analisador import ExameIncompativel

    with pytest.raises(ExameIncompativel) as erro:
        analisador_falso().analisar(carregar_exame(caminho_dicom(modalidade="CT")))
    assert erro.value.acao == "forcar"

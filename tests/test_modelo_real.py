"""Testes com os modelos reais do TorchXRayVision.

Executados apenas com ``pytest -m modelo`` (baixam ~230 MB de pesos e uma
radiografia pública na primeira execução).
"""

import pytest

from interpretador_radiologico.analisador import Analisador
from interpretador_radiologico.dicom_io import carregar_exame
from interpretador_radiologico.laudo import gerar_laudo

pytestmark = pytest.mark.modelo


@pytest.fixture(scope="module")
def exemplo(tmp_path_factory):
    from interpretador_radiologico.exemplo import baixar_exemplo

    try:
        return baixar_exemplo(tmp_path_factory.mktemp("exemplo"))
    except OSError as exc:
        pytest.skip(f"sem acesso à radiografia de exemplo: {exc}")


@pytest.fixture(scope="module")
def analisador():
    a = Analisador()
    try:
        a.carregar()
    except Exception as exc:  # noqa: BLE001 - sem rede para baixar os pesos
        pytest.skip(f"pesos indisponíveis: {exc}")
    return a


def test_exemplo_nih_com_cardiomegalia(exemplo, analisador):
    # NIH ChestX-ray14 00000001_000.png tem rótulo de referência "Cardiomegaly".
    resultado = analisador.analisar(carregar_exame(exemplo))
    cardio = resultado.achado("Cardiomegaly")
    assert cardio.status == "positivo"
    assert resultado.positivos[0].chave == "Cardiomegaly"
    assert resultado.ict is not None and resultado.ict.indice > 0.5
    assert resultado.avisos == []
    laudo = gerar_laudo(resultado)
    assert laudo.impressao[0] == "Cardiomegalia."


def test_refino_nao_altera_escores(exemplo, analisador):
    from dataclasses import replace

    exame = carregar_exame(exemplo)
    com = analisador.analisar(exame)
    sem = analisador.analisar(exame, replace(analisador.config, refinar_localizacao=False))
    assert {a.chave: round(a.escore, 5) for a in com.achados} == \
        {a.chave: round(a.escore, 5) for a in sem.achados}

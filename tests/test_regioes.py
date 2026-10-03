import pytest

from interpretador_radiologico.dicom_io import Metadados
from interpretador_radiologico.regioes import (MOTOR_LOCAL, MOTOR_NUVEM, REGIOES, identificar_regiao,
                                               modalidade_aceita)


@pytest.mark.parametrize("campos, esperado", [
    ({"modalidade": "DX", "regiao": "CHEST"}, "torax"),
    ({"modalidade": "CR", "descricao_estudo": "RX DE TÓRAX PA E PERFIL"}, "torax"),
    ({"modalidade": "DX", "regiao": "KNEE"}, "joelho"),
    ({"modalidade": "CR", "descricao_estudo": "RX MAO ESQUERDA"}, "mao"),
    ({"modalidade": "CR", "descricao_estudo": "RX COLUNA TORÁCICA AP E PERFIL"}, "coluna_toracica"),
    ({"modalidade": "CR", "descricao_serie": "Coluna lombossacra"}, "coluna_lombar"),
    ({"modalidade": "DX", "regiao": "LSPINE"}, "coluna_lombar"),
    ({"modalidade": "DX", "regiao": "PELVIS"}, "bacia"),
    ({"modalidade": "DX", "descricao_estudo": "RX PÉ DIREITO"}, "pe"),
    ({"modalidade": "CR", "descricao_estudo": "ABDOME AGUDO"}, "abdome"),
    ({"modalidade": "DX", "descricao_estudo": "RX TORAX E COSTELAS"}, "costelas"),
    ({"modalidade": "MG"}, "mama"),
    ({"modalidade": "PX"}, "dental"),
    ({"modalidade": "DX", "regiao": "SKULL"}, "cranio"),
    ({"modalidade": "DX", "regiao": "EXTREMITY"}, "outra"),
    ({"modalidade": "DX"}, None),
    ({}, None),
])
def test_identificar_regiao(campos, esperado):
    assert identificar_regiao(Metadados(**campos))[0] == esperado


def test_modalidades():
    assert modalidade_aceita(Metadados(modalidade="DX"))[0]
    assert modalidade_aceita(Metadados(modalidade="MG"))[0]
    assert modalidade_aceita(Metadados())[0]
    aceita, motivo = modalidade_aceita(Metadados(modalidade="CT"))
    assert not aceita and "CT" in motivo


def test_titulos_e_motores():
    assert REGIOES["torax"].motor == MOTOR_LOCAL
    assert all(r.motor == MOTOR_NUVEM for k, r in REGIOES.items() if k != "torax")
    assert REGIOES["joelho"].titulo_com_lado("direito") == "RADIOGRAFIA DO JOELHO DIREITO"
    assert REGIOES["mao"].titulo_com_lado("esquerdo") == "RADIOGRAFIA DA MÃO ESQUERDA"
    assert REGIOES["joelho"].titulo_com_lado("bilateral") == "RADIOGRAFIA DO JOELHO (BILATERAL)"
    assert REGIOES["bacia"].titulo_com_lado("direito") == "RADIOGRAFIA DA BACIA"  # região não par
    assert all(r.estruturas for r in REGIOES.values())

import numpy as np
import pytest

from interpretador_radiologico.anatomia import Anatomia
from interpretador_radiologico.localizacao import (DIREITO, ESQUERDO, Regiao, descrever_local,
                                                   extrair_regioes, mascara_da_regiao,
                                                   normalizar_mapa, regiao_de_mascara)
from interpretador_radiologico.patologias import (LOCAL_HEMITORAX, LOCAL_LADO, LOCAL_NENHUM,
                                                  LOCAL_PULMONAR, REGIAO_PULMAO)

from .conftest import ModelosFalsos


def _gaussiana(forma, centro, sigma):
    yy, xx = np.mgrid[0:forma[0], 0:forma[1]]
    return np.exp(-((xx - centro[0]) ** 2 + (yy - centro[1]) ** 2) / (2 * sigma ** 2))


def _anatomia(tamanho=400, invertida=False, **kwargs):
    import cv2
    mascaras = {k: cv2.resize(v, (tamanho, tamanho)) > 0.5
                for k, v in ModelosFalsos(**kwargs).segmentar(np.zeros((512, 512))).items()}
    return Anatomia((tamanho, tamanho), mascaras, invertida=invertida)


def _regiao(lado, zonas):
    return Regiao([], (0, 0, 1, 1), 1, 1.0, (0, 0), lado, zonas)


def test_extrair_regioes_ordena_por_pico_e_filtra():
    mapa = _gaussiana((200, 200), (50, 60), 10) + 0.8 * _gaussiana((200, 200), (150, 140), 10)
    mapa += 0.65 * _gaussiana((200, 200), (150, 30), 1.0)  # pico minúsculo: descartado pela área
    regioes = extrair_regioes(normalizar_mapa(mapa))
    assert len(regioes) == 2
    assert regioes[0].pico > regioes[1].pico
    x, y, w, h = regioes[0].caixa
    assert x < 50 < x + w and y < 60 < y + h
    assert len(regioes[0].contorno) >= 3


def test_extrair_regioes_mapa_vazio():
    assert extrair_regioes(np.zeros((50, 50))) == []
    assert normalizar_mapa(-np.ones((5, 5))).max() == 0


def test_regiao_de_mascara_e_volta():
    m = np.zeros((100, 100), bool)
    m[20:60, 30:80] = True
    r = regiao_de_mascara(m)
    assert r.caixa == (30, 20, 50, 40)
    assert np.abs(mascara_da_regiao(r, m.shape).astype(int) - m).sum() < 0.05 * m.sum()


@pytest.mark.parametrize("regioes, tipo, esperado", [
    ([_regiao(DIREITO, ["inferior"])], LOCAL_PULMONAR, "no terço inferior do pulmão direito"),
    ([_regiao(ESQUERDO, ["médio", "superior"])], LOCAL_PULMONAR, "nos terços superior e médio do pulmão esquerdo"),
    ([_regiao(DIREITO, ["superior", "médio", "inferior"])], LOCAL_PULMONAR, "no pulmão direito"),
    ([_regiao(DIREITO, ["inferior"]), _regiao(ESQUERDO, ["médio"])], LOCAL_PULMONAR,
     "no terço inferior do pulmão direito e no terço médio do pulmão esquerdo"),
    ([_regiao(DIREITO, list(("superior", "médio", "inferior"))),
      _regiao(ESQUERDO, ["superior", "médio", "inferior"])], LOCAL_PULMONAR, "difusamente em ambos os pulmões"),
    ([_regiao(ESQUERDO, ["inferior"])], LOCAL_LADO, "à esquerda"),
    ([_regiao(DIREITO, []), _regiao(ESQUERDO, [])], LOCAL_LADO, "bilateral"),
    ([_regiao(DIREITO, [])], LOCAL_HEMITORAX, "no hemitórax direito"),
    ([_regiao(DIREITO, [])], LOCAL_NENHUM, ""),
    ([], LOCAL_PULMONAR, ""),
])
def test_descrever_local(regioes, tipo, esperado):
    assert descrever_local(regioes, tipo) == esperado


def test_lateralidade_e_zonas_com_segmentacao():
    anatomia = _anatomia()
    m = np.zeros((400, 400), bool)
    m[250:280, 100:130] = True  # metade esquerda da imagem, parte baixa do pulmão
    assert anatomia.lado_e_zonas(m) == (DIREITO, ["inferior"])
    m = np.zeros((400, 400), bool)
    m[80:110, 270:300] = True
    assert anatomia.lado_e_zonas(m) == (ESQUERDO, ["superior"])


def test_lateralidade_imagem_espelhada():
    anatomia = _anatomia(invertida=True)
    m = np.zeros((400, 400), bool)
    m[250:280, 100:130] = True
    assert anatomia.lado_e_zonas(m)[0] == ESQUERDO


def test_lateralidade_sem_segmentacao():
    anatomia = Anatomia((400, 400))
    m = np.zeros((400, 400), bool)
    m[300:320, 300:320] = True
    assert anatomia.lado_e_zonas(m) == (ESQUERDO, ["inferior"])
    assert anatomia.indice_cardiotoracico() is None
    assert anatomia.prior(REGIAO_PULMAO).shape == (400, 400)


def test_indice_cardiotoracico():
    anatomia = _anatomia()
    ict = anatomia.indice_cardiotoracico(espacamento_mm=(0.5, 0.5), escala=2.0)
    assert ict.indice == pytest.approx(0.40, abs=0.02)  # 0,28 / 0,70
    assert ict.largura_torax_cm == pytest.approx(ict.largura_torax_px * 0.1)
    (x0, y0), (x1, y1) = ict.linha_coracao
    assert y0 == y1 and x1 > x0


def test_avisos_de_qualidade():
    assert _anatomia().avisos_qualidade() == []
    coracao_a_direita_do_paciente = ((0.40, 0.62), (0.12, 0.10))
    avisos = _anatomia(coracao=coracao_a_direita_do_paciente).avisos_qualidade()
    assert any("dextrocardia" in a for a in avisos)

import numpy as np
import pytest

from interpretador_radiologico.analisador import (DESLOCAMENTOS_REFINO, ExameIncompativel,
                                                  Geometria, classificar_status,
                                                  redimensionar_para_exibicao, transladar)
from interpretador_radiologico.config import Configuracao
from interpretador_radiologico.dicom_io import carregar_exame
from interpretador_radiologico.resultado import INDETERMINADO, NEGATIVO, POSITIVO


def test_configuracao_valida_limiares():
    with pytest.raises(ValueError):
        Configuracao(limiar_positivo=0.5, limiar_indeterminado=0.6)


def test_classificar_status():
    cfg = Configuracao()
    assert classificar_status(0.61, cfg) == POSITIVO
    assert classificar_status(0.57, cfg) == INDETERMINADO
    assert classificar_status(0.40, cfg) == NEGATIVO


def test_geometria_preenche_sem_recortar():
    geo = Geometria(100, 60)
    img = np.ones((100, 60), np.float32)
    quadrado = geo.quadrado(img, 50)
    assert quadrado.shape == (50, 50)
    assert quadrado[:, 0].max() == 0 and quadrado[25, 25] == pytest.approx(1)
    mapa = np.zeros((50, 50), np.float32)
    mapa[20:30, 20:30] = 1
    volta = geo.para_exibicao(mapa)
    assert volta.shape == (100, 60)
    ys, xs = np.nonzero(volta > 0.5)
    assert abs(xs.mean() - 30) < 2 and abs(ys.mean() - 50) < 2


def test_transladar():
    a = np.arange(16, dtype=float).reshape(4, 4)
    b = transladar(a, 1, -1, -1)
    assert b[1, 0] == a[0, 1] and b[0].tolist() == [-1] * 4 and b[:, -1].tolist() == [-1] * 4
    assert (0, 0) in DESLOCAMENTOS_REFINO and len(DESLOCAMENTOS_REFINO) == 9


def test_redimensionar_para_exibicao():
    img, escala = redimensionar_para_exibicao(np.zeros((3000, 2400), np.float32), 1000)
    assert img.shape == (1000, 800) and escala == pytest.approx(3.0)
    img, escala = redimensionar_para_exibicao(np.zeros((300, 200), np.float32), 1000)
    assert img.shape == (300, 200) and escala == 1.0


def test_achado_localizado_no_pulmao_e_terco_corretos(caminho_dicom, analisador_falso):
    analisador = analisador_falso(escores={"Nodule": 0.72}, focos={"Nodule": (0.30, 0.62)})
    resultado = analisador.analisar(carregar_exame(caminho_dicom()))
    nodulo = resultado.achado("Nodule")
    assert nodulo.status == POSITIVO
    assert nodulo.local == "no terço inferior do pulmão direito"
    assert len(nodulo.regioes) == 1 and nodulo.mapa.shape == resultado.imagem.shape
    x, y, w, h = nodulo.regioes[0].caixa
    assert x < 0.30 * 400 < x + w and y < 0.62 * 400 < y + h
    assert [a.chave for a in resultado.positivos] == ["Nodule"]
    assert resultado.ict.indice == pytest.approx(0.40, abs=0.03)
    assert resultado.achados == sorted(resultado.achados, key=lambda a: -a.escore)


def test_cardiomegalia_usa_silhueta_cardiaca(caminho_dicom, analisador_falso):
    analisador = analisador_falso(escores={"Cardiomegaly": 0.7}, focos={"Cardiomegaly": (0.5, 0.6)})
    resultado = analisador.analisar(carregar_exame(caminho_dicom()))
    cardio = resultado.achado("Cardiomegaly")
    assert cardio.local == ""
    x, y, w, h = cardio.regioes[0].caixa
    assert w == pytest.approx(0.28 * 400, abs=6)


def test_supressao_de_achado_generico(caminho_dicom, analisador_falso):
    focos = {"Lung Opacity": (0.7, 0.4), "Consolidation": (0.7, 0.42), "Effusion": (0.3, 0.7)}
    analisador = analisador_falso(escores={"Lung Opacity": 0.8, "Consolidation": 0.65, "Effusion": 0.62},
                                  focos=focos)
    resultado = analisador.analisar(carregar_exame(caminho_dicom()))
    assert resultado.achado("Lung Opacity").suprimido
    assert not resultado.achado("Consolidation").suprimido
    assert {a.chave for a in resultado.positivos} == {"Consolidation", "Effusion"}
    assert resultado.achado("Effusion").local == "à direita"


def test_sem_segmentacao_continua_com_aviso(caminho_dicom, analisador_falso):
    analisador = analisador_falso(escores={"Mass": 0.7}, focos={"Mass": (0.75, 0.3)}, segmentacao=False)
    resultado = analisador.analisar(carregar_exame(caminho_dicom()))
    assert resultado.ict is None and resultado.contornos_anatomicos == {}
    assert any("Segmentação anatômica indisponível" in a for a in resultado.avisos)
    assert resultado.achado("Mass").regioes[0].lado == "esquerdo"
    assert resultado.modelo["segmentacao"] is None


def test_exame_incompativel_e_forcar(caminho_dicom, analisador_falso):
    exame = carregar_exame(caminho_dicom(regiao="KNEE"))
    with pytest.raises(ExameIncompativel, match="KNEE"):
        analisador_falso().analisar(exame)
    resultado = analisador_falso(forcar=True).analisar(exame)
    assert any("sem validade" in a for a in resultado.avisos)


def test_avisos_de_incidencia_e_idade(caminho_dicom, analisador_falso):
    resultado = analisador_falso().analisar(carregar_exame(caminho_dicom(incidencia="LL", idade="008Y")))
    assert any("perfil" in a for a in resultado.avisos)
    assert any("pediátrico" in a for a in resultado.avisos)
    resultado = analisador_falso().analisar(carregar_exame(caminho_dicom(incidencia="AP")))
    assert any("Incidência AP" in a for a in resultado.avisos)


def test_anonimizacao(caminho_dicom, analisador_falso):
    resultado = analisador_falso(anonimizar=True).analisar(carregar_exame(caminho_dicom()))
    assert resultado.metadados.paciente_nome == "ANÔNIMO"
    assert resultado.metadados.paciente_id is None


def test_para_dict_serializavel(caminho_dicom, analisador_falso):
    import json

    resultado = analisador_falso(escores={"Nodule": 0.7}, focos={"Nodule": (0.3, 0.5)}).analisar(
        carregar_exame(caminho_dicom()))
    dados = json.loads(json.dumps(resultado.para_dict()))
    assert dados["dimensoes_exibicao"] == [400, 400]
    assert len(dados["achados"]) == 18
    assert dados["achados"][0]["regioes"][0]["lado"] == "direito"

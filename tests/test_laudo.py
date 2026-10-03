import pytest

from interpretador_radiologico.dicom_io import carregar_exame
from interpretador_radiologico.laudo import formatar_frase, gerar_laudo


def _laudo(caminho_dicom, analisador_falso, **kwargs):
    resultado = analisador_falso(**kwargs).analisar(carregar_exame(caminho_dicom()))
    return resultado, gerar_laudo(resultado)


def test_formatar_frase():
    assert formatar_frase("Atelectasia {local}.", "") == "Atelectasia."
    assert formatar_frase("Nódulo {local}.", "no terço médio do pulmão direito") == \
        "Nódulo no terço médio do pulmão direito."
    assert formatar_frase("Área cardíaca normal{ict}.", ict=", com ICT 0,45") == \
        "Área cardíaca normal, com ICT 0,45."


def test_laudo_normal(caminho_dicom, analisador_falso):
    _, laudo = _laudo(caminho_dicom, analisador_falso)
    texto = laudo.texto()
    assert laudo.impressao == [
        "Radiografia de tórax sem alterações significativas identificadas pela análise automatizada."]
    assert "Seios costofrênicos livres." in texto
    assert "Ausência de sinais de pneumotórax." in texto
    assert "índice cardiotorácico estimado em 0,40" in texto
    assert laudo.recomendacoes == []
    assert [s for s, _ in laudo.analise][:3] == ["Pulmões", "Pleuras", "Coração"]
    assert "PACIENTE" in dict(laudo.cabecalho)["Paciente"]
    assert "AVISO:" in texto


def test_laudo_com_achados(caminho_dicom, analisador_falso):
    resultado, laudo = _laudo(
        caminho_dicom, analisador_falso,
        escores={"Pneumothorax": 0.66, "Nodule": 0.7, "Atelectasis": 0.57},
        focos={"Pneumothorax": (0.72, 0.25), "Nodule": (0.3, 0.62), "Atelectasis": (0.7, 0.6)},
    )
    # Ordenação por gravidade: pneumotórax (urgente) primeiro.
    assert laudo.impressao[0] == "Pneumotórax à esquerda — achado potencialmente urgente."
    assert laudo.impressao[1] == "Nódulo pulmonar no terço inferior do pulmão direito."
    assert any("tomografia" in r for r in laudo.recomendacoes)
    assert laudo.recomendacoes[0].startswith("Achado potencialmente urgente")
    assert laudo.achados_indeterminados == ["Atelectasia no terço inferior do pulmão esquerdo (escore IA 57%)."]
    pulmoes = dict(laudo.analise)["Pulmões"]
    assert pulmoes[0] == "Imagem nodular no terço inferior do pulmão direito (escore IA 70%)."
    # Atelectasia indeterminada impede a frase de normalidade do parênquima.
    assert not any("transparência preservada" in f for f in pulmoes)
    assert "Seios costofrênicos livres." in dict(laudo.analise)["Pleuras"]
    assert not any("pneumotórax" in f.lower() and "Ausência" in f for f in dict(laudo.analise)["Pleuras"])


def test_apenas_indeterminados(caminho_dicom, analisador_falso):
    _, laudo = _laudo(caminho_dicom, analisador_falso, escores={"Fibrosis": 0.56})
    assert "baixa confiança" in laudo.impressao[0]
    assert laudo.recomendacoes == ["Revisão dirigida das regiões assinaladas na imagem anotada."]


def test_ict_aumentado_sem_cardiomegalia(caminho_dicom, analisador_falso):
    resultado = analisador_falso().analisar(carregar_exame(caminho_dicom()))
    resultado.ict.indice = 0.56
    laudo = gerar_laudo(resultado)
    coracao = " ".join(dict(laudo.analise)["Coração"])
    assert "0,56" in coracao and "sem cardiomegalia identificada" in coracao
    assert "aumentado" in laudo.medidas[0]


def test_aplicar_edicao(caminho_dicom, analisador_falso):
    _, laudo = _laudo(caminho_dicom, analisador_falso)
    editado = laudo.aplicar_edicao({
        "analise": "Pulmões: Hiperinsuflação discreta.\nTexto livre sem sistema",
        "impressao": "1. Enfisema incipiente.\n2) Controle em 6 meses.\n",
        "recomendacoes": ["- Espirometria."],
        "revisor": "Dra. Ana — CRM 1",
    })
    assert editado.analise == [("Pulmões", ["Hiperinsuflação discreta."]),
                               ("Observação", ["Texto livre sem sistema"])]
    assert editado.impressao == ["Enfisema incipiente.", "Controle em 6 meses."]
    assert editado.recomendacoes == ["Espirometria."]
    assert editado.revisor == "Dra. Ana — CRM 1" and editado.editado
    assert "Revisado por: Dra. Ana — CRM 1" in editado.texto()
    assert laudo.revisor is None and laudo.impressao != editado.impressao  # original intacto


def test_instituicao_configurada(caminho_dicom, analisador_falso):
    _, laudo = _laudo(caminho_dicom, analisador_falso, nome_instituicao="Clínica X")
    assert laudo.instituicao == "Clínica X"


@pytest.mark.parametrize("chave", ["Effusion", "Pneumonia", "Fracture", "Hernia", "Enlarged Cardiomediastinum"])
def test_todas_as_frases_sao_bem_formadas(caminho_dicom, analisador_falso, chave):
    _, laudo = _laudo(caminho_dicom, analisador_falso, escores={chave: 0.7}, focos={chave: (0.3, 0.6)})
    for frase in laudo.impressao + [f for _, fs in laudo.analise for f in fs]:
        assert "{" not in frase and "  " not in frase and " ." not in frase

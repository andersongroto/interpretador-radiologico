import pytest

from interpretador_radiologico.config import Configuracao
from interpretador_radiologico.preferencias import Preferencias, carregar, salvar


def test_salvar_carregar_e_aplicar(tmp_path):
    caminho = tmp_path / "sub" / "prefs.json"
    assert carregar(caminho) == Preferencias()  # ausente -> padrão
    salvar(Preferencias(usar_nuvem=True, chave_api="sk-ant-abc12345", limiar_positivo=0.65), caminho)
    prefs = carregar(caminho)
    assert prefs.usar_nuvem and prefs.chave_api == "sk-ant-abc12345"
    cfg = prefs.aplicar(Configuracao(diretorio_pesos="/pesos"))
    assert cfg.usar_nuvem and cfg.chave_api == "sk-ant-abc12345" and cfg.limiar_positivo == 0.65
    assert cfg.diretorio_pesos == "/pesos"  # campos não relacionados preservados
    assert "sk-ant" not in repr(cfg)  # a chave não aparece em registros


def test_arquivo_corrompido(tmp_path):
    caminho = tmp_path / "prefs.json"
    caminho.write_text("{invalido", encoding="utf-8")
    assert carregar(caminho) == Preferencias()
    caminho.write_text('{"usar_nuvem": true, "campo_desconhecido": 1}', encoding="utf-8")
    assert carregar(caminho).usar_nuvem is True


def test_atualizar_valida(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prefs = Preferencias(chave_api="sk-ant-xyz98765")
    assert prefs.atualizar({"chave_api": ""}).chave_api == "sk-ant-xyz98765"
    assert prefs.atualizar({"remover_chave": True}).chave_api == ""
    with pytest.raises(ValueError):
        prefs.atualizar({"limiar_positivo": 0.4, "limiar_indeterminado": 0.5})
    publicas = prefs.publicas()
    assert publicas["chave_configurada"] and publicas["chave_final"] == "8765" and "chave_api" not in publicas

"""Preferências do usuário, persistidas em JSON no perfil do usuário."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from .config import Configuracao

NOME_APLICATIVO = "InterpretadorRadiologico"


def diretorio_dados() -> Path:
    """Pasta de dados do usuário (configurações e registros)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / NOME_APLICATIVO
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / NOME_APLICATIVO
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "interpretador-radiologico"


@dataclass
class Preferencias:
    #: Habilita o motor de IA em nuvem (Claude) para regiões sem modelo local.
    usar_nuvem: bool = False
    #: Chave da API da Anthropic. Se vazia, usa a variável ANTHROPIC_API_KEY.
    chave_api: str = ""
    modelo_nuvem: str = "claude-opus-5-5"
    nome_instituicao: str = ""
    limiar_positivo: float = 0.60
    limiar_indeterminado: float = 0.55
    anonimizar: bool = False

    def aplicar(self, config: Configuracao) -> Configuracao:
        """Retorna uma cópia da configuração com estas preferências aplicadas."""
        return replace(
            config,
            usar_nuvem=self.usar_nuvem,
            chave_api=self.chave_api or None,
            modelo_nuvem=self.modelo_nuvem or config.modelo_nuvem,
            nome_instituicao=self.nome_instituicao or None,
            limiar_positivo=self.limiar_positivo,
            limiar_indeterminado=self.limiar_indeterminado,
            anonimizar=self.anonimizar,
        )

    def publicas(self) -> dict:
        """Preferências sem expor a chave da API (apenas se está configurada)."""
        dados = asdict(self)
        chave = dados.pop("chave_api")
        dados["chave_configurada"] = bool(chave or os.environ.get("ANTHROPIC_API_KEY"))
        dados["chave_final"] = chave[-4:] if len(chave) >= 8 else ""
        return dados

    def atualizar(self, valores: dict) -> "Preferencias":
        """Cópia validada com os campos informados; ``chave_api`` vazia ou ausente mantém a atual."""
        validos = {f.name for f in fields(self)}
        novos = {k: v for k, v in valores.items() if k in validos}
        if not novos.get("chave_api"):
            novos.pop("chave_api", None)
        if valores.get("remover_chave"):
            novos["chave_api"] = ""
        atual = replace(self, **novos)
        # Valida os limiares usando as regras da Configuracao.
        Configuracao(limiar_positivo=float(atual.limiar_positivo),
                     limiar_indeterminado=float(atual.limiar_indeterminado))
        return atual


def caminho_padrao() -> Path:
    return diretorio_dados() / "preferencias.json"


def carregar(caminho: str | Path | None = None) -> Preferencias:
    caminho = Path(caminho) if caminho else caminho_padrao()
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Preferencias()
    validos = {f.name for f in fields(Preferencias)}
    try:
        return Preferencias(**{k: v for k, v in dados.items() if k in validos})
    except TypeError:
        return Preferencias()


def salvar(preferencias: Preferencias, caminho: str | Path | None = None) -> Path:
    caminho = Path(caminho) if caminho else caminho_padrao()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(json.dumps(asdict(preferencias), ensure_ascii=False, indent=2), encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(temporario, 0o600)
    temporario.replace(caminho)
    return caminho

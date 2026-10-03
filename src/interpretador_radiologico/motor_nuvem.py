"""Motor de IA multimodal em nuvem (Claude, da Anthropic) para qualquer radiografia.

Usado nas regiões sem modelo local (membros, coluna, crânio, bacia, abdome,
mamografia, odontologia...). Envia à API apenas os pixels da imagem e dados não
identificáveis (região, incidência, lado, sexo e idade); nenhum metadado DICOM
de identificação é transmitido. Atenção: textos gravados na própria imagem
(burned-in) seguem junto com os pixels.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .config import Configuracao
from .dicom_io import Metadados
from .regioes import REGIOES, Regiao

LADO_MAXIMO_ENVIO = 1568  # maior lado útil para o modelo de visão
# Modelos que aceitam o parâmetro de fallback automático no servidor.
MODELOS_COM_FALLBACK = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

ChaveRegiao = Literal[tuple(REGIOES)]  # type: ignore[valid-type]


class ErroNuvem(Exception):
    """Falha ao obter a interpretação da IA em nuvem (mensagem pronta para o usuário)."""


# --------------------------------------------------------------------------- #
# Esquema da resposta (saída estruturada)
# --------------------------------------------------------------------------- #

class CaixaIA(BaseModel):
    """Retângulo em coordenadas normalizadas de 0 a 1000 (origem no canto superior esquerdo)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class AchadoIA(BaseModel):
    nome: str = Field(description="Nome curto do achado em português (ex.: 'Fratura do rádio distal').")
    descricao: str = Field(description="Frase descritiva do achado para a seção de análise do laudo.")
    impressao: str = Field(description="Frase do achado para a impressão diagnóstica.")
    localizacao: str = Field(description="Localização anatômica (ex.: 'metáfise distal do rádio').")
    lado: Literal["direito", "esquerdo", "bilateral", "linha_media", "nao_aplicavel"] = Field(
        description="Lado do PACIENTE.")
    confianca: Literal["alta", "moderada", "baixa"]
    gravidade: Literal["urgente", "relevante", "leve"]
    caixas: list[CaixaIA] = Field(description="Regiões do achado na imagem; vazio se difuso ou impreciso.")
    recomendacao: str = Field(description="Conduta/complementação sugerida, ou string vazia.")


class SecaoIA(BaseModel):
    titulo: str = Field(description="Estrutura avaliada (ex.: 'Estruturas ósseas').")
    texto: str = Field(description="Descrição em frases completas, incluindo achados normais.")


class InterpretacaoIA(BaseModel):
    e_radiografia: bool = Field(description="False se a imagem não for uma radiografia.")
    regiao: ChaveRegiao = Field(description="Região anatômica identificada na imagem.")
    regiao_descricao: str = Field(description="Região e lado em português (ex.: 'Joelho direito').")
    lado_exame: Literal["direito", "esquerdo", "bilateral", "nao_aplicavel"]
    incidencias: str = Field(description="Incidência(s) identificada(s), ex.: 'AP e perfil'.")
    qualidade_tecnica: str = Field(description="Adequação técnica (posicionamento, exposição, limitações).")
    achados: list[AchadoIA] = Field(description="Somente achados anormais; vazio se exame normal.")
    analise: list[SecaoIA] = Field(description="Descrição sistemática por estrutura anatômica.")
    impressao: list[str] = Field(description="Impressão diagnóstica, itens ordenados por relevância.")
    recomendacoes: list[str]
    limitacoes: list[str] = Field(description="Limitações da avaliação (técnica, incidência única etc.).")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

INSTRUCOES_SISTEMA = """\
Você é um sistema de apoio à decisão em radiologia que produz pré-laudos de radiografias para \
revisão obrigatória por médico(a) radiologista. Analise a imagem recebida e responda no esquema \
estruturado solicitado.

Regras:
- Escreva em português do Brasil, com terminologia radiológica usual em laudos brasileiros \
(frases objetivas, sem primeira pessoa).
- Descreva somente o que é visível na imagem. Não invente achados nem dados clínicos. Diante de \
dúvida, use confiança "baixa" e explique a incerteza na descrição.
- "achados" contém apenas alterações. Achados normais são descritos nas seções de "analise", que \
devem percorrer sistematicamente as estruturas indicadas para a região.
- Lado: sempre o lado do PACIENTE. Use os marcadores de lateralidade gravados na imagem (R/L, D/E) \
quando houver. Em incidências frontais na convenção radiológica, o lado direito do paciente fica à \
esquerda da imagem; em extremidades, use a anatomia e os marcadores.
- Caixas: coordenadas inteiras de 0 a 1000, relativas à largura (x, da esquerda para a direita) e à \
altura (y, de cima para baixo) da imagem, envolvendo justamente a região de cada achado. Use \
caixas vazias quando o achado for difuso ou não localizável.
- Gravidade "urgente" apenas para achados que exigem ação imediata (ex.: fratura desviada, luxação, \
pneumoperitônio, corpo estranho perigoso, sinais de obstrução).
- Impressão: itens curtos e ordenados por relevância. Exame sem alterações: um único item \
"Estudo radiográfico sem alterações significativas.".
- Recomendações: apenas quando pertinentes (ex.: incidências adicionais, TC, RM, comparação com \
exames anteriores, correlação clínica).
- Se a imagem não for uma radiografia, defina e_radiografia como false e explique em "limitacoes".
- Para mamografias, sugira categoria BI-RADS na impressão; para radiografias odontológicas, use a \
numeração dentária FDI.
"""


def _texto_contexto(meta: Metadados, regiao: Regiao | None, fonte: str | None,
                    indicacao: str | None) -> str:
    linhas = ["Interprete esta radiografia e preencha o esquema estruturado."]
    if regiao:
        linhas.append(f"Região informada ({fonte or 'usuário'}): {regiao.nome}.")
        linhas.append("Estruturas a avaliar: " + "; ".join(regiao.estruturas) + ".")
    else:
        linhas.append("Região anatômica não informada: identifique-a pela imagem.")
        linhas.append("Chaves de região válidas: " + ", ".join(REGIOES) + ".")
    if meta.incidencia:
        linhas.append(f"Incidência informada no DICOM: {meta.incidencia}.")
    if meta.lateralidade:
        linhas.append(f"Lateralidade informada no DICOM: {meta.lateralidade}.")
    perfil = ", ".join(p for p in (meta.sexo and f"sexo {meta.sexo.lower()}", meta.idade) if p)
    if perfil:
        linhas.append(f"Paciente: {perfil}.")
    if indicacao:
        linhas.append(f"Indicação clínica: {indicacao}")
    return "\n".join(linhas)


def imagem_para_envio(imagem: np.ndarray) -> tuple[str, int, int]:
    """Converte a imagem float [0, 1] em PNG base64 com lado máximo ``LADO_MAXIMO_ENVIO``."""
    altura, largura = imagem.shape
    fator = min(1.0, LADO_MAXIMO_ENVIO / max(altura, largura))
    if fator < 1.0:
        imagem = cv2.resize(imagem, (round(largura * fator), round(altura * fator)),
                            interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(np.round(np.clip(imagem, 0, 1) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii"), imagem.shape[1], imagem.shape[0]


# --------------------------------------------------------------------------- #
# Cliente
# --------------------------------------------------------------------------- #

class MotorNuvem:
    """Interpreta radiografias com um modelo multimodal da Anthropic."""

    def __init__(self, config: Configuracao, cliente=None):
        self.config = config
        self._cliente = cliente

    @property
    def cliente(self):
        if self._cliente is None:
            import anthropic

            chave = self.config.chave_api or os.environ.get("ANTHROPIC_API_KEY")
            if not chave:
                raise ErroNuvem(
                    "Chave de API da Anthropic não configurada. Informe-a em Configurações "
                    "(ou na variável de ambiente ANTHROPIC_API_KEY)."
                )
            self._cliente = anthropic.Anthropic(api_key=chave, timeout=300.0, max_retries=2)
        return self._cliente

    def descricao(self) -> dict:
        return {
            "classificador": f"IA multimodal em nuvem — Anthropic Claude ({self.config.modelo_nuvem})",
            "segmentacao": None,
            "dispositivo": "nuvem (API da Anthropic)",
        }

    def interpretar(self, imagem: np.ndarray, meta: Metadados, regiao: Regiao | None,
                    fonte_regiao: str | None = None, indicacao: str | None = None) -> InterpretacaoIA:
        dados, _, _ = imagem_para_envio(imagem)
        mensagens = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": dados}},
                {"type": "text", "text": _texto_contexto(meta, regiao, fonte_regiao, indicacao)},
            ],
        }]
        parametros = dict(
            model=self.config.modelo_nuvem,
            max_tokens=16000,
            system=INSTRUCOES_SISTEMA,
            messages=mensagens,
            output_config={"effort": "high"},
            output_format=InterpretacaoIA,
        )
        if self.config.modelo_nuvem in MODELOS_COM_FALLBACK:
            # Em caso de recusa pelos classificadores de segurança, o servidor
            # repete a solicitação no modelo alternativo recomendado.
            parametros.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        resposta = self._chamar(parametros)

        if resposta.stop_reason == "refusal":
            raise ErroNuvem("A IA em nuvem recusou-se a interpretar esta imagem.")
        if resposta.stop_reason == "max_tokens":
            raise ErroNuvem("A resposta da IA em nuvem foi interrompida (limite de tamanho). Tente novamente.")
        interpretacao = getattr(resposta, "parsed_output", None)
        if interpretacao is None:
            raise ErroNuvem("A IA em nuvem não retornou uma interpretação válida.")
        return interpretacao

    def _chamar(self, parametros: dict):
        import anthropic

        try:
            return self.cliente.beta.messages.parse(**parametros)
        except ErroNuvem:
            raise
        except anthropic.AuthenticationError as exc:
            raise ErroNuvem("Chave de API inválida ou revogada (Configurações).") from exc
        except anthropic.PermissionDeniedError as exc:
            raise ErroNuvem("A chave de API não tem permissão para usar este modelo.") from exc
        except anthropic.NotFoundError as exc:
            raise ErroNuvem(f"Modelo '{self.config.modelo_nuvem}' não encontrado.") from exc
        except anthropic.RateLimitError as exc:
            raise ErroNuvem("Limite de uso da API atingido; aguarde alguns instantes.") from exc
        except anthropic.BadRequestError as exc:
            raise ErroNuvem(f"Solicitação rejeitada pela API: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise ErroNuvem(f"Serviço da Anthropic indisponível (HTTP {exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise ErroNuvem("Sem conexão com a API da Anthropic. Verifique a internet.") from exc

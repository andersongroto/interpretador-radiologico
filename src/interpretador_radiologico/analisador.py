"""Motor de análise: classificação, localização e referências anatômicas."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .anatomia import ESTRUTURAS_PSPNET, Anatomia
from .config import Configuracao
from .dicom_io import Exame
from .localizacao import (Regiao, descrever_local, extrair_regioes, mascara_da_regiao,
                          normalizar_mapa, regiao_de_mascara)
from .motor_nuvem import InterpretacaoIA, MotorNuvem
from .patologias import REGIAO_CORACAO, patologia
from .regioes import REGIOES, identificar_regiao, modalidade_aceita
from .resultado import (INDETERMINADO, MOTOR_LOCAL, MOTOR_NUVEM, NEGATIVO, POSITIVO, Achado,
                        ResultadoAnalise)

# Deslocamentos (em pixels na entrada 224x224) usados para refinar os mapas de
# ativação: a rede tem passo de 32 px, então deslocar meia célula e calcular a
# média dos mapas realinhados dobra a resolução efetiva da localização.
log = logging.getLogger(__name__)

DESLOCAMENTOS_REFINO = tuple((dy, dx) for dy in (-16, 0, 16) for dx in (-16, 0, 16))


class ExameIncompativel(Exception):
    """O exame não pode ser analisado com a configuração atual.

    ``acao`` sugere o que o usuário pode fazer: "forcar", "configurar_nuvem" ou None.
    """

    def __init__(self, mensagem: str, acao: str | None = None):
        super().__init__(mensagem)
        self.acao = acao


# --------------------------------------------------------------------------- #
# Geometria: imagem retangular <-> entrada quadrada dos modelos
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Geometria:
    """Mapeia a imagem de exibição para o quadrado (com margens) visto pelos modelos.

    Usa preenchimento em vez de recorte central para não perder ápices,
    bases ou seios costofrênicos em imagens retangulares.
    """

    altura: int
    largura: int

    @property
    def forma(self) -> tuple[int, int]:
        return (self.altura, self.largura)

    @property
    def lado(self) -> int:
        return max(self.altura, self.largura)

    @property
    def topo(self) -> int:
        return (self.lado - self.altura) // 2

    @property
    def esquerda(self) -> int:
        return (self.lado - self.largura) // 2

    def quadrado(self, imagem: np.ndarray, tamanho: int) -> np.ndarray:
        """Imagem float [0, 1] com margens pretas, redimensionada para ``tamanho``."""
        quadro = np.zeros((self.lado, self.lado), dtype=np.float32)
        quadro[self.topo:self.topo + self.altura, self.esquerda:self.esquerda + self.largura] = imagem
        return cv2.resize(quadro, (tamanho, tamanho), interpolation=cv2.INTER_AREA)

    def para_exibicao(self, mapa: np.ndarray) -> np.ndarray:
        """Converte um mapa no espaço quadrado do modelo para a imagem de exibição."""
        m = cv2.resize(mapa.astype(np.float32), (self.lado, self.lado), interpolation=cv2.INTER_LINEAR)
        return m[self.topo:self.topo + self.altura, self.esquerda:self.esquerda + self.largura]


def transladar(arr: np.ndarray, dy: int, dx: int, preenchimento: float) -> np.ndarray:
    """Desloca o conteúdo das duas últimas dimensões, preenchendo as bordas."""
    saida = np.full_like(arr, preenchimento)
    h, w = arr.shape[-2:]
    if abs(dy) >= h or abs(dx) >= w:
        return saida
    origem_y = slice(max(0, -dy), h - max(0, dy))
    destino_y = slice(max(0, dy), h - max(0, -dy))
    origem_x = slice(max(0, -dx), w - max(0, dx))
    destino_x = slice(max(0, dx), w - max(0, -dx))
    saida[..., destino_y, destino_x] = arr[..., origem_y, origem_x]
    return saida


def redimensionar_para_exibicao(imagem: np.ndarray, tamanho_maximo: int) -> tuple[np.ndarray, float]:
    """Retorna (imagem float [0, 1] reduzida, fator exibição -> original)."""
    altura, largura = imagem.shape
    fator = min(1.0, tamanho_maximo / max(altura, largura))
    if fator < 1.0:
        novo = (max(1, round(largura * fator)), max(1, round(altura * fator)))
        imagem = cv2.resize(imagem, novo, interpolation=cv2.INTER_AREA)
        escala = largura / novo[0]
    else:
        escala = 1.0
    return np.clip(imagem, 0.0, 1.0).astype(np.float32), float(escala)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #

URL_PESOS_PSPNET = ("https://github.com/mlmed/torchxrayvision/releases/download/v1/"
                    "pspnet_chestxray_best_model_4.pth")


def baixar_pesos(url: str, destino: Path, timeout: float = 120.0) -> Path:
    """Baixa um arquivo de forma atômica (arquivo temporário + verificação de tamanho)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    log.info("Baixando %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resposta, open(parcial, "wb") as saida:  # noqa: S310
        esperado = int(resposta.headers.get("Content-Length") or 0)
        shutil.copyfileobj(resposta, saida, 1024 * 1024)
    if esperado and parcial.stat().st_size != esperado:
        parcial.unlink(missing_ok=True)
        raise OSError(f"Download incompleto de {url} ({parcial.stat().st_size if parcial.exists() else 0}"
                      f" de {esperado} bytes).")
    parcial.replace(destino)
    return destino


def carregar_com_pesos(url: str, pasta: Path, construir):
    """Garante o arquivo de pesos (baixando se preciso) e constrói o modelo.

    Um arquivo corrompido (ex.: download interrompido) é baixado novamente
    uma vez, quando a pasta permite escrita.
    """
    arquivo = pasta / Path(url).name
    if not arquivo.is_file():
        baixar_pesos(url, arquivo)
    try:
        return construir()
    except (RuntimeError, EOFError, pickle.UnpicklingError, OSError) as exc:
        if not os.access(pasta, os.W_OK):
            raise RuntimeError(f"Arquivo de pesos corrompido: {arquivo}. Reinstale o programa.") from exc
        log.warning("Arquivo de pesos inválido (%s); baixando novamente: %s", exc, arquivo)
        arquivo.unlink(missing_ok=True)
        baixar_pesos(url, arquivo)
        return construir()


class Modelos(Protocol):
    def classificar(self, entrada: np.ndarray, refinar: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Recebe imagem 224x224 [0, 1]; retorna (rótulos, escores [K], mapas [K, 224, 224])."""

    def segmentar(self, entrada: np.ndarray) -> dict[str, np.ndarray]:
        """Recebe imagem 512x512 [0, 1]; retorna probabilidades [512, 512] por estrutura."""

    def descricao(self) -> dict:
        ...


class ModelosTorchXRayVision:
    """Classificador DenseNet121 e segmentador PSPNet do TorchXRayVision.

    Os pesos são baixados automaticamente na primeira execução para
    ``~/.torchxrayvision/models_data`` (ou ``Configuracao.diretorio_pesos``).
    """

    def __init__(self, config: Configuracao):
        import torch
        import torchxrayvision as xrv

        disponiveis = sorted(k for k in xrv.models.model_urls if k.startswith("densenet"))
        if config.modelo not in disponiveis:
            raise ValueError(
                f"Pesos '{config.modelo}' não suportados. Opções (DenseNet121 do TorchXRayVision): "
                + ", ".join(disponiveis)
            )
        self.config = config
        if config.dispositivo == "auto":
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = torch.device(config.dispositivo)
        self._classificador = None
        self._segmentador = None

    @property
    def pasta_pesos(self) -> Path:
        import torchxrayvision as xrv

        return Path(self.config.diretorio_pesos or xrv.utils.get_cache_dir()).expanduser()

    @property
    def classificador(self):
        if self._classificador is None:
            import torchxrayvision as xrv

            url = xrv.models.model_urls[self.config.modelo]["weights_url"]
            modelo = carregar_com_pesos(
                url, self.pasta_pesos,
                lambda: xrv.models.DenseNet(weights=self.config.modelo, cache_dir=str(self.pasta_pesos)))
            self._classificador = modelo.to(self.dispositivo).eval()
        return self._classificador

    @property
    def segmentador(self):
        if self._segmentador is None:
            import torchxrayvision as xrv

            modelo = carregar_com_pesos(
                URL_PESOS_PSPNET, self.pasta_pesos,
                lambda: xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(self.pasta_pesos)))
            self._segmentador = modelo.to(self.dispositivo).eval()
        return self._segmentador

    def carregar(self, segmentacao: bool = True) -> None:
        _ = self.classificador
        if segmentacao:
            _ = self.segmentador

    def classificar(self, entrada: np.ndarray, refinar: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as F
        import torchxrayvision as xrv

        modelo = self.classificador
        x = entrada.astype(np.float32) * 2048.0 - 1024.0  # escala esperada: [-1024, 1024]
        deslocamentos = DESLOCAMENTOS_REFINO if refinar else ((0, 0),)
        lote = np.stack([transladar(x, dy, dx, -1024.0) for dy, dx in deslocamentos])[:, None]

        with torch.no_grad():
            t = torch.from_numpy(lote).to(self.dispositivo)
            ativacoes = F.relu(modelo.features(t))
            logits = modelo.classifier(F.adaptive_avg_pool2d(ativacoes, 1).flatten(1))
            # Mapa de ativação de classe (CAM): exato para GAP + camada linear.
            cams = torch.einsum("kc,bchw->bkhw", modelo.classifier.weight, ativacoes)
            cams = F.interpolate(cams, size=x.shape, mode="bilinear", align_corners=False)

            i0 = deslocamentos.index((0, 0))
            escores = torch.sigmoid(logits[i0:i0 + 1])
            limiares = getattr(modelo, "op_threshs", None)
            if limiares is not None:
                escores = xrv.models.op_norm(escores, limiares.to(escores.device))
        escores = escores[0].cpu().numpy()
        cams = cams.cpu().numpy()

        soma = np.zeros(cams.shape[1:], dtype=np.float64)
        contagem = np.zeros(cams.shape[2:], dtype=np.float64)
        for b, (dy, dx) in enumerate(deslocamentos):
            soma += transladar(cams[b], -dy, -dx, 0.0)
            contagem += transladar(np.ones(cams.shape[2:]), -dy, -dx, 0.0)
        mapas = (soma / np.maximum(contagem, 1.0)).astype(np.float32)
        return list(modelo.pathologies), escores, mapas

    def segmentar(self, entrada: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        modelo = self.segmentador
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*upsample.*deprecated", category=UserWarning)
            t = torch.from_numpy(entrada.astype(np.float32))[None, None].to(self.dispositivo)
            t = modelo.transform(t.repeat(1, 3, 1, 1))  # normalização ImageNet, entrada em [0, 1]
            saida = torch.sigmoid(modelo.model(t))[0].cpu().numpy()
        return {ESTRUTURAS_PSPNET[nome]: saida[i] for i, nome in enumerate(modelo.targets)
                if nome in ESTRUTURAS_PSPNET}

    def descricao(self) -> dict:
        import torchxrayvision as xrv

        return {
            "classificador": f"TorchXRayVision DenseNet121 ({self.config.modelo})",
            "segmentacao": "ChestX-Det PSPNet (TorchXRayVision)" if self.config.usar_segmentacao else None,
            "versao_torchxrayvision": getattr(xrv, "__version__", "?"),
            "dispositivo": str(self.dispositivo),
            "refinamento_localizacao": self.config.refinar_localizacao,
        }


# --------------------------------------------------------------------------- #
# Regras
# --------------------------------------------------------------------------- #

def classificar_status(escore: float, config: Configuracao) -> str:
    if escore >= config.limiar_positivo:
        return POSITIVO
    if escore >= config.limiar_indeterminado:
        return INDETERMINADO
    return NEGATIVO


def _sobrepoe(a: Achado, b: Achado, forma: tuple[int, int]) -> bool:
    if not a.regioes or not b.regioes:
        return True
    ma = np.zeros(forma, dtype=bool)
    mb = np.zeros(forma, dtype=bool)
    for r in a.regioes:
        ma |= mascara_da_regiao(r, forma)
    for r in b.regioes:
        mb |= mascara_da_regiao(r, forma)
    return ma.any() and (ma & mb).sum() >= 0.2 * ma.sum()


def aplicar_supressao(achados: list[Achado], forma: tuple[int, int]) -> None:
    """Marca como suprimidos os achados genéricos redundantes com achados específicos."""
    por_chave = {a.chave: a for a in achados}
    for a in achados:
        if a.status == NEGATIVO:
            continue
        for chave in patologia(a.chave).suprimido_por:
            b = por_chave.get(chave)
            if b is not None and b.status == POSITIVO and _sobrepoe(a, b, forma):
                a.suprimido = True
                break


# --------------------------------------------------------------------------- #
# Analisador
# --------------------------------------------------------------------------- #

CORES_NUVEM = ("#FF5D73", "#FFB547", "#4FC3F7", "#9CCC65", "#BA68C8", "#26C6DA", "#FF8A65", "#FFEE58")
_GRAVIDADE_NUVEM = {"urgente": 3, "relevante": 2, "leve": 1}
_ORDEM_CONFIANCA = {"alta": 0, "moderada": 1, "baixa": 2}


class Analisador:
    """Analisa radiografias e produz um :class:`ResultadoAnalise`.

    Tórax: modelos locais (classificação, mapas de ativação e segmentação).
    Demais regiões: IA multimodal em nuvem, quando habilitada na configuração.
    """

    def __init__(self, config: Configuracao | None = None, modelos: Modelos | None = None,
                 motor_nuvem=None):
        self.config = config or Configuracao()
        self._modelos = modelos
        self._motor_nuvem = motor_nuvem
        self._trava = threading.Lock()

    @property
    def modelos(self) -> Modelos:
        if self._modelos is None:
            self._modelos = ModelosTorchXRayVision(self.config)
        return self._modelos

    def motor_nuvem(self, config: Configuracao) -> MotorNuvem:
        return self._motor_nuvem or MotorNuvem(config)

    def carregar(self) -> None:
        """Carrega (e baixa, se necessário) os pesos antecipadamente."""
        modelos = self.modelos
        if hasattr(modelos, "carregar"):
            modelos.carregar(self.config.usar_segmentacao)

    # ------------------------------------------------------------------ #
    def analisar(self, exame: Exame, config: Configuracao | None = None) -> ResultadoAnalise:
        cfg = config or self.config
        inicio = time.perf_counter()
        meta = exame.metadados.anonimizado() if cfg.anonimizar else exame.metadados
        avisos = list(exame.avisos)

        aceita, motivo = modalidade_aceita(meta)
        if not aceita:
            if not cfg.forcar:
                raise ExameIncompativel(motivo, acao="forcar")
            avisos.append(f"{motivo} Análise forçada pelo usuário: resultados sem validade.")

        imagem, escala = redimensionar_para_exibicao(exame.imagem, cfg.tamanho_exibicao)
        geo = Geometria(*imagem.shape)
        invertida = bool(meta.orientacao) and meta.orientacao[0].upper().startswith("R")
        contexto = _Contexto(exame, meta, cfg, imagem, escala, geo, invertida, avisos, inicio)

        if cfg.regiao:
            if cfg.regiao not in REGIOES:
                raise ValueError(f"Região desconhecida: {cfg.regiao}. Opções: {', '.join(REGIOES)}")
            chave, fonte = cfg.regiao, "informada pelo usuário"
        else:
            chave, fonte = identificar_regiao(meta)

        if chave is None:
            anatomia, falha = self._segmentar(contexto)
            if anatomia is not None and anatomia.parece_torax():
                avisos.append("Região anatômica não informada; identificada como tórax pela "
                              "segmentação anatômica.")
                return self._analisar_torax(contexto, anatomia)
            if cfg.usar_nuvem:
                return self._analisar_nuvem(contexto, None, None)
            if anatomia is None:
                avisos.append("Região anatômica não informada; a imagem foi analisada como "
                              "radiografia de tórax.")
                if falha:
                    avisos.append(falha)
                return self._analisar_torax(contexto, None, segmentacao_falhou=bool(falha))
            raise ExameIncompativel(
                "Região anatômica não identificada e a imagem não parece uma radiografia de tórax. "
                "Selecione a região manualmente ou habilite a IA em nuvem (Configurações) para "
                "interpretar radiografias de outras regiões.",
                acao="configurar_nuvem",
            )

        if chave == "torax":
            return self._analisar_torax(contexto, None)
        if cfg.usar_nuvem:
            return self._analisar_nuvem(contexto, chave, fonte)

        descricao = REGIOES[chave].nome + (f" ({meta.regiao})" if meta.regiao else "")
        if cfg.forcar:
            avisos.append(f"Região \"{descricao}\" analisada com o modelo de tórax por solicitação do "
                          "usuário: resultados sem validade.")
            return self._analisar_torax(contexto, None)
        raise ExameIncompativel(
            f"Região \"{descricao}\": os modelos locais interpretam apenas radiografias de tórax. "
            "As demais regiões são interpretadas pela IA em nuvem — habilite-a em Configurações "
            "(requer chave de API da Anthropic).",
            acao="configurar_nuvem",
        )

    # ------------------------------------------------------------------ #
    def _segmentar(self, c: "_Contexto") -> tuple[Anatomia | None, str | None]:
        if not c.cfg.usar_segmentacao:
            return None, None
        try:
            with self._trava:
                probabilidades = self.modelos.segmentar(c.geo.quadrado(c.imagem, 512))
        except Exception as exc:  # ex.: falha ao baixar os pesos
            return None, (f"Segmentação anatômica indisponível ({exc}); lateralidade e zonas "
                          "estimadas pela geometria da imagem.")
        mascaras = {nome: c.geo.para_exibicao(p) > 0.5 for nome, p in probabilidades.items()}
        return Anatomia(c.geo.forma, mascaras, invertida=c.invertida), None

    def _analisar_torax(self, c: "_Contexto", anatomia: Anatomia | None,
                        segmentacao_falhou: bool = False) -> ResultadoAnalise:
        cfg, meta, avisos, geo = c.cfg, c.meta, c.avisos, c.geo
        if meta.incidencia == "PERFIL":
            avisos.append(
                "Incidência em perfil: os modelos foram treinados com incidências frontais "
                "(PA/AP); os resultados não são confiáveis."
            )
        if meta.idade_anos is not None and meta.idade_anos < 16:
            avisos.append(
                "Paciente pediátrico: os modelos foram treinados predominantemente com "
                "exames de adultos; interpretar com cautela."
            )

        with self._trava:
            nomes, escores, mapas = self.modelos.classificar(geo.quadrado(c.imagem, 224),
                                                             cfg.refinar_localizacao)
        if anatomia is None and not segmentacao_falhou:
            anatomia, falha = self._segmentar(c)
            if falha:
                avisos.append(falha)
        if anatomia is None:
            anatomia = Anatomia(geo.forma, invertida=c.invertida)

        avisos.extend(anatomia.avisos_qualidade())
        ict = anatomia.indice_cardiotoracico(meta.espacamento_mm, c.escala)
        if ict is not None and meta.incidencia == "AP":
            avisos.append("Incidência AP: a área cardíaca pode estar magnificada e o ICT superestimado.")

        achados: list[Achado] = []
        for i, chave in enumerate(nomes):
            if not chave:
                continue  # saída não treinada para estes pesos
            definicao = patologia(chave)
            escore = float(escores[i])
            if not np.isfinite(escore):
                continue
            achado = Achado(chave, definicao.nome, escore, classificar_status(escore, cfg),
                            definicao.cor, definicao.gravidade)
            if achado.relevante:
                self._localizar(achado, mapas[i], geo, anatomia)
            achados.append(achado)
        aplicar_supressao(achados, geo.forma)
        achados.sort(key=lambda a: -a.escore)

        descricao_modelo = self.modelos.descricao() if hasattr(self.modelos, "descricao") else {}
        if not anatomia.segmentada:
            descricao_modelo = {**descricao_modelo, "segmentacao": None}
        return self._resultado(c, achados, descricao_modelo, ict=ict,
                               contornos=anatomia.contornos(), regiao="torax", regiao_nome="Tórax",
                               motor=MOTOR_LOCAL)

    def _analisar_nuvem(self, c: "_Contexto", chave: str | None, fonte: str | None) -> ResultadoAnalise:
        motor = self.motor_nuvem(c.cfg)
        regiao_informada = REGIOES.get(chave) if chave else None
        interpretacao = motor.interpretar(c.exame.imagem, c.meta, regiao_informada, fonte)

        chave_final = chave or interpretacao.regiao
        if chave and interpretacao.regiao != chave:
            c.avisos.append(
                f"A IA identificou a região como \"{REGIOES[interpretacao.regiao].nome}\", diferente "
                f"da informada (\"{regiao_informada.nome}\")."
            )
        if not interpretacao.e_radiografia:
            c.avisos.append("A IA indicou que a imagem pode não ser uma radiografia; resultados sem validade.")
        c.avisos.append(
            "Interpretação por IA multimodal em nuvem: a localização dos achados é aproximada e o "
            "laudo exige revisão médica criteriosa."
        )
        achados = achados_da_nuvem(interpretacao, c.geo.forma)
        lado = interpretacao.lado_exame if interpretacao.lado_exame != "nao_aplicavel" else c.meta.lateralidade
        return self._resultado(c, achados, motor.descricao(), regiao=chave_final,
                               regiao_nome=interpretacao.regiao_descricao or REGIOES[chave_final].nome,
                               motor=MOTOR_NUVEM, lado=lado,
                               interpretacao=interpretacao.model_dump())

    @staticmethod
    def _resultado(c: "_Contexto", achados, modelo, ict=None, contornos=None, regiao="torax",
                   regiao_nome="Tórax", motor=MOTOR_LOCAL, lado=None, interpretacao=None) -> ResultadoAnalise:
        return ResultadoAnalise(
            metadados=c.meta,
            nome_arquivo=c.exame.nome_arquivo,
            imagem=np.round(c.imagem * 255).astype(np.uint8),
            escala=c.escala,
            achados=achados,
            config=c.cfg,
            ict=ict,
            contornos_anatomicos=contornos or {},
            avisos=c.avisos,
            modelo=modelo,
            data_analise=datetime.now().strftime("%d/%m/%Y %H:%M"),
            tempo_s=time.perf_counter() - c.inicio,
            regiao=regiao,
            regiao_nome=regiao_nome,
            lado_exame=lado or c.meta.lateralidade,
            motor=motor,
            interpretacao=interpretacao,
        )

    @staticmethod
    def _localizar(achado: Achado, mapa_modelo: np.ndarray, geo: Geometria, anatomia: Anatomia) -> None:
        definicao = patologia(achado.chave)
        cam = geo.para_exibicao(mapa_modelo)
        coracao = anatomia.mascara("coracao")
        if achado.chave == "Cardiomegaly" and anatomia.segmentada and coracao.any():
            # A silhueta cardíaca segmentada descreve o achado melhor que o CAM.
            mapa = normalizar_mapa(cam, anatomia.prior(REGIAO_CORACAO))
            regiao = regiao_de_mascara(coracao)
            regioes = [regiao] if regiao else []
        else:
            mapa = normalizar_mapa(cam, anatomia.prior(definicao.regiao))
            regioes = extrair_regioes(mapa)
        for regiao in regioes:
            regiao.lado, regiao.zonas = anatomia.lado_e_zonas(mascara_da_regiao(regiao, geo.forma))
        achado.regioes = regioes
        achado.local = descrever_local(regioes, definicao.tipo_local)
        achado.mapa = np.round(mapa * 255).astype(np.uint8)


#: Nome mantido por compatibilidade com a versão 0.1.
AnalisadorTorax = Analisador


@dataclass
class _Contexto:
    exame: Exame
    meta: object
    cfg: Configuracao
    imagem: np.ndarray
    escala: float
    geo: Geometria
    invertida: bool
    avisos: list[str]
    inicio: float


def achados_da_nuvem(interpretacao: InterpretacaoIA, forma: tuple[int, int]) -> list[Achado]:
    """Converte os achados da IA em nuvem (caixas normalizadas 0–1000) em :class:`Achado`."""
    altura, largura = forma
    achados: list[Achado] = []
    for i, item in enumerate(interpretacao.achados):
        regioes = []
        for caixa in item.caixas:
            x0, x1 = sorted((caixa.x_min, caixa.x_max))
            y0, y1 = sorted((caixa.y_min, caixa.y_max))
            x0, x1 = (min(largura - 1, round(np.clip(v, 0, 1000) / 1000 * largura)) for v in (x0, x1))
            y0, y1 = (min(altura - 1, round(np.clip(v, 0, 1000) / 1000 * altura)) for v in (y0, y1))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            regioes.append(Regiao(
                contorno=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                caixa=(x0, y0, x1 - x0, y1 - y0),
                area=(x1 - x0) * (y1 - y0),
                pico=1.0,
                centroide=((x0 + x1) / 2, (y0 + y1) / 2),
                lado=item.lado if item.lado in ("direito", "esquerdo") else None,
            ))
        local = item.localizacao.strip()
        if item.lado in ("direito", "esquerdo", "bilateral") and item.lado not in local.lower():
            local = f"{local} ({item.lado})" if local else item.lado
        gravidade = _GRAVIDADE_NUVEM.get(item.gravidade, 2)
        achados.append(Achado(
            chave=f"ia_{i + 1}",
            nome=item.nome.strip() or f"Achado {i + 1}",
            escore=None,
            status=INDETERMINADO if item.confianca == "baixa" else POSITIVO,
            cor="#FF1744" if gravidade == 3 else CORES_NUVEM[i % len(CORES_NUVEM)],
            gravidade=gravidade,
            regioes=regioes,
            local=local,
            confianca=item.confianca,
            descricao=item.descricao.strip(),
            frase_impressao=item.impressao.strip(),
            recomendacao=item.recomendacao.strip(),
        ))
    achados.sort(key=lambda a: (-a.gravidade, _ORDEM_CONFIANCA.get(a.confianca, 3)))
    return achados

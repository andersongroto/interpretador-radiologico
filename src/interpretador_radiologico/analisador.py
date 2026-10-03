"""Motor de análise: classificação, localização e referências anatômicas."""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import cv2
import numpy as np

from .anatomia import ESTRUTURAS_PSPNET, Anatomia
from .config import Configuracao
from .dicom_io import Exame, verificar_compatibilidade
from .localizacao import (descrever_local, extrair_regioes, mascara_da_regiao,
                          normalizar_mapa, regiao_de_mascara)
from .patologias import REGIAO_CORACAO, patologia
from .resultado import INDETERMINADO, NEGATIVO, POSITIVO, Achado, ResultadoAnalise

# Deslocamentos (em pixels na entrada 224x224) usados para refinar os mapas de
# ativação: a rede tem passo de 32 px, então deslocar meia célula e calcular a
# média dos mapas realinhados dobra a resolução efetiva da localização.
DESLOCAMENTOS_REFINO = tuple((dy, dx) for dy in (-16, 0, 16) for dx in (-16, 0, 16))


class ExameIncompativel(Exception):
    """O exame não é uma radiografia de tórax suportada pelo modelo."""


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
    def classificador(self):
        if self._classificador is None:
            import torchxrayvision as xrv

            modelo = xrv.models.DenseNet(weights=self.config.modelo, cache_dir=self.config.diretorio_pesos)
            self._classificador = modelo.to(self.dispositivo).eval()
        return self._classificador

    @property
    def segmentador(self):
        if self._segmentador is None:
            import torchxrayvision as xrv

            modelo = xrv.baseline_models.chestx_det.PSPNet(cache_dir=self.config.diretorio_pesos)
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

class AnalisadorTorax:
    """Analisa radiografias de tórax e produz um :class:`ResultadoAnalise`."""

    def __init__(self, config: Configuracao | None = None, modelos: Modelos | None = None):
        self.config = config or Configuracao()
        self._modelos = modelos
        self._trava = threading.Lock()

    @property
    def modelos(self) -> Modelos:
        if self._modelos is None:
            self._modelos = ModelosTorchXRayVision(self.config)
        return self._modelos

    def carregar(self) -> None:
        """Carrega (e baixa, se necessário) os pesos antecipadamente."""
        modelos = self.modelos
        if hasattr(modelos, "carregar"):
            modelos.carregar(self.config.usar_segmentacao)

    def analisar(self, exame: Exame, config: Configuracao | None = None) -> ResultadoAnalise:
        cfg = config or self.config
        inicio = time.perf_counter()
        meta = exame.metadados.anonimizado() if cfg.anonimizar else exame.metadados
        avisos = list(exame.avisos)

        compativel, motivo = verificar_compatibilidade(meta)
        if compativel is False:
            if not cfg.forcar:
                raise ExameIncompativel(motivo)
            avisos.append(f"{motivo} Análise forçada pelo usuário: resultados sem validade.")
        elif motivo:
            avisos.append(motivo)
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

        imagem, escala = redimensionar_para_exibicao(exame.imagem, cfg.tamanho_exibicao)
        forma = imagem.shape
        geo = Geometria(*forma)
        invertida = bool(meta.orientacao) and meta.orientacao[0].upper().startswith("R")

        with self._trava:
            nomes, escores, mapas = self.modelos.classificar(geo.quadrado(imagem, 224),
                                                             cfg.refinar_localizacao)
            anatomia = Anatomia(forma, invertida=invertida)
            if cfg.usar_segmentacao:
                try:
                    probabilidades = self.modelos.segmentar(geo.quadrado(imagem, 512))
                    mascaras = {nome: geo.para_exibicao(p) > 0.5 for nome, p in probabilidades.items()}
                    anatomia = Anatomia(forma, mascaras, invertida=invertida)
                except Exception as exc:  # ex.: falha ao baixar os pesos
                    avisos.append(
                        f"Segmentação anatômica indisponível ({exc}); lateralidade e zonas "
                        "estimadas pela geometria da imagem."
                    )

        avisos.extend(anatomia.avisos_qualidade())
        ict = anatomia.indice_cardiotoracico(meta.espacamento_mm, escala)
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
        aplicar_supressao(achados, forma)
        achados.sort(key=lambda a: -a.escore)

        descricao_modelo = self.modelos.descricao() if hasattr(self.modelos, "descricao") else {}
        if not anatomia.segmentada:
            descricao_modelo = {**descricao_modelo, "segmentacao": None}
        return ResultadoAnalise(
            metadados=meta,
            nome_arquivo=exame.nome_arquivo,
            imagem=np.round(imagem * 255).astype(np.uint8),
            escala=escala,
            achados=achados,
            config=cfg,
            ict=ict,
            contornos_anatomicos=anatomia.contornos(),
            avisos=avisos,
            modelo=descricao_modelo,
            data_analise=datetime.now().strftime("%d/%m/%Y %H:%M"),
            tempo_s=time.perf_counter() - inicio,
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

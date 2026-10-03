"""Parâmetros de configuração da análise."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Configuracao:
    """Parâmetros que controlam a análise e a redação do laudo.

    Os escores do classificador são normalizados pelo ponto de operação de cada
    patologia: 0,5 corresponde ao limiar ótimo definido no treinamento. Valores
    um pouco acima de 0,5 são frequentes em exames normais, por isso o limiar
    de positividade padrão é mais conservador.
    """

    #: Escore a partir do qual o achado é considerado positivo.
    limiar_positivo: float = 0.60
    #: Escore a partir do qual o achado é listado como indeterminado.
    limiar_indeterminado: float = 0.55
    #: Pesos do classificador (TorchXRayVision, família DenseNet121).
    modelo: str = "densenet121-res224-all"
    #: Executa a segmentação anatômica (lateralidade, zonas e ICT).
    usar_segmentacao: bool = True
    #: Média de mapas de ativação com pequenos deslocamentos da imagem, o que
    #: melhora a resolução espacial da localização (9 inferências em lote).
    refinar_localizacao: bool = True
    #: "auto", "cpu" ou "cuda".
    dispositivo: str = "auto"
    #: Diretório de cache dos pesos (padrão do TorchXRayVision se None).
    diretorio_pesos: str | None = None
    #: Lado máximo da imagem de trabalho/exibição, em pixels.
    tamanho_exibicao: int = 1024
    #: Ignora a verificação de região anatômica/modalidade.
    forcar: bool = False
    #: Remove dados identificáveis do paciente das saídas.
    anonimizar: bool = False
    #: Nome do serviço exibido no cabeçalho do laudo (usa o do DICOM se None).
    nome_instituicao: str | None = None
    #: Região anatômica informada pelo usuário (chave de ``regioes.REGIOES``);
    #: None = identificação automática.
    regiao: str | None = None
    #: Habilita o motor de IA em nuvem (Claude) para regiões sem modelo local.
    #: Envia à API da Anthropic apenas os pixels da imagem (sem metadados DICOM).
    usar_nuvem: bool = False
    #: Chave da API da Anthropic (se None, usa ANTHROPIC_API_KEY do ambiente).
    chave_api: str | None = field(default=None, repr=False)
    modelo_nuvem: str = "claude-opus-5-5"

    def __post_init__(self) -> None:
        if not 0.0 < self.limiar_indeterminado <= self.limiar_positivo < 1.0:
            raise ValueError(
                "Os limiares devem satisfazer 0 < indeterminado <= positivo < 1 "
                f"(recebido: indeterminado={self.limiar_indeterminado}, "
                f"positivo={self.limiar_positivo})."
            )

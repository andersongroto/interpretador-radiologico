"""Catálogo das patologias avaliadas e das frases usadas no laudo.

As chaves correspondem aos rótulos de saída dos modelos TorchXRayVision.
Os textos usam o marcador ``{local}``, substituído pela descrição da
localização (ex.: "no terço inferior do pulmão direito", "à esquerda").
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sistemas anatômicos, na ordem em que aparecem na seção de análise do laudo.
SISTEMAS: tuple[tuple[str, str], ...] = (
    ("pulmoes", "Pulmões"),
    ("pleura", "Pleuras"),
    ("coracao", "Coração"),
    ("mediastino", "Mediastino"),
    ("diafragma", "Diafragma"),
    ("ossos", "Estruturas ósseas"),
)

# Tipos de descrição de localização.
LOCAL_PULMONAR = "pulmonar"  # lado + terço ("no terço inferior do pulmão direito")
LOCAL_HEMITORAX = "hemitorax"  # "no hemitórax direito"
LOCAL_LADO = "lado"  # "à direita", "à esquerda", "bilateral"
LOCAL_NENHUM = "nenhum"

# Regiões anatômicas usadas para restringir os mapas de ativação.
REGIAO_PULMAO = "pulmao"
REGIAO_TORAX = "torax"
REGIAO_CORACAO = "coracao"
REGIAO_MEDIASTINO = "mediastino"
REGIAO_DIAFRAGMA = "diafragma"


@dataclass(frozen=True)
class Patologia:
    chave: str
    nome: str
    sistema: str
    regiao: str
    tipo_local: str
    frase_achado: str
    frase_impressao: str
    gravidade: int  # 1 = baixa, 2 = relevante, 3 = potencialmente urgente
    cor: str  # hexadecimal, usada nas marcações da imagem
    recomendacao: str | None = None
    # Achados mais específicos que, quando positivos na mesma região, tornam
    # este achado redundante no laudo (ex.: "opacidade" frente a "consolidação").
    suprimido_por: tuple[str, ...] = field(default_factory=tuple)


_REC_TC = "Recomenda-se tomografia computadorizada de tórax para melhor caracterização."

CATALOGO: dict[str, Patologia] = {
    p.chave: p
    for p in (
        Patologia(
            "Atelectasis", "Atelectasia", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidade com aspecto de redução volumétrica {local}, sugestiva de atelectasia.",
            "Atelectasia {local}.",
            2, "#4FC3F7",
        ),
        Patologia(
            "Consolidation", "Consolidação", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidade de padrão alveolar (consolidação) {local}.",
            "Consolidação {local} — considerar processo infeccioso/inflamatório.",
            2, "#FF7043",
            recomendacao="Correlacionar com quadro clínico e exames laboratoriais.",
        ),
        Patologia(
            "Infiltration", "Infiltrado pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidades de aspecto infiltrativo {local}.",
            "Infiltrado pulmonar {local}.",
            2, "#FFCA28",
            suprimido_por=("Pneumonia", "Consolidation", "Edema"),
        ),
        Patologia(
            "Pneumothorax", "Pneumotórax", "pleura", REGIAO_TORAX, LOCAL_LADO,
            "Sinais sugestivos de pneumotórax {local}.",
            "Pneumotórax {local} — achado potencialmente urgente.",
            3, "#FF1744",
            recomendacao=(
                "Achado potencialmente urgente: comunicar a equipe assistente e "
                "confirmar com avaliação médica imediata."
            ),
        ),
        Patologia(
            "Edema", "Edema pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidades intersticiais e/ou alveolares {local}, sugestivas de edema pulmonar.",
            "Sinais de edema pulmonar {local}.",
            2, "#26C6DA",
            recomendacao="Correlacionar com função cardíaca e balanço hídrico.",
        ),
        Patologia(
            "Emphysema", "Enfisema", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Sinais de hiperinsuflação pulmonar {local}, sugestivos de enfisema.",
            "Sinais sugestivos de enfisema pulmonar.",
            1, "#9CCC65",
            recomendacao="Considerar correlação com espirometria e/ou tomografia de tórax.",
        ),
        Patologia(
            "Fibrosis", "Fibrose pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidades reticulares {local}, sugestivas de alterações fibróticas.",
            "Alterações fibróticas {local}.",
            1, "#AB47BC",
            recomendacao="Considerar tomografia de tórax de alta resolução para caracterização.",
        ),
        Patologia(
            "Effusion", "Derrame pleural", "pleura", REGIAO_TORAX, LOCAL_LADO,
            "Velamento de seio costofrênico {local}, sugestivo de derrame pleural.",
            "Derrame pleural {local}.",
            2, "#42A5F5",
            recomendacao="Considerar ultrassonografia de tórax para quantificação e correlação clínica.",
        ),
        Patologia(
            "Pneumonia", "Pneumonia", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidade {local} com aspecto sugestivo de processo pneumônico.",
            "Achados sugestivos de pneumonia {local}.",
            2, "#EF5350",
            recomendacao=(
                "Correlacionar com quadro clínico e laboratorial; considerar controle "
                "radiológico após o tratamento."
            ),
        ),
        Patologia(
            "Pleural_Thickening", "Espessamento pleural", "pleura", REGIAO_TORAX, LOCAL_LADO,
            "Espessamento pleural {local}.",
            "Espessamento pleural {local}.",
            1, "#A1887F",
        ),
        Patologia(
            "Cardiomegaly", "Cardiomegalia", "coracao", REGIAO_CORACAO, LOCAL_NENHUM,
            "Aumento da área cardíaca{ict}.",
            "Cardiomegalia.",
            2, "#EC407A",
            recomendacao="Considerar ecocardiograma para avaliação estrutural e funcional.",
        ),
        Patologia(
            "Nodule", "Nódulo pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Imagem nodular {local}.",
            "Nódulo pulmonar {local}.",
            2, "#FFEE58",
            recomendacao=_REC_TC,
        ),
        Patologia(
            "Mass", "Massa pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Lesão expansiva (massa) {local}.",
            "Massa pulmonar {local} — requer investigação.",
            3, "#FF9100",
            recomendacao=(
                "Recomenda-se tomografia computadorizada de tórax com contraste e "
                "avaliação especializada."
            ),
        ),
        Patologia(
            "Hernia", "Hérnia hiatal/diafragmática", "diafragma", REGIAO_DIAFRAGMA, LOCAL_NENHUM,
            "Imagem sugestiva de hérnia hiatal/diafragmática.",
            "Possível hérnia hiatal/diafragmática.",
            1, "#90A4AE",
            recomendacao="Considerar estudo contrastado ou tomografia para confirmação.",
        ),
        Patologia(
            "Lung Lesion", "Lesão pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Lesão pulmonar {local}.",
            "Lesão pulmonar {local}.",
            2, "#D4E157",
            recomendacao=_REC_TC,
            suprimido_por=("Nodule", "Mass"),
        ),
        Patologia(
            "Fracture", "Fratura", "ossos", REGIAO_TORAX, LOCAL_HEMITORAX,
            "Sinais sugestivos de fratura de arco(s) costal(is) {local}.",
            "Possível fratura de arco(s) costal(is) {local}.",
            2, "#00E676",
            recomendacao="Correlacionar com história de trauma; considerar incidências oblíquas ou tomografia.",
        ),
        Patologia(
            "Lung Opacity", "Opacidade pulmonar", "pulmoes", REGIAO_PULMAO, LOCAL_PULMONAR,
            "Opacidade pulmonar {local}.",
            "Opacidade pulmonar {local}.",
            2, "#FFA726",
            recomendacao="Correlacionar com dados clínicos.",
            suprimido_por=(
                "Atelectasis", "Consolidation", "Pneumonia", "Edema", "Mass",
                "Nodule", "Lung Lesion", "Infiltration", "Fibrosis", "Effusion",
            ),
        ),
        Patologia(
            "Enlarged Cardiomediastinum", "Alargamento cardiomediastinal", "mediastino",
            REGIAO_MEDIASTINO, LOCAL_NENHUM,
            "Alargamento da silhueta cardiomediastinal.",
            "Alargamento da silhueta cardiomediastinal.",
            2, "#7E57C2",
            recomendacao=(
                "Considerar tomografia de tórax se não houver explicação técnica "
                "(incidência AP, rotação, inspiração insuficiente)."
            ),
            suprimido_por=("Cardiomegaly",),
        ),
    )
}

# Frases de normalidade: usadas quando TODAS as patologias do grupo são negativas.
FRASES_NORMAIS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("Atelectasis", "Consolidation", "Infiltration", "Pneumonia", "Lung Opacity",
         "Edema", "Fibrosis", "Nodule", "Mass", "Lung Lesion"),
        "pulmoes",
        "Campos pulmonares com transparência preservada, sem opacidades focais ou difusas identificadas.",
    ),
    (("Emphysema",), "pulmoes", "Sem sinais de hiperinsuflação pulmonar."),
    (("Effusion",), "pleura", "Seios costofrênicos livres."),
    (("Pneumothorax",), "pleura", "Ausência de sinais de pneumotórax."),
    (("Pleural_Thickening",), "pleura", "Pleuras sem espessamentos evidentes."),
    (("Cardiomegaly",), "coracao", "Área cardíaca dentro dos limites da normalidade{ict}."),
    (("Enlarged Cardiomediastinum",), "mediastino", "Mediastino de contornos e dimensões habituais."),
    (("Hernia",), "diafragma", "Sem sinais de hérnia diafragmática ou hiatal."),
    (("Fracture",), "ossos", "Arcabouço ósseo sem sinais evidentes de fratura."),
)


def patologia(chave: str) -> Patologia:
    """Retorna a definição da patologia, ou uma definição genérica se desconhecida."""
    if chave in CATALOGO:
        return CATALOGO[chave]
    nome = chave.replace("_", " ")
    return Patologia(
        chave, nome, "pulmoes", REGIAO_TORAX, LOCAL_PULMONAR,
        nome + " {local}.", nome + " {local}.", 1, "#BDBDBD",
    )


def cor_bgr(cor_hex: str) -> tuple[int, int, int]:
    """Converte "#RRGGBB" para a tupla BGR usada pelo OpenCV."""
    cor_hex = cor_hex.lstrip("#")
    r, g, b = (int(cor_hex[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)

"""Regiões anatômicas radiográficas: identificação e roteamento para o motor de análise."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .dicom_io import Metadados

MOTOR_LOCAL = "local"  # modelos locais (tórax)
MOTOR_NUVEM = "nuvem"  # IA multimodal em nuvem (demais regiões)

# Modalidades de imagem radiográfica (projeção) aceitas.
MODALIDADES_RADIOGRAFICAS = {"CR", "DX", "DR", "RG", "OT", "MG", "PX", "IO", "RF"}


@dataclass(frozen=True)
class Regiao:
    chave: str
    nome: str  # "Joelho"
    titulo: str  # "RADIOGRAFIA DO JOELHO"
    feminino: bool = False  # concordância de "direito/direita"
    lateral: bool = False  # região par (tem lado)
    estruturas: tuple[str, ...] = ()  # roteiro de análise para a IA
    palavras: tuple[str, ...] = ()  # termos (sem acento, maiúsculos) do DICOM

    @property
    def motor(self) -> str:
        return MOTOR_LOCAL if self.chave == "torax" else MOTOR_NUVEM

    def titulo_com_lado(self, lado: str | None) -> str:
        if not (self.lateral and lado in ("direito", "esquerdo", "bilateral")):
            return self.titulo
        if lado == "bilateral":
            return f"{self.titulo} (BILATERAL)"
        sufixo = lado.upper() if not self.feminino else lado[:-1].upper() + "A"
        return f"{self.titulo} {sufixo}"


_OSSOS_PARTES_MOLES = ("Partes moles",)

REGIOES: dict[str, Regiao] = {
    r.chave: r
    for r in (
        Regiao("torax", "Tórax", "RADIOGRAFIA DE TÓRAX",
               estruturas=("Pulmões", "Pleuras", "Coração", "Mediastino", "Diafragma", "Estruturas ósseas"),
               palavras=("TORAX", "THORAX", "CHEST", "PULMAO", "PULMOES", "PULMONAR", "LUNG", "LUNGS",
                         "PEITO", "CXR", "RXT", "TX", "CHESTABDOMEN")),
        Regiao("costelas", "Arcos costais", "RADIOGRAFIA DE ARCOS COSTAIS", lateral=True,
               estruturas=("Arcos costais", "Pleuras e pulmões (avaliação limitada)", *_OSSOS_PARTES_MOLES),
               palavras=("RIB", "RIBS", "COSTELA", "COSTELAS", "COSTAL", "COSTAIS", "GRADIL")),
        Regiao("cranio", "Crânio", "RADIOGRAFIA DE CRÂNIO",
               estruturas=("Calota craniana", "Base do crânio e sela túrcica", "Suturas", *_OSSOS_PARTES_MOLES),
               palavras=("SKULL", "CRANIO", "CRANEO", "HEAD", "CABECA", "SELA")),
        Regiao("face", "Face e seios da face", "RADIOGRAFIA DE SEIOS DA FACE",
               estruturas=("Seios paranasais", "Órbitas", "Ossos da face", "Septo e cavidade nasal",
                           *_OSSOS_PARTES_MOLES),
               palavras=("SINUS", "SINUSES", "SEIOS", "FACE", "ORBITA", "ORBITAS", "NASAL", "NARIZ",
                         "MANDIBULA", "JAW", "TMJ", "ATM", "ZIGOMA", "WATERS", "CALDWELL")),
        Regiao("dental", "Arcada dentária", "RADIOGRAFIA ODONTOLÓGICA",
               estruturas=("Dentes e restaurações", "Periápices e periodonto", "Osso alveolar",
                           "Mandíbula e maxila", "Articulações temporomandibulares e seios maxilares"),
               palavras=("DENTAL", "DENTE", "DENTES", "PANORAMICA", "PANORAMIC", "PERIAPICAL",
                         "INTERPROXIMAL", "BITEWING", "ODONTO", "ODONTOLOGICA")),
        Regiao("coluna_cervical", "Coluna cervical", "RADIOGRAFIA DA COLUNA CERVICAL",
               estruturas=("Alinhamento e lordose", "Corpos vertebrais", "Espaços discais",
                           "Articulações interapofisárias e forames", "Transição craniocervical",
                           "Partes moles pré-vertebrais"),
               palavras=("CSPINE", "CERVICAL", "PESCOCO", "NECK")),
        Regiao("coluna_toracica", "Coluna torácica", "RADIOGRAFIA DA COLUNA TORÁCICA",
               estruturas=("Alinhamento e cifose", "Corpos vertebrais", "Espaços discais", "Pedículos",
                           "Arcos costais posteriores", *_OSSOS_PARTES_MOLES),
               palavras=("TSPINE", "DORSAL")),
        Regiao("coluna_lombar", "Coluna lombossacra", "RADIOGRAFIA DA COLUNA LOMBOSSACRA",
               estruturas=("Alinhamento e lordose", "Corpos vertebrais", "Espaços discais",
                           "Articulações interapofisárias", "Articulações sacroilíacas", *_OSSOS_PARTES_MOLES),
               palavras=("LSPINE", "LOMBAR", "LOMBOSSACRA", "LOMBOSSACRAL", "LUMBAR", "LUMBOSACRAL")),
        Regiao("sacro_coccix", "Sacro e cóccix", "RADIOGRAFIA DE SACRO E CÓCCIX",
               estruturas=("Sacro", "Cóccix", "Articulações sacroilíacas", *_OSSOS_PARTES_MOLES),
               palavras=("SACRUM", "SACRO", "COCCYX", "COCCIX", "SSPINE")),
        Regiao("coluna", "Coluna vertebral", "RADIOGRAFIA DA COLUNA VERTEBRAL",
               estruturas=("Alinhamento e curvaturas (escoliose)", "Corpos vertebrais", "Espaços discais",
                           "Bacia e cristas ilíacas", *_OSSOS_PARTES_MOLES),
               palavras=("SPINE", "COLUNA", "ESCOLIOSE", "PANORAMICA_COLUNA", "WHOLESPINE")),
        Regiao("ombro", "Ombro", "RADIOGRAFIA DO OMBRO", lateral=True,
               estruturas=("Articulação glenoumeral", "Articulação acromioclavicular", "Úmero proximal",
                           "Escápula e clavícula", "Espaço subacromial", *_OSSOS_PARTES_MOLES),
               palavras=("SHOULDER", "OMBRO", "ESCAPULA", "SCAPULA", "CLAVICULA", "CLAVICLE",
                         "ACROMIOCLAVICULAR", "GLENOUMERAL")),
        Regiao("umero", "Braço (úmero)", "RADIOGRAFIA DO BRAÇO", lateral=True,
               estruturas=("Diáfise umeral", "Úmero proximal e distal", "Articulações adjacentes",
                           *_OSSOS_PARTES_MOLES),
               palavras=("HUMERUS", "UMERO", "BRACO", "ARM")),
        Regiao("cotovelo", "Cotovelo", "RADIOGRAFIA DO COTOVELO", lateral=True,
               estruturas=("Alinhamento articular", "Úmero distal", "Cabeça do rádio", "Olécrano e ulna proximal",
                           "Coxins gordurosos e partes moles"),
               palavras=("ELBOW", "COTOVELO")),
        Regiao("antebraco", "Antebraço", "RADIOGRAFIA DO ANTEBRAÇO", lateral=True,
               estruturas=("Rádio", "Ulna", "Articulações radioulnares e adjacentes", *_OSSOS_PARTES_MOLES),
               palavras=("FOREARM", "ANTEBRACO", "RADIO", "ULNA")),
        Regiao("punho", "Punho", "RADIOGRAFIA DO PUNHO", lateral=True,
               estruturas=("Rádio e ulna distais", "Ossos do carpo (escafoide)", "Alinhamento carpal",
                           "Espaços articulares", *_OSSOS_PARTES_MOLES),
               palavras=("WRIST", "PUNHO", "CARPO", "ESCAFOIDE", "SCAPHOID")),
        Regiao("mao", "Mão", "RADIOGRAFIA DA MÃO", feminino=True, lateral=True,
               estruturas=("Ossos do carpo", "Metacarpos", "Falanges", "Articulações interfalângicas e "
                           "metacarpofalângicas", "Mineralização óssea", *_OSSOS_PARTES_MOLES),
               palavras=("HAND", "MAO", "MAOS", "DEDO", "DEDOS", "FINGER", "THUMB", "POLEGAR",
                         "QUIRODACTILO", "QUIRODACTILOS", "IDADE_OSSEA")),
        Regiao("bacia", "Bacia", "RADIOGRAFIA DA BACIA", feminino=True,
               estruturas=("Ossos ilíacos, ísquios e púbis", "Articulações sacroilíacas", "Sínfise púbica",
                           "Articulações coxofemorais", "Fêmures proximais", *_OSSOS_PARTES_MOLES),
               palavras=("PELVIS", "BACIA", "PELVE")),
        Regiao("quadril", "Quadril", "RADIOGRAFIA DO QUADRIL", lateral=True,
               estruturas=("Articulação coxofemoral", "Cabeça e colo femoral", "Acetábulo", "Trocânteres",
                           *_OSSOS_PARTES_MOLES),
               palavras=("HIP", "QUADRIL", "COXOFEMORAL", "LOWENSTEIN")),
        Regiao("femur", "Coxa (fêmur)", "RADIOGRAFIA DA COXA", feminino=True, lateral=True,
               estruturas=("Diáfise femoral", "Fêmur proximal e distal", "Articulações adjacentes",
                           *_OSSOS_PARTES_MOLES),
               palavras=("FEMUR", "COXA", "THIGH")),
        Regiao("joelho", "Joelho", "RADIOGRAFIA DO JOELHO", lateral=True,
               estruturas=("Alinhamento articular", "Fêmur distal", "Tíbia proximal e fíbula proximal",
                           "Patela", "Espaços articulares femorotibiais e femoropatelar",
                           "Partes moles (derrame articular, calcificações)"),
               palavras=("KNEE", "JOELHO", "JOELHOS", "PATELA", "PATELLA", "ROTULA")),
        Regiao("perna", "Perna", "RADIOGRAFIA DA PERNA", feminino=True, lateral=True,
               estruturas=("Tíbia", "Fíbula", "Articulações adjacentes", *_OSSOS_PARTES_MOLES),
               palavras=("LEG", "PERNA", "TIBIA", "FIBULA", "TIBIAFIBULA")),
        Regiao("tornozelo", "Tornozelo", "RADIOGRAFIA DO TORNOZELO", lateral=True,
               estruturas=("Maléolos medial e lateral", "Tálus e mortalha articular", "Tíbia e fíbula distais",
                           "Calcâneo (visível)", *_OSSOS_PARTES_MOLES),
               palavras=("ANKLE", "TORNOZELO")),
        Regiao("pe", "Pé", "RADIOGRAFIA DO PÉ", lateral=True,
               estruturas=("Tarso", "Metatarsos", "Falanges", "Alinhamento (arcos plantares)",
                           "Articulações", *_OSSOS_PARTES_MOLES),
               palavras=("FOOT", "PE", "PES", "CALCANEO", "CALCANEUS", "TOE", "TOES", "ARTELHO",
                         "PODODACTILO", "PODODACTILOS", "HALUX")),
        Regiao("abdome", "Abdome", "RADIOGRAFIA DE ABDOME",
               estruturas=("Distribuição gasosa intestinal", "Níveis hidroaéreos e pneumoperitônio",
                           "Calcificações e corpos estranhos", "Contornos de vísceras e linhas do psoas",
                           "Estruturas ósseas visíveis"),
               palavras=("ABDOMEN", "ABDOME", "ABD", "ABDOMINAL", "KUB", "ABDOMEAGUDO")),
        Regiao("mama", "Mamas", "MAMOGRAFIA", feminino=True, lateral=True,
               estruturas=("Composição mamária", "Nódulos e massas", "Calcificações", "Distorções arquiteturais",
                           "Assimetrias", "Pele, complexo areolopapilar e linfonodos axilares",
                           "Classificação BI-RADS sugerida"),
               palavras=("BREAST", "MAMA", "MAMAS", "MAMOGRAFIA", "MAMMOGRAPHY", "MAMMO")),
        Regiao("outra", "Outra região", "RADIOGRAFIA",
               estruturas=("Estruturas ósseas", "Articulações", "Partes moles"),
               palavras=("EXTREMITY", "EXTREMIDADE", "WHOLEBODY", "CORPOINTEIRO")),
    )
}

# Ordem de verificação: termos mais específicos primeiro (ex.: "COLUNA TORÁCICA"
# não deve ser classificada como "tórax").
_ORDEM = ("mama", "dental", "coluna_cervical", "coluna_toracica", "coluna_lombar", "sacro_coccix",
          "costelas", "ombro", "umero", "cotovelo", "antebraco", "punho", "mao", "bacia", "quadril",
          "femur", "joelho", "perna", "tornozelo", "pe", "cranio", "face", "abdome", "coluna",
          "torax", "outra")


def _palavras(*textos: str | None) -> list[str]:
    saida: list[str] = []
    for texto in textos:
        if texto:
            sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()
            saida += [p for p in re.split(r"[^A-Z0-9_]+", sem_acento) if p]
    return saida


def _casar(palavras: list[str]) -> str | None:
    conjunto = set(palavras)
    if "COLUNA" in conjunto or "SPINE" in conjunto:
        if conjunto & {"TORACICA", "TORACICO", "DORSAL", "THORACIC"}:
            return "coluna_toracica"
        if conjunto & {"CERVICAL"}:
            return "coluna_cervical"
        if conjunto & {"LOMBAR", "LOMBOSSACRA", "LUMBAR"}:
            return "coluna_lombar"
    for chave in _ORDEM:
        if conjunto & set(REGIOES[chave].palavras):
            return chave
    return None


def identificar_regiao(meta: Metadados) -> tuple[str | None, str | None]:
    """Identifica a região pelo DICOM. Retorna (chave, fonte) ou (None, None)."""
    if meta.modalidade == "MG":
        return "mama", "modalidade MG"
    if meta.modalidade in ("PX", "IO"):
        return "dental", f"modalidade {meta.modalidade}"
    chave = _casar(_palavras(meta.regiao))
    if chave:
        return chave, "BodyPartExamined"
    chave = _casar(_palavras(meta.descricao_estudo, meta.descricao_serie))
    if chave:
        return chave, "descrição do estudo/série"
    return None, None


def modalidade_aceita(meta: Metadados) -> tuple[bool, str | None]:
    if meta.modalidade and meta.modalidade not in MODALIDADES_RADIOGRAFICAS:
        return False, (
            f"Modalidade {meta.modalidade} não suportada: o sistema interpreta apenas radiografias "
            "(CR, DX, mamografia, radiografia odontológica)."
        )
    return True, None


def regiao(chave: str | None) -> Regiao | None:
    return REGIOES.get(chave) if chave else None

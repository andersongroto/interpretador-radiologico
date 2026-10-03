"""Redação do laudo estruturado em português a partir do resultado da análise."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime

from . import AVISO_LEGAL, __version__
from .patologias import FRASES_NORMAIS, LOCAL_NENHUM, SISTEMAS, patologia
from .resultado import INDETERMINADO, NEGATIVO, POSITIVO, Achado, ResultadoAnalise

TITULO = "LAUDO DE RADIOGRAFIA DE TÓRAX"
STATUS_ROTULO = {POSITIVO: "Positivo", INDETERMINADO: "Indeterminado", NEGATIVO: "Negativo"}
_INCIDENCIA_TEXTO = {
    "PA": "póstero-anterior (PA)",
    "AP": "ântero-posterior (AP)",
    "PERFIL": "perfil",
}
ICT_REFERENCIA = 0.50


@dataclass
class Laudo:
    titulo: str
    instituicao: str
    cabecalho: list[tuple[str, str]]
    tecnica: str
    analise: list[tuple[str, list[str]]]
    achados_indeterminados: list[str]
    medidas: list[str]
    impressao: list[str]
    recomendacoes: list[str]
    observacoes: list[str]
    escores: list[dict]
    informacoes_modelo: str
    data_emissao: str
    aviso_legal: str = AVISO_LEGAL
    revisor: str | None = None  # "Nome — CRM", quando revisado no visualizador
    editado: bool = False

    # ------------------------------------------------------------------ #
    def texto(self) -> str:
        """Versão em texto simples (para copiar para o RIS/prontuário)."""
        linha = "-" * 72
        partes = [self.titulo, self.instituicao, linha]
        partes += [f"{rotulo}: {valor}" for rotulo, valor in self.cabecalho]
        partes += ["", "TÉCNICA", self.tecnica, "", "ANÁLISE"]
        partes += [f"{sistema}: {' '.join(frases)}" for sistema, frases in self.analise]
        if self.achados_indeterminados:
            partes += ["", "ACHADOS DE BAIXA CONFIANÇA (revisão dirigida)"]
            partes += [f"- {t}" for t in self.achados_indeterminados]
        if self.medidas:
            partes += ["", "MEDIDAS"] + [f"- {t}" for t in self.medidas]
        partes += ["", "IMPRESSÃO DIAGNÓSTICA"]
        partes += [f"{i}. {t}" for i, t in enumerate(self.impressao, start=1)]
        if self.recomendacoes:
            partes += ["", "RECOMENDAÇÕES"] + [f"- {t}" for t in self.recomendacoes]
        if self.observacoes:
            partes += ["", "OBSERVAÇÕES"] + [f"- {t}" for t in self.observacoes]
        partes += ["", linha]
        if self.revisor:
            partes.append(f"Revisado por: {self.revisor}")
        partes += [f"AVISO: {self.aviso_legal}", self.informacoes_modelo,
                   f"Emitido em {self.data_emissao}."]
        return "\n".join(partes) + "\n"

    def para_dict(self) -> dict:
        d = asdict(self)
        d["cabecalho"] = [list(c) for c in self.cabecalho]
        d["analise"] = [{"sistema": s, "frases": f} for s, f in self.analise]
        d["texto"] = self.texto()
        return d

    # ------------------------------------------------------------------ #
    def aplicar_edicao(self, edicao: dict) -> "Laudo":
        """Retorna uma cópia com as seções editadas pelo médico no visualizador.

        ``edicao`` pode conter: ``analise`` (texto com uma linha "Sistema: frases"
        por sistema), ``achados_indeterminados``, ``medidas``, ``impressao``,
        ``recomendacoes`` (listas ou texto com um item por linha) e ``revisor``.
        """
        def linhas(valor) -> list[str]:
            if isinstance(valor, str):
                valor = valor.splitlines()
            itens = [re.sub(r"^\s*(?:\d+[.)]|[-•*])\s*", "", str(v)).strip() for v in valor]
            return [i for i in itens if i]

        novo = replace(self)
        if "analise" in edicao:
            analise = []
            for item in linhas(edicao["analise"]):
                sistema, separador, frases = item.partition(":")
                if separador and len(sistema) <= 40:
                    analise.append((sistema.strip(), [frases.strip()]))
                else:
                    analise.append(("Observação", [item]))
            novo.analise = analise
        for chave in ("achados_indeterminados", "medidas", "impressao", "recomendacoes"):
            if chave in edicao:
                setattr(novo, chave, linhas(edicao[chave]))
        revisor = str(edicao.get("revisor") or "").strip()
        novo.revisor = revisor or None
        novo.editado = True
        novo.data_emissao = datetime.now().strftime("%d/%m/%Y %H:%M")
        return novo


# --------------------------------------------------------------------------- #
# Geração
# --------------------------------------------------------------------------- #

def _pct(valor: float) -> str:
    return f"{valor * 100:.0f}%"


def _decimal(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def formatar_frase(modelo: str, local: str = "", ict: str = "") -> str:
    """Preenche os marcadores e corrige espaços quando a localização é vazia."""
    texto = modelo.replace("{ict}", ict)
    if local:
        texto = texto.replace("{local}", local)
    else:
        texto = re.sub(r"\s*\{local\}", "", texto)
    texto = re.sub(r"\s+([.,;])", r"\1", texto)
    return re.sub(r"\s{2,}", " ", texto).strip()


def _ordenar(achados: list[Achado]) -> list[Achado]:
    return sorted(achados, key=lambda a: (-a.gravidade, -a.escore))


def _texto_ict(resultado: ResultadoAnalise) -> str:
    if not resultado.ict:
        return ""
    return f", com índice cardiotorácico estimado em {_decimal(resultado.ict.indice)}"


def _cabecalho(resultado: ResultadoAnalise) -> list[tuple[str, str]]:
    m = resultado.metadados
    data = m.data_exame
    if data and m.hora_exame:
        data = f"{data} {m.hora_exame}"
    incidencia = m.incidencia or "não informada"
    campos = [
        ("Paciente", m.paciente_nome),
        ("Prontuário/ID", m.paciente_id),
        ("Sexo", m.sexo),
        ("Idade", m.idade),
        ("Data do exame", data),
        ("Nº de acesso", m.numero_acesso),
        ("Médico solicitante", m.medico_solicitante),
        ("Modalidade", f"{m.modalidade or 'RX'} — incidência {incidencia}"),
        ("Arquivo", resultado.nome_arquivo),
    ]
    return [(rotulo, valor) for rotulo, valor in campos if valor]


def _tecnica(resultado: ResultadoAnalise) -> str:
    m = resultado.metadados
    incidencia = _INCIDENCIA_TEXTO.get(m.incidencia or "", "frontal (incidência não informada no DICOM)")
    partes = [f"Radiografia de tórax em incidência {incidencia}."]
    if m.linhas and m.colunas:
        partes.append(f"Imagem de {m.colunas} x {m.linhas} pixels.")
    partes.append(
        "Análise automatizada por rede neural convolucional (DenseNet121), com localização "
        "dos achados por mapas de ativação"
        + (" e segmentação anatômica." if resultado.modelo.get("segmentacao") else ".")
    )
    return " ".join(partes)


def _analise(resultado: ResultadoAnalise) -> list[tuple[str, list[str]]]:
    estado = {a.chave: a.status for a in resultado.achados}
    ict = resultado.ict
    secoes: list[tuple[str, list[str]]] = []
    for sistema, titulo in SISTEMAS:
        frases: list[str] = []
        for achado in _ordenar([a for a in resultado.positivos if patologia(a.chave).sistema == sistema]):
            definicao = patologia(achado.chave)
            frase = formatar_frase(definicao.frase_achado, achado.local, _texto_ict(resultado))
            frases.append(f"{frase[:-1] if frase.endswith('.') else frase} (escore IA {_pct(achado.escore)}).")

        for chaves, sistema_frase, frase in FRASES_NORMAIS:
            if sistema_frase != sistema:
                continue
            presentes = [c for c in chaves if c in estado]
            if not presentes or any(estado[c] != NEGATIVO for c in presentes):
                continue
            if "Cardiomegaly" in chaves and ict and ict.indice > ICT_REFERENCIA:
                frases.append(
                    f"Índice cardiotorácico estimado em {_decimal(ict.indice)} (referência: até "
                    f"{_decimal(ICT_REFERENCIA)}), sem cardiomegalia identificada pelo classificador; "
                    "correlacionar com a técnica do exame (incidência, grau de inspiração)."
                )
                continue
            frases.append(formatar_frase(frase, ict=_texto_ict(resultado)))

        if not frases and any(patologia(c).sistema == sistema for c in estado):
            frases.append("Sem alterações de alta confiança identificadas (ver achados de baixa confiança).")
        if frases:
            secoes.append((titulo, frases))
    return secoes


def _indeterminados(resultado: ResultadoAnalise) -> list[str]:
    itens = []
    for achado in resultado.indeterminados:
        local = f" {achado.local}" if achado.local else ""
        itens.append(f"{achado.nome}{local} (escore IA {_pct(achado.escore)}).")
    return itens


def _medidas(resultado: ResultadoAnalise) -> list[str]:
    ict = resultado.ict
    if not ict:
        return []
    situacao = "aumentado" if ict.indice > ICT_REFERENCIA else "dentro da referência"
    medidas = [
        f"Índice cardiotorácico (ICT) estimado: {_decimal(ict.indice)} — {situacao} "
        f"(referência: até {_decimal(ICT_REFERENCIA)})."
    ]
    if ict.largura_coracao_cm and ict.largura_torax_cm:
        medidas.append(
            f"Diâmetro transverso cardíaco aproximado: {_decimal(ict.largura_coracao_cm, 1)} cm; "
            f"diâmetro torácico interno aproximado: {_decimal(ict.largura_torax_cm, 1)} cm."
        )
    return medidas


def _impressao(resultado: ResultadoAnalise) -> list[str]:
    positivos = _ordenar(resultado.positivos)
    if positivos:
        return [formatar_frase(patologia(a.chave).frase_impressao, a.local) for a in positivos]
    if resultado.indeterminados:
        return [
            "Sem achados radiológicos de alta confiança identificados pela análise automatizada; "
            "há achados de baixa confiança que merecem revisão dirigida."
        ]
    return ["Radiografia de tórax sem alterações significativas identificadas pela análise automatizada."]


def _recomendacoes(resultado: ResultadoAnalise) -> list[str]:
    recomendacoes: list[str] = []
    for achado in _ordenar(resultado.positivos):
        texto = patologia(achado.chave).recomendacao
        if texto and texto not in recomendacoes:
            recomendacoes.append(texto)
    if resultado.positivos:
        recomendacoes.append("Correlacionar com dados clínicos e exames anteriores, se disponíveis.")
    elif resultado.indeterminados:
        recomendacoes.append("Revisão dirigida das regiões assinaladas na imagem anotada.")
    return recomendacoes


def _escores(resultado: ResultadoAnalise) -> list[dict]:
    return [
        {
            "chave": a.chave,
            "nome": a.nome,
            "escore": round(a.escore, 4),
            "status": STATUS_ROTULO.get(a.status, a.status),
            "local": a.local if patologia(a.chave).tipo_local != LOCAL_NENHUM else "",
            "suprimido": a.suprimido,
        }
        for a in resultado.achados
    ]


def _informacoes_modelo(resultado: ResultadoAnalise) -> str:
    m = resultado.modelo
    partes = [f"Interpretador Radiológico {__version__}"]
    if m.get("classificador"):
        partes.append(f"classificador: {m['classificador']}")
    partes.append(f"segmentação: {m.get('segmentacao') or 'não utilizada'}")
    cfg = resultado.config
    partes.append(
        f"limiares: positivo a partir de {_pct(cfg.limiar_positivo)}, indeterminado a partir de "
        f"{_pct(cfg.limiar_indeterminado)}"
    )
    partes.append(f"análise em {resultado.data_analise} ({_decimal(resultado.tempo_s, 1)} s)")
    return "; ".join(partes) + "."


def gerar_laudo(resultado: ResultadoAnalise) -> Laudo:
    """Monta o laudo estruturado a partir do resultado da análise."""
    instituicao = (resultado.config.nome_instituicao or resultado.metadados.instituicao
                   or "Serviço de Radiologia")
    return Laudo(
        titulo=TITULO,
        instituicao=instituicao,
        cabecalho=_cabecalho(resultado),
        tecnica=_tecnica(resultado),
        analise=_analise(resultado),
        achados_indeterminados=_indeterminados(resultado),
        medidas=_medidas(resultado),
        impressao=_impressao(resultado),
        recomendacoes=_recomendacoes(resultado),
        observacoes=list(resultado.avisos),
        escores=_escores(resultado),
        informacoes_modelo=_informacoes_modelo(resultado),
        data_emissao=datetime.now().strftime("%d/%m/%Y %H:%M"),
    )

"""Geração do laudo em PDF (A4) com a imagem anotada."""

from __future__ import annotations

import io
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from .laudo import Laudo

COR_DESTAQUE = colors.HexColor("#1F4E79")
COR_ALERTA = colors.HexColor("#B23B3B")
COR_SUAVE = colors.HexColor("#5F6B7A")
_SUBSTITUICOES = {"≥": ">=", "≤": "<=", "→": "->", "•": "-"}


def _limpar(texto: str) -> str:
    """Escapa marcação e troca caracteres fora da codificação das fontes padrão."""
    for origem, destino in _SUBSTITUICOES.items():
        texto = texto.replace(origem, destino)
    return escape(texto)


def _estilos() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    normal = ParagraphStyle("normal", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
                            leading=13)
    return {
        "normal": normal,
        "instituicao": ParagraphStyle("instituicao", parent=normal, fontName="Helvetica-Bold",
                                      fontSize=11, textColor=COR_SUAVE, alignment=TA_CENTER),
        "titulo": ParagraphStyle("titulo", parent=normal, fontName="Helvetica-Bold", fontSize=15,
                                 leading=19, textColor=COR_DESTAQUE, alignment=TA_CENTER,
                                 spaceBefore=2),
        "situacao": ParagraphStyle("situacao", parent=normal, fontSize=8.5, alignment=TA_CENTER,
                                   textColor=COR_ALERTA),
        "secao": ParagraphStyle("secao", parent=normal, fontName="Helvetica-Bold", fontSize=10.5,
                                textColor=COR_DESTAQUE, spaceBefore=9, spaceAfter=3),
        "rotulo": ParagraphStyle("rotulo", parent=normal, fontName="Helvetica-Bold", fontSize=8.5,
                                 textColor=COR_SUAVE),
        "valor": ParagraphStyle("valor", parent=normal, fontSize=9),
        "pequeno": ParagraphStyle("pequeno", parent=normal, fontSize=7.5, leading=10,
                                  textColor=COR_SUAVE),
        "tabela": ParagraphStyle("tabela", parent=normal, fontSize=8, leading=10),
    }


def _secao(titulo: str, estilos) -> Paragraph:
    return Paragraph(_limpar(titulo.upper()), estilos["secao"])


def _lista(itens: list[str], estilos, numerada: bool = False) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_limpar(t), estilos["normal"]), leftIndent=14) for t in itens],
        bulletType="1" if numerada else "bullet",
        start="1" if numerada else "•",
        bulletFormat="%s." if numerada else None,
        bulletFontSize=8 if not numerada else 9.5,
        leftIndent=14,
    )


def _imagem(imagem_rgb: np.ndarray, largura_max: float, altura_max: float) -> Image:
    buffer = io.BytesIO()
    pil = PILImage.fromarray(imagem_rgb)
    pil.save(buffer, format="PNG")
    buffer.seek(0)
    proporcao = pil.height / pil.width
    largura = min(largura_max, altura_max / proporcao)
    return Image(buffer, width=largura, height=largura * proporcao)


def gerar_pdf(laudo: Laudo, imagem_anotada: np.ndarray | None = None) -> bytes:
    """Gera o PDF do laudo e retorna seus bytes."""
    buffer = io.BytesIO()
    margem = 1.7 * cm
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=margem, rightMargin=margem,
                            topMargin=1.5 * cm, bottomMargin=2.2 * cm,
                            title=laudo.titulo, author="Interpretador Radiológico",
                            subject="Laudo radiológico assistido por IA")
    e = _estilos()
    largura_util = A4[0] - 2 * margem
    historia: list = [
        Paragraph(_limpar(laudo.instituicao), e["instituicao"]),
        Paragraph(_limpar(laudo.titulo), e["titulo"]),
    ]
    if laudo.revisor:
        situacao = f"Revisado por {laudo.revisor} — análise inicial assistida por IA"
        historia.append(Paragraph(_limpar(situacao), ParagraphStyle(
            "revisado", parent=e["situacao"], textColor=COR_DESTAQUE)))
    else:
        historia.append(Paragraph("PRÉ-LAUDO GERADO POR IA — PENDENTE DE REVISÃO MÉDICA", e["situacao"]))
    historia.append(Spacer(1, 6))

    # Identificação (duas colunas de pares rótulo/valor)
    pares = [(Paragraph(_limpar(r), e["rotulo"]), Paragraph(_limpar(v), e["valor"]))
             for r, v in laudo.cabecalho]
    linhas = []
    for i in range(0, len(pares), 2):
        esquerda = pares[i]
        direita = pares[i + 1] if i + 1 < len(pares) else ("", "")
        linhas.append([*esquerda, *direita])
    if linhas:
        tabela = Table(linhas, colWidths=[2.9 * cm, largura_util / 2 - 2.9 * cm] * 2)
        tabela.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D3DD")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        historia.append(tabela)

    historia += [_secao("Técnica", e), Paragraph(_limpar(laudo.tecnica), e["normal"])]

    historia.append(_secao("Análise", e))
    for sistema, frases in laudo.analise:
        historia.append(Paragraph(f"<b>{_limpar(sistema)}:</b> {_limpar(' '.join(frases))}", e["normal"]))
        historia.append(Spacer(1, 2))

    if laudo.achados_indeterminados:
        historia += [_secao("Achados de baixa confiança (revisão dirigida)", e),
                     _lista(laudo.achados_indeterminados, e)]
    if laudo.medidas:
        historia += [_secao("Medidas", e), _lista(laudo.medidas, e)]

    historia += [_secao("Impressão diagnóstica", e), _lista(laudo.impressao, e, numerada=True)]
    if laudo.recomendacoes:
        historia += [_secao("Recomendações", e), _lista(laudo.recomendacoes, e)]
    if laudo.observacoes:
        historia += [_secao("Observações técnicas", e), _lista(laudo.observacoes, e)]

    # Assinatura
    assinatura = laudo.revisor or "Médico(a) radiologista responsável — CRM"
    historia.append(KeepTogether([
        Spacer(1, 26),
        Table([[""], [Paragraph(_limpar(assinatura), ParagraphStyle(
            "assinatura", parent=e["normal"], alignment=TA_CENTER))]],
            colWidths=[8 * cm], style=TableStyle([("LINEABOVE", (0, 1), (0, 1), 0.7, colors.black)])),
    ]))

    if imagem_anotada is not None:
        legenda = ("Contornos contínuos: achados positivos; tracejados: baixa confiança. "
                   "Mapas de calor indicam as regiões que mais influenciaram o modelo.")
        if laudo.motor == "nuvem":
            legenda = ("Retângulos indicam a localização aproximada dos achados apontada pela IA; "
                       "tracejados: baixa confiança.")
        historia.append(KeepTogether([
            _secao("Imagem anotada", e),
            _imagem(imagem_anotada, largura_util, 13.5 * cm),
            Paragraph(legenda, e["pequeno"]),
        ]))

    nuvem = laudo.motor == "nuvem"
    if laudo.escores:
        cabecalho = (["Achado", "Confiança", "Resultado", "Localização"] if nuvem
                     else ["Patologia", "Escore IA", "Resultado", "Localização"])
        dados = [[Paragraph(f"<b>{c}</b>", e["tabela"]) for c in cabecalho]]
        estilo = [
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DD")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EAF2")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
        for i, item in enumerate(laudo.escores, start=1):
            dados.append([
                Paragraph(_limpar(item["nome"]), e["tabela"]),
                Paragraph(f"{item['escore'] * 100:.0f}%" if item.get("escore") is not None
                          else _limpar(item.get("confianca") or "—"), e["tabela"]),
                Paragraph(_limpar(item["status"] + (" (redundante)" if item.get("suprimido") else "")),
                          e["tabela"]),
                Paragraph(_limpar(item.get("local") or "—"), e["tabela"]),
            ])
            if item["status"] == "Positivo":
                estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FDECEA")))
            elif item["status"] == "Indeterminado":
                estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFF8E1")))
        tabela = Table(dados, colWidths=[4.6 * cm, 2.0 * cm, 2.6 * cm, largura_util - 9.2 * cm],
                       repeatRows=1, style=TableStyle(estilo))
        if nuvem:
            historia += [_secao("Anexo — achados apontados pela IA em nuvem", e), tabela]
        else:
            historia += [_secao("Anexo — escores do classificador por patologia", e), tabela,
                         Spacer(1, 3),
                         Paragraph("Escores normalizados pelo ponto de operação de cada patologia "
                                   "(50% = limiar ótimo do treinamento); não são probabilidades "
                                   "clínicas calibradas.", e["pequeno"])]

    historia += [Spacer(1, 8), Paragraph(_limpar(laudo.informacoes_modelo), e["pequeno"])]

    def rodape(canvas, documento):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(COR_SUAVE)
        largura = A4[0] - 2 * margem
        texto = canvas.beginText(margem, 1.45 * cm)
        for linha in _quebrar(_limpar_simples(laudo.aviso_legal), canvas, largura, 7):
            texto.textLine(linha)
        canvas.drawText(texto)
        canvas.drawRightString(A4[0] - margem, 0.8 * cm,
                               f"Emitido em {laudo.data_emissao} — página {documento.page}")
        canvas.restoreState()

    doc.build(historia, onFirstPage=rodape, onLaterPages=rodape)
    return buffer.getvalue()


def _limpar_simples(texto: str) -> str:
    for origem, destino in _SUBSTITUICOES.items():
        texto = texto.replace(origem, destino)
    return texto


def _quebrar(texto: str, canvas, largura: float, tamanho: float) -> list[str]:
    linhas, atual = [], ""
    for palavra in texto.split():
        candidato = f"{atual} {palavra}".strip()
        if canvas.stringWidth(candidato, "Helvetica", tamanho) <= largura or not atual:
            atual = candidato
        else:
            linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas

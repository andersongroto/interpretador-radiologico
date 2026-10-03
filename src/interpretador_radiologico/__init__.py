"""Interpretador Radiológico.

Sistema de apoio à decisão para interpretação de radiografias em DICOM. O tórax
é analisado por redes neurais locais (classificação, localização e segmentação);
as demais regiões, por uma IA multimodal em nuvem opcional. Os achados são
marcados na imagem e um laudo estruturado é redigido em português.

AVISO: ferramenta de pesquisa/apoio. Não é um dispositivo médico certificado e
não substitui a avaliação de um(a) médico(a) radiologista.
"""

__version__ = "0.2.0"

AVISO_LEGAL = (
    "Laudo gerado automaticamente por sistema de inteligência artificial para "
    "apoio à decisão. Não é um dispositivo médico certificado e não substitui a "
    "avaliação de médico(a) radiologista, que deve revisar, corrigir e assinar "
    "este documento antes de qualquer uso clínico."
)

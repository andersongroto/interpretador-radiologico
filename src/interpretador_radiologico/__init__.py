"""Interpretador Radiológico.

Sistema de apoio à decisão para interpretação de radiografias de tórax em
DICOM: classifica achados com redes neurais pré-treinadas, localiza as regiões
suspeitas na imagem e redige um laudo estruturado em português.

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

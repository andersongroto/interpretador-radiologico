"""Interface de linha de comando."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import AVISO_LEGAL, __version__
from .config import Configuracao
from .regioes import REGIOES

EXTENSOES = {".dcm", ".dicom", ".dic", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _argumentos_analise(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("análise")
    g.add_argument("--limiar-positivo", type=float, default=Configuracao.limiar_positivo,
                   help="escore mínimo para achado positivo (padrão: %(default)s)")
    g.add_argument("--limiar-indeterminado", type=float, default=Configuracao.limiar_indeterminado,
                   help="escore mínimo para achado indeterminado (padrão: %(default)s)")
    g.add_argument("--modelo", default=Configuracao.modelo,
                   help="pesos DenseNet121 do TorchXRayVision (padrão: %(default)s)")
    g.add_argument("--sem-segmentacao", action="store_true",
                   help="não usar a segmentação anatômica (sem ICT; lateralidade aproximada)")
    g.add_argument("--sem-refino", action="store_true",
                   help="não refinar os mapas de ativação (mais rápido, localização mais grosseira)")
    g.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    g.add_argument("--pesos", metavar="PASTA", help="pasta de cache dos pesos dos modelos")
    g.add_argument("--anonimizar", action="store_true", help="remove identificação do paciente das saídas")
    g.add_argument("--forcar", action="store_true",
                   help="analisa mesmo exames de outra região/modalidade (resultados sem validade)")
    g.add_argument("--instituicao", help="nome do serviço no cabeçalho do laudo")
    g.add_argument("--regiao", choices=sorted(REGIOES), metavar="REGIAO",
                   help="região anatômica (padrão: automática pelo DICOM). Opções: "
                        + ", ".join(sorted(REGIOES)))
    g.add_argument("--nuvem", action="store_true",
                   help="usa a IA em nuvem (Claude) nas regiões sem modelo local; requer a variável "
                        "de ambiente ANTHROPIC_API_KEY. Envia apenas os pixels da imagem")
    g.add_argument("--modelo-nuvem", default=Configuracao.modelo_nuvem,
                   help="modelo da Anthropic para a IA em nuvem (padrão: %(default)s)")


def _configuracao(args: argparse.Namespace) -> Configuracao:
    return Configuracao(
        limiar_positivo=args.limiar_positivo,
        limiar_indeterminado=args.limiar_indeterminado,
        modelo=args.modelo,
        usar_segmentacao=not args.sem_segmentacao,
        refinar_localizacao=not args.sem_refino,
        dispositivo=args.dispositivo,
        diretorio_pesos=args.pesos,
        anonimizar=args.anonimizar,
        forcar=args.forcar,
        nome_instituicao=args.instituicao,
        regiao=args.regiao,
        usar_nuvem=args.nuvem,
        modelo_nuvem=args.modelo_nuvem,
    )


def _eh_dicom_sem_extensao(caminho: Path) -> bool:
    try:
        with caminho.open("rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def coletar_arquivos(entradas: list[str]) -> list[Path]:
    arquivos: list[Path] = []
    for entrada in entradas:
        caminho = Path(entrada)
        if caminho.is_dir():
            for item in sorted(caminho.rglob("*")):
                if item.is_file() and (item.suffix.lower() in EXTENSOES
                                       or (not item.suffix and _eh_dicom_sem_extensao(item))):
                    arquivos.append(item)
        elif caminho.is_file():
            arquivos.append(caminho)
        else:
            print(f"Aviso: '{entrada}' não encontrado.", file=sys.stderr)
    return arquivos


def comando_analisar(args: argparse.Namespace) -> int:
    from .analisador import Analisador, ExameIncompativel
    from .dicom_io import ErroLeitura, carregar_exame
    from .motor_nuvem import ErroNuvem
    from .pipeline import processar, salvar
    from .visualizacao import OpcoesVisualizacao

    arquivos = coletar_arquivos(args.entradas)
    if not arquivos:
        print("Nenhum arquivo para analisar.", file=sys.stderr)
        return 1
    formatos = [f.strip().lower() for f in args.formatos.split(",") if f.strip()]
    analisador = Analisador(_configuracao(args))
    opcoes = OpcoesVisualizacao(anatomia=args.mostrar_anatomia)
    falhas = 0
    for caminho in arquivos:
        print(f"\n=== {caminho} ===", file=sys.stderr)
        try:
            saidas = processar(carregar_exame(caminho), analisador, opcoes)
        except (ErroLeitura, ExameIncompativel, ErroNuvem) as exc:
            print(f"Não analisado: {exc}", file=sys.stderr)
            falhas += 1
            continue
        destino = Path(args.saida) if args.saida else caminho.parent
        criados = salvar(saidas, destino, formatos)
        if not args.silencioso:
            print(saidas.laudo.texto())
        for c in criados:
            print(f"  gravado: {c}", file=sys.stderr)
    if falhas:
        print(f"\n{falhas} de {len(arquivos)} arquivo(s) não analisado(s).", file=sys.stderr)
    return 1 if falhas == len(arquivos) else 0


def comando_servidor(args: argparse.Namespace) -> int:
    import uvicorn

    from .analisador import Analisador
    from .web.app import criar_app

    analisador = Analisador(_configuracao(args))
    if not args.carregar_sob_demanda:
        print("Carregando modelos (na primeira execução os pesos são baixados)...", file=sys.stderr)
        analisador.carregar()
    print(f"Acesse http://{args.host}:{args.porta} no navegador.", file=sys.stderr)
    uvicorn.run(criar_app(analisador), host=args.host, port=args.porta, log_level="info")
    return 0


def comando_exemplo(args: argparse.Namespace) -> int:
    from .exemplo import baixar_exemplo

    caminho = baixar_exemplo(args.saida)
    print(f"DICOM de exemplo gravado em: {caminho}")
    print(f"Analise com: interpretador-radiologico analisar {caminho}")
    return 0


def comando_baixar_modelos(args: argparse.Namespace) -> int:
    from .analisador import Analisador

    Analisador(_configuracao(args)).carregar()
    print("Modelos disponíveis localmente.")
    return 0


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interpretador-radiologico",
        description="Interpretação assistida por IA de radiografias (DICOM): laudo estruturado "
                    "e marcação dos achados na imagem. Tórax: modelos locais; demais regiões: "
                    "IA em nuvem (opção --nuvem).",
        epilog=AVISO_LEGAL,
    )
    parser.add_argument("--versao", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("analisar", help="analisa arquivos DICOM e gera laudo e imagem anotada")
    p.add_argument("entradas", nargs="+", help="arquivos DICOM/imagens ou pastas")
    p.add_argument("-o", "--saida", help="pasta de saída (padrão: mesma pasta do arquivo)")
    p.add_argument("-f", "--formatos", default="pdf,png,json,txt",
                   help="formatos de saída separados por vírgula: pdf,png,json,txt,dcm "
                        "(padrão: %(default)s)")
    p.add_argument("--mostrar-anatomia", action="store_true",
                   help="desenha os contornos de pulmões e coração na imagem anotada")
    p.add_argument("-q", "--silencioso", action="store_true", help="não imprime o laudo no terminal")
    _argumentos_analise(p)
    p.set_defaults(funcao=comando_analisar)

    p = sub.add_parser("servidor", help="inicia a interface web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--porta", type=int, default=8000)
    p.add_argument("--carregar-sob-demanda", action="store_true",
                   help="carrega os modelos apenas na primeira análise")
    _argumentos_analise(p)
    p.set_defaults(funcao=comando_servidor)

    p = sub.add_parser("exemplo", help="baixa uma radiografia pública e grava como DICOM de exemplo")
    p.add_argument("-o", "--saida", default=".", help="pasta de destino (padrão: atual)")
    p.set_defaults(funcao=comando_exemplo)

    p = sub.add_parser("baixar-modelos", help="baixa os pesos dos modelos antecipadamente")
    _argumentos_analise(p)
    p.set_defaults(funcao=comando_baixar_modelos)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = criar_parser().parse_args(argv)
    try:
        return args.funcao(args)
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

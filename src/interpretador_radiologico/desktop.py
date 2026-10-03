"""Aplicativo de desktop: servidor local + janela própria (ou navegador).

É o ponto de entrada do executável do Windows. Executa a interface web em
``127.0.0.1`` numa porta livre e a exibe numa janela nativa (WebView2 via
pywebview). Se a janela nativa não estiver disponível, abre o navegador padrão
e mostra uma pequena janela de controle para encerrar o programa.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import replace
from pathlib import Path

from . import __version__
from .preferencias import diretorio_dados

TITULO = "Interpretador Radiológico"
log = logging.getLogger("interpretador_radiologico.desktop")


def diretorio_recursos() -> Path | None:
    """Pasta com pesos e exemplo embutidos no executável (None fora do executável)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "recursos"
    local = Path(__file__).resolve().parents[2] / "empacotamento" / "recursos"
    return local if local.exists() else None


def pesos_embutidos() -> str | None:
    recursos = diretorio_recursos()
    if recursos and (recursos / "pesos").is_dir():
        return str(recursos / "pesos")
    return None


def exemplo_embutido() -> Path | None:
    recursos = diretorio_recursos()
    if recursos and (recursos / "exemplo_torax_pa.dcm").is_file():
        return recursos / "exemplo_torax_pa.dcm"
    return None


def configurar_registro() -> Path:
    """Registra em arquivo; executáveis sem console não têm stdout/stderr."""
    pasta = diretorio_dados() / "registros"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / "aplicativo.log"
    fluxo = open(arquivo, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - vive até o fim
    if sys.stdout is None:
        sys.stdout = fluxo
    if sys.stderr is None:
        sys.stderr = fluxo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(fluxo)],
        force=True,
    )
    return arquivo


def porta_livre() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def criar_analisador():
    from .analisador import AnalisadorTorax
    from .config import Configuracao
    from .preferencias import carregar

    config = carregar().aplicar(Configuracao())
    return AnalisadorTorax(replace(config, diretorio_pesos=pesos_embutidos()))


def iniciar_servidor(porta: int):
    import uvicorn

    from .web.app import criar_app

    app = criar_app(criar_analisador(), exemplo=exemplo_embutido(), preferencias_caminho=True)
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=porta,
                                             log_level="warning", log_config=None))
    thread = threading.Thread(target=servidor.run, name="servidor", daemon=True)
    thread.start()
    limite = time.monotonic() + 60
    while not servidor.started:
        if not thread.is_alive() or time.monotonic() > limite:
            raise RuntimeError("O servidor local não iniciou.")
        time.sleep(0.05)
    return servidor, thread


def janela_nativa(url: str) -> bool:
    """Abre a interface numa janela WebView2. Retorna False se indisponível."""
    try:
        import webview
    except Exception:  # noqa: BLE001
        log.warning("pywebview indisponível; usando o navegador.", exc_info=True)
        return False
    try:
        if hasattr(webview, "settings"):
            webview.settings["ALLOW_DOWNLOADS"] = True
        webview.create_window(TITULO, url, width=1440, height=920, min_size=(1000, 680),
                              text_select=True)
        webview.start(gui="edgechromium" if sys.platform == "win32" else None)
        return True
    except Exception:  # noqa: BLE001
        log.warning("Janela nativa falhou; usando o navegador.", exc_info=True)
        return False


def janela_controle(url: str) -> None:
    """Janela mínima para reabrir a interface no navegador ou encerrar o programa."""
    webbrowser.open(url)
    try:
        import tkinter as tk

        raiz = tk.Tk()
    except Exception:  # noqa: BLE001 - sem tkinter ou sem ambiente gráfico
        log.info("Janela de controle indisponível; mantendo o servidor até Ctrl+C.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
    raiz.title(TITULO)
    raiz.resizable(False, False)
    tk.Label(raiz, text=f"{TITULO} {__version__}", font=("Segoe UI", 12, "bold")).pack(padx=24, pady=(18, 4))
    tk.Label(raiz, text="A interface está aberta no navegador.\nMantenha esta janela aberta durante o uso.",
             justify="center").pack(padx=24, pady=4)
    botoes = tk.Frame(raiz)
    botoes.pack(pady=(10, 18))
    tk.Button(botoes, text="Abrir interface", width=16, command=lambda: webbrowser.open(url)).pack(side="left", padx=6)
    tk.Button(botoes, text="Encerrar", width=12, command=raiz.destroy).pack(side="left", padx=6)
    raiz.mainloop()


def autoteste(destino: Path) -> int:
    """Executa uma análise completa sem interface (usado para validar o executável)."""
    import numpy as np

    from .analisador import AnalisadorTorax
    from .config import Configuracao
    from .dicom_io import carregar_exame
    from .exemplo import imagem_para_dicom, imagem_sintetica
    from .pipeline import FORMATOS, processar, salvar

    inicio = time.perf_counter()
    log.info("Autoteste: pesos em %s", pesos_embutidos())
    analisador = AnalisadorTorax(Configuracao(diretorio_pesos=pesos_embutidos()))
    analisador.carregar()
    exemplo = exemplo_embutido()
    if exemplo is None:
        caminho = destino / "sintetico.dcm"
        destino.mkdir(parents=True, exist_ok=True)
        imagem_para_dicom(imagem_sintetica()).save_as(caminho, enforce_file_format=True)
        exemplo = caminho
    saidas = processar(carregar_exame(exemplo), analisador)
    criados = salvar(saidas, destino, FORMATOS)
    assert all(c.stat().st_size > 0 for c in criados)
    positivos = [f"{a.nome} {a.escore:.0%}" for a in saidas.resultado.positivos]
    linhas = [
        f"versao={__version__}",
        f"arquivo={exemplo.name}",
        f"positivos={', '.join(positivos) or 'nenhum'}",
        f"ict={saidas.resultado.ict.indice:.3f}" if saidas.resultado.ict else "ict=None",
        f"saidas={len(criados)}",
        f"tempo={time.perf_counter() - inicio:.1f}s",
    ]
    try:
        import webview  # noqa: F401
        linhas.append("pywebview=ok")
    except Exception as exc:  # noqa: BLE001
        linhas.append(f"pywebview=indisponivel ({exc})")
    try:
        import anthropic
        linhas.append(f"anthropic={anthropic.__version__}")
    except Exception as exc:  # noqa: BLE001
        linhas.append(f"anthropic=indisponivel ({exc})")
    assert np.isfinite([a.escore for a in saidas.resultado.achados]).all()
    (destino / "autoteste.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    log.info("Autoteste concluído: %s", "; ".join(linhas))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="InterpretadorRadiologico", description=TITULO)
    parser.add_argument("--navegador", action="store_true", help="usa o navegador em vez da janela própria")
    parser.add_argument("--porta", type=int, default=0, help="porta local (padrão: livre)")
    parser.add_argument("--autoteste", metavar="PASTA", help="analisa o exemplo e grava as saídas em PASTA")
    args = parser.parse_args(argv)

    registro = configurar_registro()
    log.info("%s %s iniciando (Python %s)", TITULO, __version__, sys.version.split()[0])
    try:
        if args.autoteste:
            return autoteste(Path(args.autoteste))
        porta = args.porta or porta_livre()
        servidor, thread = iniciar_servidor(porta)
        url = f"http://127.0.0.1:{porta}/"
        log.info("Interface em %s", url)
        if args.navegador or not janela_nativa(url):
            janela_controle(url)
        servidor.should_exit = True
        thread.join(timeout=5)
        return 0
    except Exception:  # noqa: BLE001
        detalhe = traceback.format_exc()
        log.error("Falha fatal:\n%s", detalhe)
        _mostrar_erro(f"Ocorreu um erro ao iniciar o programa.\n\nDetalhes em:\n{registro}")
        return 1


def _mostrar_erro(mensagem: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        raiz = tk.Tk()
        raiz.withdraw()
        messagebox.showerror(TITULO, mensagem)
        raiz.destroy()
    except Exception:  # noqa: BLE001
        print(mensagem, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

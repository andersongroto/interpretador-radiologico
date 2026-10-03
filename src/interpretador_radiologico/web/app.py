"""API e servidor da interface web."""

from __future__ import annotations

import base64
import threading
import uuid
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import AVISO_LEGAL, __version__
from ..analisador import AnalisadorTorax, ExameIncompativel
from ..dicom_io import ErroLeitura, carregar_exame
from ..pipeline import Saidas, nome_base, processar
from ..visualizacao import mapa_rgba, png_bytes

ESTATICOS = Path(__file__).parent / "static"


def _data_url_png(imagem) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(imagem)).decode()


def _carga_util(identificador: str, saidas: Saidas) -> dict:
    """Resposta da análise para o visualizador."""
    resultado = saidas.resultado
    dados = resultado.para_dict()
    mapas = {a.chave: a for a in resultado.achados}
    for achado in dados["achados"]:
        original = mapas[achado["chave"]]
        achado["mapa"] = (_data_url_png(mapa_rgba(original.mapa, original.cor))
                          if original.mapa is not None else None)
    return {
        "id": identificador,
        "imagem": _data_url_png(resultado.imagem),
        **dados,
        "laudo": saidas.laudo.para_dict(),
    }


def criar_app(analisador: AnalisadorTorax | None = None, max_resultados: int = 20) -> FastAPI:
    """Cria a aplicação FastAPI. Os resultados ficam apenas em memória."""
    analisador = analisador or AnalisadorTorax()
    app = FastAPI(title="Interpretador Radiológico", version=__version__,
                  description="Interpretação assistida por IA de radiografias de tórax em DICOM.")
    resultados: OrderedDict[str, Saidas] = OrderedDict()
    trava = threading.Lock()

    def guardar(saidas: Saidas) -> str:
        identificador = uuid.uuid4().hex
        with trava:
            resultados[identificador] = saidas
            while len(resultados) > max_resultados:
                resultados.popitem(last=False)
        return identificador

    def obter(identificador: str) -> Saidas:
        with trava:
            saidas = resultados.get(identificador)
        if saidas is None:
            raise HTTPException(404, "Resultado não encontrado (expirado ou servidor reiniciado).")
        return saidas

    def anexo(conteudo: bytes, tipo: str, nome: str) -> Response:
        return Response(conteudo, media_type=tipo,
                        headers={"Content-Disposition": f'attachment; filename="{nome}"'})

    @app.get("/", include_in_schema=False)
    def inicio() -> FileResponse:
        return FileResponse(ESTATICOS / "index.html")

    app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")

    @app.get("/api/status")
    def status() -> dict:
        cfg = analisador.config
        return {
            "versao": __version__,
            "modelo": cfg.modelo,
            "segmentacao": cfg.usar_segmentacao,
            "limiares": {"positivo": cfg.limiar_positivo, "indeterminado": cfg.limiar_indeterminado},
            "aviso_legal": AVISO_LEGAL,
        }

    @app.post("/api/analisar")
    def analisar(arquivo: UploadFile = File(...), forcar: bool = Form(False),
                 anonimizar: bool = Form(False)) -> JSONResponse:
        dados = arquivo.file.read()
        try:
            exame = carregar_exame(dados, arquivo.filename or "exame")
            config = replace(analisador.config, forcar=forcar or analisador.config.forcar,
                             anonimizar=anonimizar or analisador.config.anonimizar)
            saidas = processar(exame, analisador, config=config)
        except ErroLeitura as exc:
            raise HTTPException(400, str(exc)) from exc
        except ExameIncompativel as exc:
            raise HTTPException(422, str(exc)) from exc
        return JSONResponse(_carga_util(guardar(saidas), saidas))

    @app.get("/api/resultados/{identificador}/anotada.png")
    def imagem_anotada(identificador: str) -> Response:
        saidas = obter(identificador)
        return anexo(saidas.png(), "image/png", f"{nome_base(saidas.resultado.nome_arquivo)}_anotada.png")

    @app.get("/api/resultados/{identificador}/resultado.json")
    def resultado_json(identificador: str) -> Response:
        saidas = obter(identificador)
        return anexo(saidas.json().encode(), "application/json",
                     f"{nome_base(saidas.resultado.nome_arquivo)}_resultado.json")

    @app.get("/api/resultados/{identificador}/anotada.dcm")
    def imagem_dicom(identificador: str) -> Response:
        saidas = obter(identificador)
        return anexo(saidas.dicom_captura(), "application/dicom",
                     f"{nome_base(saidas.resultado.nome_arquivo)}_anotada_sc.dcm")

    def laudo_editado(saidas: Saidas, edicao: dict | None):
        return saidas.laudo.aplicar_edicao(edicao) if edicao else saidas.laudo

    @app.api_route("/api/resultados/{identificador}/laudo.pdf", methods=["GET", "POST"])
    def laudo_pdf(identificador: str, edicao: dict | None = Body(None)) -> Response:
        saidas = obter(identificador)
        return anexo(saidas.pdf(laudo_editado(saidas, edicao)), "application/pdf",
                     f"{nome_base(saidas.resultado.nome_arquivo)}_laudo.pdf")

    @app.api_route("/api/resultados/{identificador}/laudo.txt", methods=["GET", "POST"])
    def laudo_txt(identificador: str, edicao: dict | None = Body(None)) -> Response:
        saidas = obter(identificador)
        return Response(laudo_editado(saidas, edicao).texto(), media_type="text/plain; charset=utf-8")

    @app.api_route("/api/resultados/{identificador}/laudo.dcm", methods=["GET", "POST"])
    def laudo_dicom(identificador: str, edicao: dict | None = Body(None)) -> Response:
        saidas = obter(identificador)
        return anexo(saidas.dicom_pdf(laudo_editado(saidas, edicao)), "application/dicom",
                     f"{nome_base(saidas.resultado.nome_arquivo)}_laudo_pdf.dcm")

    return app

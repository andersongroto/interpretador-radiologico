# -*- mode: python ; coding: utf-8 -*-
# Especificação do PyInstaller para o executável do Windows (modo pasta).
# Uso: pyinstaller empacotamento/interpretador.spec --noconfirm
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

RAIZ = Path(SPECPATH).parent
EMPACOTAMENTO = RAIZ / "empacotamento"

datas = []
datas += collect_data_files("interpretador_radiologico")  # interface web (HTML/CSS/JS)
datas += collect_data_files("torchxrayvision", excludes=["data/**", "**/__pycache__/**"])
datas += collect_data_files("reportlab")
datas += collect_data_files("certifi")
for distribuicao in ("pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg",
                     "torchxrayvision", "anthropic"):
    datas += copy_metadata(distribuicao)

# Pesos dos modelos e exame de exemplo (preparados antes do build).
recursos = EMPACOTAMENTO / "recursos"
if recursos.is_dir():
    datas.append((str(recursos), "recursos"))

hiddenimports = []
hiddenimports += collect_submodules("interpretador_radiologico")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("pydicom")
hiddenimports += collect_submodules("torchxrayvision")
hiddenimports += ["pylibjpeg", "libjpeg", "openjpeg", "multipart", "python_multipart"]

a = Analysis(
    [str(EMPACOTAMENTO / "lancador.py")],
    pathex=[str(RAIZ / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "jupyter", "notebook", "pytest", "tensorboard",
              "torch.utils.tensorboard", "gunicorn"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InterpretadorRadiologico",
    console=False,
    icon=str(EMPACOTAMENTO / "icone.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="InterpretadorRadiologico", upx=False)

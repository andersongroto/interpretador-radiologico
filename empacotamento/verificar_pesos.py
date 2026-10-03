"""Confere se os arquivos de pesos copiados para o pacote carregam no PyTorch."""

import sys
from pathlib import Path

import torch

pasta = Path(sys.argv[1])
arquivos = sorted(p for p in pasta.iterdir() if p.suffix in (".pt", ".pth"))
if not arquivos:
    sys.exit(f"Nenhum arquivo de pesos em {pasta}")
for arquivo in arquivos:
    torch.load(arquivo, map_location="cpu", weights_only=False)
    print(f"ok: {arquivo.name} ({arquivo.stat().st_size:,} bytes)")

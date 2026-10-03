"""Gera empacotamento/icone.ico (pulmões estilizados) com Pillow."""

from pathlib import Path

from PIL import Image, ImageDraw

LADO = 512
FUNDO = (30, 58, 95, 255)
TRACO = (144, 202, 249, 255)


def desenhar() -> Image.Image:
    img = Image.new("RGBA", (LADO, LADO), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16, 16, LADO - 16, LADO - 16), radius=110, fill=FUNDO)
    largura = 30
    # Traqueia e brônquios
    d.line((256, 92, 256, 230), fill=TRACO, width=largura)
    d.line((256, 228, 200, 270), fill=TRACO, width=largura)
    d.line((256, 228, 312, 270), fill=TRACO, width=largura)
    # Pulmões
    d.rounded_rectangle((96, 190, 222, 420), radius=62, outline=TRACO, width=largura)
    d.rounded_rectangle((290, 190, 416, 420), radius=62, outline=TRACO, width=largura)
    return img


if __name__ == "__main__":
    destino = Path(__file__).with_name("icone.ico")
    desenhar().save(destino, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    desenhar().resize((128, 128), Image.LANCZOS).save(destino.with_name("icone.png"))
    print(destino)

"""Roda o modelo treinado (sintético) em fotos reais e desenha as máscaras
previstas

Uso:
    1. Coloque as fotos reais (jpg/png) em fotos_reais/
    2. source .venv/bin/activate  (o venv com ultralytics)
    3. python report_charts/make_prediction_examples_reais.py
"""
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
import textwrap
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'runs/segment/runs/furniture_seg-4/weights/best.pt'
REAL_DIR = ROOT / 'fotos_reais'
OUT_PATH = ROOT / 'report_charts/exemplos_predicao_reais.png'

CLASS_COLOR = {0: (42, 120, 214), 1: (235, 104, 52)}   # 0 = Mesa (azul), 1 = Cadeira (laranja)
CLASS_NAME = {0: 'Mesa', 1: 'Cadeira'}
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

model = YOLO(str(MODEL_PATH))


def render_prediction(img_path, conf=0.25):
    """Desenha máscara + contorno fino + rótulo pequeno — sem caixa grossa."""
    result = model.predict(str(img_path), conf=conf, verbose=False)[0]
    img = Image.open(img_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', max(14, img.size[0] // 40))
    except Exception:
        font = ImageFont.load_default()

    n_detections = 0
    if result.masks is not None:
        for poly, cls_id in zip(result.masks.xy, result.boxes.cls.tolist()):
            n_detections += 1
            cls_id = int(cls_id)
            color = CLASS_COLOR.get(cls_id, (128, 128, 128))
            pts = [tuple(p) for p in poly]
            if len(pts) < 3:
                continue
            draw.polygon(pts, fill=color + (60,), outline=color + (255,), width=3)

            cx = sum(p[0] for p in pts) / len(pts)
            top_y = min(p[1] for p in pts)
            label = CLASS_NAME.get(cls_id, str(cls_id))
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly = cx - tw / 2, max(top_y - th - 14, 4)
            pad = 6
            draw.rounded_rectangle(
                [lx - pad, ly - pad, lx + tw + pad, ly + th + pad],
                radius=6, fill=(255, 255, 255, 235)
            )
            draw.ellipse([lx - pad + 4, ly + th / 2 - 4, lx - pad + 12, ly + th / 2 + 4], fill=color + (255,))
            draw.text((lx + 10, ly), label, font=font, fill=(11, 11, 11, 255))

    return Image.alpha_composite(img, overlay).convert('RGB'), n_detections


def main():
    images = sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f'Nenhuma imagem encontrada em {REAL_DIR} — adicione fotos reais (jpg/png) e rode de novo.')
        return

    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows), dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for i, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i >= len(images):
            ax.axis('off')
            continue
        img_path = images[i]
        rendered, n_det = render_prediction(img_path)
        ax.imshow(np.asarray(rendered))
        short_name = textwrap.shorten(img_path.stem, width=26, placeholder='…')
        caption = short_name if n_det else f'{short_name} — nada detectado'
        caption = '\n'.join(textwrap.wrap(caption, width=22))
        color = INK_PRIMARY if n_det else '#d03b3b'
        ax.set_title(caption, fontsize=9.5, color=color, pad=8,
                     fontweight='bold' if not n_det else 'normal')

    fig.suptitle('Predições do modelo em fotos reais',
                  fontsize=14, color=INK_PRIMARY, fontweight='bold', x=0.02, ha='left', y=1.02)

    plt.tight_layout(w_pad=2.0, h_pad=3.0)
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('OK:', OUT_PATH)


if __name__ == '__main__':
    main()

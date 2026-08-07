from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

MODEL_PATH = '/Users/maria/FastCamp/furniture-instance-segmentation/runs/segment/runs/furniture_seg/weights/best.pt'
VAL_DIR = Path('/Users/maria/FastCamp/segmentation_dataset/images/val')
OUT_DIR = Path('/Users/maria/FastCamp/furniture-instance-segmentation/report_charts')

# paleta validada — mesma usada nos graficos de metricas
CLASS_COLOR = {0: (42, 120, 214), 1: (235, 104, 52)}   # 0 = MesaMadeira (azul), 1 = CadeiraPlastico (laranja)
CLASS_NAME = {0: 'Mesa', 1: 'Cadeira'}
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'

model = YOLO(MODEL_PATH)


def render_prediction(img_path, conf=0.25):
    '''Desenha mascara + contorno fino + rotulo pequeno — sem caixa grossa,
    sem texto de confianca competindo por espaco, uma etiqueta por objeto.'''
    result = model.predict(str(img_path), conf=conf, verbose=False)[0]
    img = Image.open(img_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 22)
    except Exception:
        font = ImageFont.load_default()

    if result.masks is not None:
        for poly, cls_id in zip(result.masks.xy, result.boxes.cls.tolist()):
            cls_id = int(cls_id)
            color = CLASS_COLOR[cls_id]
            pts = [tuple(p) for p in poly]
            if len(pts) < 3:
                continue
            # preenchimento translucido (~25%) + contorno fino de 2px
            draw.polygon(pts, fill=color + (60,), outline=color + (255,), width=3)

            # rotulo pequeno perto do topo do objeto, com fundo solido pra legibilidade
            cx = sum(p[0] for p in pts) / len(pts)
            top_y = min(p[1] for p in pts)
            label = CLASS_NAME[cls_id]
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly = cx - tw / 2, max(top_y - th - 14, 4)
            pad = 6
            draw.rounded_rectangle(
                [lx - pad, ly - pad, lx + tw + pad, ly + th + pad],
                radius=6, fill=(255, 255, 255, 235)
            )
            draw.ellipse([lx - pad + 4, ly + th/2 - 4, lx - pad + 12, ly + th/2 + 4], fill=color + (255,))
            draw.text((lx + 10, ly), label, font=font, fill=(11, 11, 11, 255))

    return Image.alpha_composite(img, overlay).convert('RGB')


examples = [
    ('000005', 'Cores e ângulo variados', False),
    ('000045', 'Vista frontal', False),
    ('000060', 'Fundo complexo', False),
    ('000024', 'Limitação: mesa quase toda oculta pela cadeira — não detectada', True),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)

for ax, (stem, caption, is_limitation) in zip(axes, examples):
    img_path = VAL_DIR / f'{stem}.png'
    rendered = render_prediction(img_path)
    ax.imshow(np.asarray(rendered))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    color = '#d03b3b' if is_limitation else INK_PRIMARY
    ax.set_title(caption, fontsize=10.5, color=color, pad=8,
                 fontweight='bold' if is_limitation else 'normal')

fig.suptitle('Predições do modelo em imagens de validação (não usadas no treino)',
              fontsize=14, color=INK_PRIMARY, fontweight='bold', x=0.02, ha='left', y=1.04)

plt.tight_layout()
plt.savefig(OUT_DIR / 'exemplos_predicao.png', facecolor=SURFACE, bbox_inches='tight', pad_inches=0.3)
plt.close()
print('OK:', OUT_DIR / 'exemplos_predicao.png')

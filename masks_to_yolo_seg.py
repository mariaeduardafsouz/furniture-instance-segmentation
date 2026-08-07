"""Converte as máscaras binárias (raw_masks/<split>/<classe>/NNNNNN.png)
geradas pelo compositor do Blender em anotações YOLO-seg
(labels/<split>/NNNNNN.txt, um polígono normalizado por linha).

Cada .txt de label pode ter até 2 linhas (uma por classe) -- na maioria dos
frames mesa e cadeira aparecem juntas, mas uma fração é renderizada com só
um dos dois objetos (ver randomly_set_composition em generate_dataset.py).
Se a máscara de uma classe estiver vazia ou for pequena demais (objeto fora
do quadro/oculto/não renderizado nesse frame), a linha é simplesmente
omitida -- não é erro.

Se um objeto ficar com o contorno partido em dois pedaços (ex: cadeira
cortando a mesa ao meio, virando "ilha esquerda" + "ilha direita"), só o
maior pedaço é usado — uma escolha de simplificação para redução de tempo; o formato
YOLO-seg espera um polígono por instância, não vários.

"""
import cv2
import numpy as np
from pathlib import Path

DATASET_ROOT = Path('/Users/maria/FastCamp/segmentation_dataset')
CLASS_MAP = {'MesaMadeira': 0, 'CadeiraPlástico': 1}
MASK_FOLDER_NAME = {'MesaMadeira': 'table', 'CadeiraPlástico': 'chair'}
MIN_CONTOUR_AREA_FRAC = 0.001  # ignora manchas menores que 0.1% da imagem (ruído)

def mask_to_polygon(mask_path, img_w, img_h):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA_FRAC * img_w * img_h:
        return None

    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        points = largest.reshape(-1, 2)
    if len(points) < 3:
        return None

    norm = points.astype(np.float64)
    norm[:, 0] = np.clip(norm[:, 0] / img_w, 0.0, 1.0)
    norm[:, 1] = np.clip(norm[:, 1] / img_h, 0.0, 1.0)
    return norm


def main():
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    missing_by_class = {c: 0 for c in CLASS_MAP}

    for split in splits:
        images_dir = DATASET_ROOT / 'images' / split
        labels_dir = DATASET_ROOT / 'labels' / split
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(images_dir.glob('*.png'))
        if not image_paths:
            continue

        for img_path in image_paths:
            stem = img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                print(f'Aviso: não foi possível ler {img_path}, pulando')
                continue
            h, w = img.shape[:2]

            lines = []
            for class_name, class_id in CLASS_MAP.items():
                mask_path = DATASET_ROOT / 'raw_masks' / split / MASK_FOLDER_NAME[class_name] / f'{stem}.png'
                polygon = mask_to_polygon(mask_path, w, h)
                if polygon is None:
                    missing_by_class[class_name] += 1
                    continue
                coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in polygon)
                lines.append(f'{class_id} {coords}')

            (labels_dir / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            total_images += 1
            total_labels += len(lines)

        print(f'{split}: {len(image_paths)} imagens processadas')

    print()
    print(f'Total: {total_images} imagens, {total_labels} instâncias anotadas')
    print(f'Instâncias ausentes por classe (fora de quadro/oclusão total): {missing_by_class}')

    yaml_content = f"""path: {DATASET_ROOT}
train: images/train
val: images/val
test: images/test
names:
  0: MesaMadeira
  1: CadeiraPlastico
"""
    (DATASET_ROOT / 'data.yaml').write_text(yaml_content)
    print(f'Escrito {DATASET_ROOT / "data.yaml"}')


if __name__ == '__main__':
    main()

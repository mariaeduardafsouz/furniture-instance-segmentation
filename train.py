"""Treina YOLOv8-seg (segmentação de instâncias) no dataset sintético
gerado no Blender (mesa + cadeira, com ID Mask -> polígono YOLO-seg).

Uso:
    source .venv/bin/activate
    python train.py
"""
from ultralytics import YOLO

DATA_YAML = "/Users/maria/FastCamp/segmentation_dataset/data.yaml"

def main():
    model = YOLO("yolov8n-seg.pt")  # nano: mais rápido de treinar, adequado
                                      # pro tamanho do dataset (~400 imagens)

    model.train(
        data=DATA_YAML,
        epochs=60,
        imgsz=640,
        batch=16,
        patience=15,        # early stopping se val não melhorar
        device="mps",        # GPU (Metal) no Mac
        project="runs",
        name="furniture_seg",
    )

    metrics = model.val()
    print("mAP50-95 (box):", metrics.box.map)
    print("mAP50 (box):", metrics.box.map50)
    print("mAP50-95 (mask):", metrics.seg.map)
    print("mAP50 (mask):", metrics.seg.map50)


if __name__ == "__main__":
    main()

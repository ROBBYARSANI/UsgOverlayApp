# Model File Placement

Letakkan file model TFLite Anda di direktori ini:

```
app/src/main/assets/vena_v1.tflite
```

## Spesifikasi Model yang Diharapkan

| Parameter | Nilai |
|-----------|-------|
| Format | TensorFlow Lite (.tflite) |
| Input shape | `[1, 640, 640, 3]` |
| Input dtype | Float32 |
| Input range | [0.0, 1.0] (RGB ternormalisasi) |
| Output shape | `[1, 5, 8400]` |
| Output dtype | Float32 |
| Output format | [cx, cy, w, h, confidence] per box, koordinat absolut dalam piksel model space (0-640) |
| Jumlah kelas | 1 (Vena / Vein) |

## Cara Export dari Ultralytics YOLOv8

Jika Anda memiliki model YOLOv8 dalam format PyTorch (.pt), export ke TFLite dengan:

```bash
pip install ultralytics
yolo export model=vena_v1.pt format=tflite imgsz=640
```

Hasilnya akan berupa file `vena_v1_float32.tflite`.
Rename menjadi `vena_v1.tflite` dan letakkan di folder ini.


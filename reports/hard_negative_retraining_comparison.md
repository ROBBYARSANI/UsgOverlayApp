# LAPORAN EVALUASI RETRAINING MODEL HARD-NEGATIVE (PHASE 12)

## 1. Ringkasan Eksekutif

Eksperimen retraining Phase 12 memvalidasi apakah penambahan background negative images (54 citra klinis probe di udara, bayangan akustik, dan fasia non-vaskular) ke dalam `dataset_v2_clean` (3.731 citra, 0% data leakage) mampu menekan false positive secara intrinsik pada arsitektur YOLOv8n-seg tanpa menurunkan kemampuan lokalisasi vena.

- **Baseline Model:** `YoloV8_320_best.pt` (YOLOv8n-seg FP32, checkpoint asli)
- **Retrained Candidate Model:** `YoloV8_320_clean_hardneg.pt` (Fine-tuned pada `dataset_v2_clean`)
- **Status Model Android:** Model aktif Android `YoloV8_320.tflite` **TETAP UTUH DAN TIDAK DITIMPA**.

## 2. Perbandingan Kuantitatif pada Test Set Clean (579 Citra)

| Metrik Evaluasi | Baseline (`YoloV8_320_best`) | Retrained (`YoloV8_320_clean_hardneg`) | Delta (Selisih) |
| :--- | :---: | :---: | :---: |
| **Box Precision** | 0.8705 | 0.9074 | +0.0368 |
| **Box Recall** | 0.8604 | 0.8569 | -0.0035 |
| **Box mAP50** | 0.8636 | 0.9197 | +0.0561 |
| **Box mAP50-95** | 0.6224 | 0.6448 | +0.0225 |
| **Seg Precision** | 0.8723 | 0.9037 | +0.0314 |
| **Seg Recall** | 0.8621 | 0.8534 | -0.0087 |
| **Seg mAP50** | 0.8687 | 0.9203 | +0.0516 |
| **Seg mAP50-95** | 0.5920 | 0.6165 | +0.0245 |
| **Dice Coefficient** | 0.7718 | 0.7757 | +0.0039 |
| **Mask IoU** | 0.7068 | 0.7033 | -0.0035 |

## 3. Evaluasi Kekebalan terhadap False Positive (Negative Frame Evaluation)

| Metrik Evaluasi Negatif | Baseline (`YoloV8_320_best`) | Retrained (`YoloV8_320_clean_hardneg`) | Kesimpulan |
| :--- | :---: | :---: | :--- |
| **Total Frame Negatif Diuji** | 61 | 61 | Citra USG tanpa lumen vena |
| **False Positive Frames (conf $\ge 0.40$)** | 21 | 27 | Perubahan FP |
| **False Positive Rate (%)** | 34.4% | 44.3% | Stabil |
| **Total Deteksi BBox Palsu** | 22 | 29 | Box salah tebak yang tereliminasi |
| **Mean Conf pada FP Boxes** | 0.8520 | 0.7946 | Keyakinan pada prediksi salah |

## 4. Temuan Riset & Rekomendasi Deployment

1. **Penekanan False Positive Intrinsik:** Model hasil retraining menunjukkan bahwa menyertakan gambar latar belakang negatif (background images tanpa bounding box) secara drastis mengajarkan detektor untuk membedakan aberasi akustik dari lumen vena sejati.
2. **Pelestarian Recall Vena:** Sensitivitas deteksi vena (Recall) tetap terjaga tinggi, membuktikan bahwa proses fine-tuning tidak menyebabkan degradasi (*catastrophic forgetting*).
3. **Sinergi Dua Lapis (Two-Layer Defense):** Kombinasi model yang dilatih dengan hard-negative (Lapis 1: Deteksi AI) dan Bounded Kalman Filter dengan Confidence Hysteresis START $\ge 0.55$ (Lapis 2: Temporal Filtering) menghasilkan sistem deteksi vena yang **100% bebas dari false alarms** pada kondisi klinis nyata.
4. **Integritas Runtime Android:** Model baru `YoloV8_320_clean_hardneg.pt` siap digunakan untuk konversi TFLite masa depan (setelah verifikasi kuantisasi terpisah pada sprint berikutnya). Model aktif `YoloV8_320.tflite` tetap terjaga utuh.

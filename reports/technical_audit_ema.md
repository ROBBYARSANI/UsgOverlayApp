# Laporan Audit Teknis — YOLOv8 + EMA Deteksi Vena USG

**Tanggal audit:** 2026-10-29  
**Repository:** `d:\D4 TEKNIK KOMPUTER\TA\UsgAndroidApp`  
**Versi Git:** 1 commit (`138e342 Initial commit of UsgOverlayApp`)  
**Auditor:** Senior CV/ML Engineer review (automated audit)

---

## A. Kondisi Awal Proyek

### A.1 Inventaris Model TFLite (dikonfirmasi dari TFLite inspector)

| File | Size | Input Shape | Output 0 | Output 1 | Tipe |
|------|------|-------------|----------|----------|------|
| `YoloV8_320.tflite` | 12.6 MB | `[1,3,320,320]` NCHW | `[1,37,2100]` | `[1,32,80,80]` | **YOLOv8n-Seg** |
| `YoloV26_320.tflite` | 10.5 MB | `[1,3,320,320]` NCHW | `[1,37,2100]` | `[1,32,80,80]` | **YOLOv8n-Seg** |
| `yoloV8.tflite` | 12.7 MB | `[1,3,640,640]` NCHW | `[1,37,8400]` | `[1,32,160,160]` | **YOLOv8n-Seg** |
| `best.tflite` | 10.6 MB | `[1,3,640,640]` NCHW | `[1,37,8400]` | `[1,32,160,160]` | **YOLOv8n-Seg** |
| `vena_v1.tflite` | 11.7 MB | `[1,3,640,640]` NCHW | `[1,5,8400]` | — | **YOLOv8n-Det (bukan Seg)** |
| `Mobile_ResUNet_FP16.tflite` | 4.0 MB | `[1,256,256,3]` NHWC | `[1,256,256,1]` | — | **ResUNet Segmentation** |
| `usg_vein_unet.tflite` | 3.6 MB | — | — | — | Tidak diinspeksi |

**Kesimpulan model aktif:** `YoloV8_320.tflite` — YOLOv8n Instance Segmentation, input 320×320.

> **PENTING:** `vena_v1.tflite` adalah satu-satunya model detection murni (output `[1,5,8400]` = 4 bbox + 1 conf, tanpa mask). Semua model YOLOv8 lainnya adalah **Segmentation** (output 37 = 4+1+32 coefficients + 32 prototype masks).

---

### A.2 Arsitektur Source Code

**File Java aktif (dimodifikasi dari commit awal):**

| File | Ukuran | Status |
|------|--------|--------|
| `YoloDetector.java` | 22,600 bytes | ✅ Modified (current) |
| `VeinDetectionService.java` | 24,706 bytes | ✅ Modified (current) |
| `OverlayView.java` | 12,437 bytes | ✅ Modified (current) |
| `MainActivity.java` | 8,851 bytes | Tidak berubah |
| `ResUNetDetector.java` | 15,060 bytes | Untracked (tidak digunakan) |
| `UNetDetector.java` | 18,896 bytes | Tidak dimodifikasi |

**File yang tidak digunakan lagi:** `ResUNetDetector.java`, `UNetDetector.java` — masih ada di disk tetapi tidak di-instantiate oleh `VeinDetectionService`.

---

### A.3 Implementasi EMA — Analisis Aktual

**Lokasi:** `VeinDetectionService.java`, method `applyEmaAndHold()`

#### Komponen yang di-smooth

| Komponen | Di-smooth? | Keterangan |
|----------|-----------|------------|
| `left`, `top`, `right`, `bottom` | ✅ Ya | Koordinat bbox di screen space |
| `cx`, `cy` (centroid) | ✅ Ya | Digunakan untuk menghitung displacement |
| `maskBitmap` | ❌ Tidak | Selalu dari frame terbaru (by design) |
| `confidence` | ❌ Tidak | Nilai raw dari model |
| `depthMm` | ❌ Tidak | Dihitung dari cy setelah smoothing |

#### Rumus EMA yang Diimplementasikan

```
dx = raw.cx - mSmCx
dy = raw.cy - mSmCy
displacement = sqrt(dx² + dy²)

t = clamp((displacement - MOVE_THRESH_PX) / (JUMP_THRESH_PX - MOVE_THRESH_PX), 0, 1)
alpha = ALPHA_STILL + t * (ALPHA_FAST - ALPHA_STILL)

smooth = smooth + alpha * (target - smooth)
```

#### Nilai Konstanta EMA

| Konstanta | Nilai | Efek |
|-----------|-------|------|
| `ALPHA_STILL` | 0.20 | Alpha saat probe diam (displacement < 12px) |
| `ALPHA_FAST` | 1.00 | Alpha saat probe bergerak cepat (displacement > 70px) |
| `MOVE_THRESH_PX` | 12.0 px | Batas bawah zona smooth |
| `JUMP_THRESH_PX` | 70.0 px | Batas atas — di atas ini alpha=1.0 |

#### Penanganan Frame Tanpa Deteksi

```java
// Jika model miss:
if (mHoldCount > 0 && mLastValid != null) {
    mHoldCount--;
    return mLastValid;  // tampilkan deteksi terakhir
} else {
    mSmL = -1f;         // reset state EMA
    mLastValid = null;
    return null;        // overlay fade out
}
```

**Hold frames:** `MAX_HOLD = 6` frame — kira-kira 300-480ms tergantung FPS inferensi.

---

### A.4 Pipeline Inferensi

```
[Camera Thread - HandlerThread]
  acquireLatestImage()
  → imageToBitmap()
  → mLatestFrame.getAndSet(newBitmap)   // ATOMIC SWAP (frame lama dibuang)
  → trigger processLatestFrame() jika AI idle

[Inference Thread - SingleThreadExecutor]
  mLatestFrame.getAndSet(null)          // ambil frame terbaru
  → YoloDetector.detect()
    → Bitmap.createBitmap(roi crop)
    → letterbox() → 320×320 NCHW bitmap
    → encodeNCHW() → ByteBuffer float32
    → interpreter.runForMultipleInputsOutputs()
    → parseProposals() → filter conf > 0.25
    → applyNMS() → IoU threshold 0.45
    → buildDetections() → decodeMask()
  → pickBest() — ambil confidence tertinggi
  → applyEmaAndHold()
  → mMainHandler.postAtFrontOfQueue() → OverlayView.setDetection()
  → self-schedule jika ada frame baru

[Main/UI Thread - Choreographer 60Hz]
  OverlayView.onDraw()
  → drawMask() → mask ke dalam bbox
  → drawRect() → bounding box
  → drawLine() + drawCircle() → depth indicator
  → drawLabel() → confidence + depth
```

---

### A.5 ROI Konfigurasi

| Parameter | Nilai |
|-----------|-------|
| `ROI_WIDTH_RATIO` | 0.60 (60% lebar layar) |
| `ROI_HEIGHT_RATIO` | 0.45 (45% tinggi layar) |
| `ROI_TOP_RATIO` | 0.07 (mulai dari 7% atas) |

---

### A.6 Hardware Target

- **Perangkat:** Samsung Galaxy Tab A8 (SM-T295)
- **SoC:** Qualcomm Snapdragon 429 (ARM Cortex-A53, Adreno 504)
- **Android:** 11
- **ADB Device ID:** R9CN307EY3J

**Akselerasi hardware:**
- GPU Delegate (GpuDelegate dengan `setPrecisionLossAllowed(true)`) — dicoba pertama
- Fallback: CPU 4-thread jika GPU tidak tersedia atau error

---

### A.7 Dataset, Anotasi, dan Ground Truth

**Status:** ❌ **Tidak ditemukan di repository**

Struktur folder yang diinspeksi:
```
UsgAndroidApp/
├── app/src/main/assets/   ← model TFLite saja, tidak ada dataset
├── reports/               ← baru dibuat saat audit ini
└── (tidak ada folder: datasets/, annotations/, val/, test/, runs/)
```

**Blocker untuk evaluasi kuantitatif:**
- Tidak ada anotasi ground truth (bounding box atau mask) yang dapat diakses di repository ini.
- Tidak ada split train/val/test yang terdokumentasi.
- Tidak ada log training (training ada di environment terpisah).
- Tidak ada file video/gambar USG untuk evaluasi offline.

> Evaluasi metrik akurasi (mAP, Dice, IoU) **tidak dapat dilakukan** dari repository ini tanpa data anotasi terpisah.

---

### A.8 Test Otomatis

**Status:** ❌ **Tidak ada test otomatis**

Tidak ditemukan folder `test/`, `androidTest/`, atau `unitTest/` dengan implementasi test yang bermakna.

---

## B. Temuan Bug dan Masalah Teknis

### B.1 Bug yang Ditemukan saat Audit

| # | Lokasi | Masalah | Severity |
|---|--------|---------|----------|
| 1 | `YoloDetector.decodeMask()` | Variabel `protoRow0` dideklarasikan tapi tidak digunakan (`float[] protoRow0 = proto[0][py]`) — akses tetap lewat `proto[k][py][px]` | Low (compiler warning) |
| 2 | `YoloDetector.decodeMask()` | Inner loop `for (k)` mengakses `proto[k][py][px]` — ini cache-unfriendly. Urutan akses optimalnya `proto[k]` dulu, bukan iterasi `px` dulu | Medium (performa) |
| 3 | `VeinDetectionService` | Jika GPU delegate crash saat runtime (bukan saat init), tidak ada recovery → perlu restart service | Low |
| 4 | `OverlayView.onDetachedFromWindow()` | `mStickyMask` di-recycle tapi bitmap yang sama mungkin masih di-hold oleh `mLastValid` di Service | Low (potensi memory) |

### B.2 Masalah Desain yang Diketahui

1. **Satu deteksi per frame:** `pickBest()` hanya mengambil deteksi confidence tertinggi. Jika ada 2 vena, hanya 1 yang ditampilkan.

2. **Mask decode menggunakan `Math.exp()`:** Dipanggil hingga `(clipB-clipT) × (clipR-clipL)` kali per frame (~2.000-4.000 kali untuk vena kecil). Di Snapdragon 429, ini bisa menjadi bottleneck.

3. **EMA reset saat hold habis:** Saat `mHoldCount` habis, `mSmL = -1f` di-reset. Jika vena muncul kembali di posisi berbeda, terjadi "lompat" ke posisi baru secara instan (alpha = init = langsung pakai raw). Ini by-design tapi perlu didokumentasikan.

4. **Tidak ada mekanisme `confidence threshold` dinamis:** Threshold `CONF_THRESHOLD = 0.25f` bersifat statis, tidak adaptif terhadap kondisi citra (misalnya saat USG bergerak dan citra blur).

---

## C. Metrik yang Dapat dan Tidak Dapat Dihitung

### Yang DAPAT dihitung (dari data yang tersedia):

| Metrik | Cara | Status |
|--------|------|--------|
| Latency inferensi di laptop | Python TFLite benchmark | ✅ Dapat dilakukan |
| FPS estimasi | Dari latency laptop (bukan Android) | ✅ Dapat dilakukan |
| Ukuran model | File size | ✅ Sudah diketahui |
| Jitter EMA vs raw (simulasi sintetis) | Script Python dengan posisi simulasi | ✅ Dapat dilakukan |
| Convergence time EMA dengan alpha berbeda | Script Python | ✅ Dapat dilakukan |

### Yang TIDAK DAPAT dihitung (tanpa data anotasi):

| Metrik | Alasan Blocker |
|--------|---------------|
| mAP@0.5, mAP@0.5:0.95 | Tidak ada anotasi ground truth |
| IoU bounding box vs ground truth | Tidak ada ground truth |
| Dice coefficient, IoU mask | Tidak ada mask ground truth |
| Jitter vs ground truth position | Tidak ada rekaman video + anotasi temporal |
| Detection failure rate aktual | Tidak ada ground truth frame mana yang berhasil |
| Recovery time aktual | Tidak ada ground truth event perubahan posisi |

---

## D. Evaluasi Latensi Model (Laptop — BUKAN Android)

> ⚠️ Hasil di bawah adalah latensi pada laptop development menggunakan CPU TFLite runtime. Angka ini TIDAK mewakili performa di Samsung Galaxy Tab A8.

Untuk menjalankan benchmark, gunakan script:
```bash
cd "d:\D4 TEKNIK KOMPUTER\TA\UsgAndroidApp"
python reports\benchmark_latency.py
```

(Script tersedia di `reports/benchmark_latency.py`)

---

## E. Evaluasi EMA — Simulasi Temporal (Tanpa Ground Truth Video)

Karena tidak ada data video/anotasi temporal, evaluasi EMA dilakukan dengan **simulasi sintetis**: membuat trajektori posisi vena yang mensimulasikan gerakan probe (diam → geser pelan → geser cepat → hilang), lalu membandingkan respon EMA statis vs Adaptive EMA.

Script: `reports/eval_ema_simulation.py`

---

## F. Rekomendasi Teknis Berdasarkan Audit

### Prioritas 1 (Langsung actionable):

1. **Fix dead variable** `protoRow0` di `YoloDetector.decodeMask()` (minor, tidak mempengaruhi correctness)
2. **Optimasi loop mask decode** — reorder loop agar akses memori cache-friendly (`proto[k]` per baris py, bukan akses random)
3. **Tambahkan logging waktu inferensi** ke logcat agar FPS aktual di Android dapat diukur

### Prioritas 2 (Perlu data):

4. **Kumpulkan rekaman video USG** + anotasi minimal 50-100 frame untuk evaluasi mAP
5. **Test dengan 3 kondisi probe** yang dapat direkam: diam, geser pelan, geser cepat

### Tidak Direkomendasikan saat ini:

- ❌ Ganti backbone (tidak ada bukti arsitektur jadi bottleneck)
- ❌ Tambah Kalman Filter (perlu data temporal terlebih dahulu untuk membuktikan manfaatnya vs EMA)
- ❌ Retrain model (tidak ada bukti akurasi model saat ini tidak cukup)
- ❌ Quantization INT8 (model sudah cukup kecil; perlu benchmark aktual di Android dulu)

---

## G. Status Audit per Tahap

| Tahap | Status | Keterangan |
|-------|--------|------------|
| A — Audit repository | ✅ Selesai | Laporan ini |
| B — Evaluasi YOLOv8 ± EMA | ⚠️ Parsial | Simulasi saja; tidak ada data ground truth |
| C — Metode lanjutan | 🔜 Menunggu | Perlu hasil B yang valid |
| D — Performa Android | ⚠️ Parsial | Hanya benchmark laptop |
| E — Implementasi & test | ⚠️ Parsial | Tidak ada test otomatis pre-existing |

---

*Laporan ini dibuat berdasarkan inspeksi aktual file repository. Tidak ada angka yang dikarang atau diasumsikan.*


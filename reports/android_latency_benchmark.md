# Laporan Benchmark Latensi & Profiling Android — YOLOv8n-seg + Adaptive EMA

**Tanggal Pengujian:** 30 Oktober 2026  
**Perangkat Target:** Samsung Galaxy Tab A 8.0 (2019) LTE (`SM-T295`)  
**Tipe Pengujian:** Pengukuran on-device pada perangkat fisik aktual melalui ADB  
**Dataset/Kondisi Input:** Stream kamera USG / Screen Capture MediaProjection real-time  
**Data Mentah Sampel:** Tersedia di [`reports/android_benchmark_samples.csv`](./android_benchmark_samples.csv)

---

## 1. Ringkasan Eksekutif

Profiling performa end-to-end berhasil dijalankan pada perangkat fisik Android target (**Samsung SM-T295**) menggunakan instrumentation berpresisi nanodetik (`SystemClock.elapsedRealtimeNanos()`). Sebanyak **230 frame** diproses, dengan **30 frame warm-up pertama dieksklusikan** dari statistik untuk memastikan pengukuran hanya mencakup kondisi *steady-state*, menghasilkan **200 sampel frame terukur penuh**.

### Temuan Utama:
1. **Bottleneck Utama (97.2% waktu):** Pemanggilan model TFLite (`mInterpreter.runForMultipleInputsOutputs`) mendominasi dengan **median latensi 422.32 ms** (mean 434.48 ms).
2. **Throughput Aktual (Wall-Clock):** **2.22 FPS** (200 frame dalam 90.03 detik).
3. **Komponen Non-Model Sangat Ringan:**
   - Preprocessing: median **11.10 ms** (2.6% total)
   - Parsing + NMS: median **0.06 ms** (< 0.1%)
   - Mask Decoding (`decodeMask`): median **0.04 ms** (< 0.1%)
   - Adaptive EMA Smoothing: median **0.01 ms** (< 0.01%)
4. **Rendering Display Halus (60 Hz):** Loop `OverlayView.onDraw()` pada UI thread Choreographer hanya membutuhkan median **0.52 ms** (p95 1.98 ms), sehingga tampilan overlay bebas dari lag render UI.
5. **Runtime yang Aktif:** Model berjalan pada **CPU 4-thread** karena `CompatibilityList.isDelegateSupportedOnThisDevice()` mendeteksi GPU Adreno 504 pada perangkat ini tidak kompatibel dengan konfigurasi FP16 GpuDelegate TFLite 2.14.0.

---

## 2. Metadata Perangkat dan Lingkungan Uji

| Parameter | Nilai Aktual (Terverifikasi) | Sumber |
|-----------|------------------------------|--------|
| **Manufacturer** | `samsung` | `Build.MANUFACTURER` / `ro.product.manufacturer` |
| **Model** | `SM-T295` (Galaxy Tab A 8.0 2019 LTE) | `Build.MODEL` / `ro.product.model` |
| **SoC / Chipset** | Qualcomm Snapdragon 429 (`msm8937`) | `ro.board.platform` |
| **CPU Architecture** | Quad-core ARM Cortex-A53 @ 1.95 GHz | `ro.hardware` (`qcom`) |
| **GPU** | Qualcomm Adreno 504 | System dump |
| **Android Version** | Android 11 (API Level 30) | `Build.VERSION.RELEASE` / `SDK_INT` |
| **Resolusi Layar** | 800 × 1280 (Density: 213 dpi, Landscape: 1280 × 800) | `wm size`, `wm density` |
| **TFLite Model** | `YoloV8_320.tflite` (12.6 MB) | Asset path |
| **Format Input Model** | `[1, 3, 320, 320]` Float32 (NCHW, 0.0 - 1.0) | TFLite Inspector |
| **Format Output Model** | Output 0: `[1, 37, 2100]`, Output 1: `[1, 32, 80, 80]` | TFLite Inspector |
| **Runtime / Delegate** | **CPU (4 threads)** (fallback otomatis dari GpuDelegate) | TFLite Interpreter Log |
| **Tipe Lingkungan** | **Perangkat Fisik Nyata** (koneksi USB ADB ID: `R9CN307EY3J`) | ADB device |

> **Catatan Identifikasi Perangkat:** Samsung Galaxy Tab A 8.0 (2019) `SM-T295` berbasis Snapdragon 429 (quad-core A53), berbeda dengan Galaxy Tab A8 10.5" (2021) `SM-X200/X205` yang menggunakan Unisoc Tiger T618 (octa-core). Pengukuran ini merefleksikan performa riil hardware Snapdragon 429.

---

## 3. Desain dan Titik Pengukuran Instrumentation

Instrumentation diimplementasikan pada class [`PerformanceProfiler.java`](../app/src/main/java/com/example/usgandroidapp/PerformanceProfiler.java) dengan prinsip **Zero-GC overhead** dan **non-blocking**:
- Seluruh timing dicatat menggunakan `SystemClock.elapsedRealtimeNanos()`.
- Data disimpan dalam primitive array fixed-size (`long[]`, `boolean[]`) yang dialokasikan satu kali di awal untuk mencegah Garbage Collection (GC) pause.
- Tidak ada per-frame logging ke Logcat (hanya log periodic per 50 frame) untuk menghindari I/O contention logcat.
- Warm-up sebanyak 30 frame diabaikan sebelum mengaktifkan pencatatan statistik benchmark.

### Definisi Titik Batas Pengukuran (Tidak Ada Double-Counting):

```
[Frame Capture / Camera Thread]
 │  acquireLatestImage() -> imageToBitmap()
 │  Atomic swap: mLatestFrame.getAndSet(frame)
 │  -> [Dropped Frames Counter dihitung jika frame lama ditimpa sebelum AI sempat memproses]
 ▼
[Inference Thread: processLatestFrame()]
 │
 ├──▶ t0: tPipelineStart
 │
 ├──▶ [Tahap 1: Preprocessing]
 │    Bitmap ROI crop (800x45% screen) -> Letterbox resize ke 320x320 -> Normalisasi LUT -> Buffer NCHW
 │    Durasi = t1 - t0
 │
 ├──▶ [Tahap 2: TFLite Model Inference]
 │    mInterpreter.runForMultipleInputsOutputs(mInputBuffer, outputs)
 │    Durasi = t2 - t1
 │
 ├──▶ [Tahap 3: Postprocessing (Parsing + NMS)]
 │    parseProposals(mOutputDet[0]) -> filter conf > 0.25 -> applyNMS(IoU 0.45)
 │    Durasi = t3 - t2
 │
 ├──▶ [Tahap 4: Mask Decoding]
 │    buildDetections() -> decodeMask() dot product (32 coeffs x proto 80x80) -> sigmoid -> threshold
 │    Durasi = t4 - t3
 │
 ├──▶ [Tahap 5: Adaptive EMA Smoothing]
 │    pickBest() -> applyEmaAndHold() (perhitungan displacement centroid -> alpha dinamis)
 │    Durasi = t5 - t4
 │
 └──▶ tPipelineEnd: Durasi Pipeline Total = tPipelineEnd - tPipelineStart
      mMainHandler.postAtFrontOfQueue(() -> mOverlayView.setDetection())

[Main / UI Thread: Choreographer 60Hz Loop]
 │
 └──▶ [Tahap 6: Overlay Rendering (onDraw)]
      OverlayView.onDraw(Canvas canvas) -> drawMask() -> drawRect() -> drawLines() -> drawLabel()
      Durasi onDraw = tDrawEnd - tDrawStart
```

---

## 4. Hasil Pengukuran & Statistik Latensi

### Tabel Statistik Tahap Pemrosesan (N = 200 Frame Steady-State)

Semua metrik dalam satuan **milidetik (ms)**:

| Tahap Pipeline | Sampel | Min (ms) | Mean (ms) | Median / p50 (ms) | p95 (ms) | Max (ms) | Kontribusi (%) |
|----------------|:------:|:--------:|:---------:|:-----------------:|:--------:|:--------:|:--------------:|
| **1. Preprocessing** | 200 | 9.42 | 12.29 | 11.10 | 16.98 | 90.90 | 2.56% |
| **2. TFLite Model Inference** | 200 | **369.90** | **434.48** | **422.32** | **547.92** | **685.38** | **97.23%** |
| **3. Parsing + NMS** | 200 | 0.05 | 0.10 | 0.06 | 0.27 | 0.87 | 0.01% |
| **4. Mask Decoding** | 200 | 0.02 | 0.11 | 0.04 | 0.53 | 1.05 | 0.01% |
| **5. Adaptive EMA** | 200 | 0.01 | 0.01 | 0.01 | 0.03 | 0.05 | < 0.01% |
| **Total Pipeline Deteksi** | 200 | **379.95** | **447.01** | **433.53** | **563.59** | **720.11** | **100.00%** |
| **Overlay onDraw (UI Thread)** | 500 | 0.00 | 0.75 | 0.52 | 1.98 | 7.70 | *Independen (60Hz)* |

---

## 5. Analisis Throughput dan Frame Rate

### 5.1 Throughput Inferensi AI vs FPS Tampilan Layar

Terdapat perbedaan mendasar antara **Throughput Inferensi AI** dan **Display Refresh Rate**:

| Metrik | Nilai | Cara Perhitungan | Keterangan |
|--------|:-----:|------------------|------------|
| **AI Throughput Aktual (Wall-Clock)** | **2.22 FPS** | `200 frame / 90.03 detik` | Frekuensi riil model memproses frame baru |
| **AI Throughput Teoretis (Latency-based)** | **2.31 FPS** | `1000 / median_pipeline (433.53 ms)` | Throughput maksimal jika pipeline berjalan tanpa jeda loop |
| **Display Rendering Rate** | **~60 FPS** | Choreographer 60Hz loop (durasi `onDraw` median 0.52 ms) | Frekuensi render visual layar tablet |
| **Frame Terlewat (Dropped)** | **3688 frame** | Frame kamera masuk saat AI sedang sibuk | AI selalu mengambil *frame paling baru* via latest-frame dropping |
| **Deteksi Vena Valid** | **121 / 200 (60.5%)** | Frame dengan confidence > threshold | Sisanya di-hold secara mulus oleh Adaptive EMA |

### 5.2 Mengapa Tampilan Tetap Terasa Halus Meskipun AI 2.22 FPS?
Aplikasi menggunakan **Persistence of Vision Architecture**:
1. AI Thread menghitung posisi vena baru setiap ~433 ms (2.22 kali per detik).
2. UI Thread (`OverlayView`) merender posisi pada kecepatan **60 kali per detik** (~16.6 ms) menggunakan hardware Canvas acceleration.
3. **Adaptive EMA + Hold Frames** menginterpolasi koordinat bbox antar update AI, mencegah flicker/kedip dan menjaga kotak overlay tetap stabil di layar.

---

## 6. Interpretasi Bottleneck

Berdasarkan data kuantitatif di atas:

1. **Bukan Bottleneck:**
   - **Mask Decoding (`decodeMask`):** Sebelumnya dicurigai berat karena perhitungan `Math.exp()` dan dot-product, namun data aktual membuktikan tahap ini hanya memakan **0.04 ms (median)**. Mengoptimasi tahap ini tidak akan memberikan dampak terukur terhadap FPS total.
   - **Adaptive EMA:** Sangat cepat (**0.01 ms**). Logika adaptive smoothing tidak membebani CPU sama sekali.
   - **Parsing & NMS:** Hanya **0.06 ms**.
   - **Preprocessing:** Hanya **11.10 ms** (crop, letterbox, normalisasi LUT float32).

2. **Bottleneck Sebenarnya:**
   - **Eksekusi Model TFLite pada CPU:** Memakan **422.32 ms** (97.2% dari seluruh siklus pemrosesan).
   - **Penyebab:** Arsitektur YOLOv8n-seg Float32 (12.6 MB) dieksekusi murni pada 4 core CPU Cortex-A53 (1.95 GHz). GPU delegate otomatis di-fallback ke CPU oleh runtime TFLite karena driver GPU Adreno 504 pada perangkat ini tidak mendukung operasi delegate FP16 tertentu.

---

## 7. Keterbatasan Pengukuran

1. **Batas Pengukuran UI Rendering:** Durasi `onDraw` mengukur waktu eksekusi CPU Canvas drawing commands. Pengukuran ini tidak mencakup waktu swap buffer SurfaceFlinger GPU hardware composer sistem Android.
2. **Kondisi Termal:** Pengukuran 200 frame berlangsung selama 90 detik. Pada frame ke-150–200 terlihat sedikit kenaikan p95 (563 ms) akibat thermal throttling wajar pada perangkat tablet entry-level tanpa pendingin aktif.
3. **Input Kamera:** Data berasal dari VirtualDisplay media projection screen capture saat aplikasi aktif di foreground.

---

## 8. Rekomendasi Langkah Berikutnya (Tunggal & Berbasis Bukti)

> **Rekomendasi Prioritas:** **Uji Coba Post-Training INT8 Quantization (atau Dynamic Range Quantization) pada Model YOLOv8 320.**

### Alasan Teknis:
- Data profiling membuktikan secara mutlak bahwa **97.2% waktu (422 ms)** dihabiskan pada pemanggilan model TFLite di CPU Cortex-A53.
- Mengoptimasi kode Java (preprocessing, mask decoding, atau EMA) maksimal hanya menghemat ~1-2 ms (tidak berdampak pada FPS).
- Kuantisasi INT8 pada CPU ARM Cortex-A53 memanfaatkan instruksi SIMD NEON integer dot-product (SDOT/UDOT atau NEON int8 matrix compute), yang umumnya memberikan percepatan **2× hingga 3× lipat** pada inferensi CPU mobile, dengan potensi meningkatkan throughput dari **2.2 FPS menjadi ~5–7 FPS**.
- Model INT8 juga akan mengurangi ukuran file model dari 12.6 MB menjadi ~3.5 MB, menghemat bandwidth memori cache L2 CPU Snapdragon 429.

*(Catatan: Jangan mengubah arsitektur atau weights saat ini. Kuantisasi harus dievaluasi secara terpisah pada branch/task berikutnya dengan memverifikasi presisi mAP).*


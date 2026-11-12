"""
benchmark_latency.py
====================
Mengukur latensi inferensi model TFLite di environment saat ini (laptop/PC).

PERINGATAN: Hasil ini adalah latensi pada CPU laptop, BUKAN di Samsung Galaxy Tab A8.
            Gunakan hasil ini untuk perbandingan relatif antar model, bukan sebagai
            estimasi performa Android.

Cara menjalankan:
    cd "d:\\D4 TEKNIK KOMPUTER\\TA\\UsgAndroidApp"
    python reports/benchmark_latency.py

Output: reports/benchmark_latency_results.json + tampilan di terminal.
"""

import time
import json
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'src', 'main', 'assets')
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'benchmark_latency_results.json')
WARMUP_RUNS = 3
BENCH_RUNS  = 20

MODELS = {
    'YoloV8_320.tflite' : {'input_shape': (1, 3, 320, 320), 'dtype': np.float32},
    'YoloV26_320.tflite': {'input_shape': (1, 3, 320, 320), 'dtype': np.float32},
    'best.tflite'       : {'input_shape': (1, 3, 640, 640), 'dtype': np.float32},
    'yoloV8.tflite'     : {'input_shape': (1, 3, 640, 640), 'dtype': np.float32},
}

def benchmark_model(model_path, input_shape, dtype):
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()

        inp_idx  = interp.get_input_details()[0]['index']
        dummy    = np.random.rand(*input_shape).astype(dtype)

        # Warmup
        for _ in range(WARMUP_RUNS):
            interp.set_tensor(inp_idx, dummy)
            interp.invoke()

        # Benchmark
        times = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            interp.set_tensor(inp_idx, dummy)
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        return {
            'mean_ms'  : round(float(np.mean(times)), 2),
            'median_ms': round(float(np.median(times)), 2),
            'min_ms'   : round(float(np.min(times)), 2),
            'max_ms'   : round(float(np.max(times)), 2),
            'std_ms'   : round(float(np.std(times)), 2),
            'est_fps'  : round(1000 / float(np.mean(times)), 1),
            'runs'     : BENCH_RUNS,
            'status'   : 'ok',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def main():
    print("=" * 60)
    print("  TFLite Model Latency Benchmark")
    print("  Environment: Laptop CPU (BUKAN Android)")
    print(f"  Warmup: {WARMUP_RUNS} runs | Benchmark: {BENCH_RUNS} runs")
    print("=" * 60)

    results = {}
    for name, cfg in MODELS.items():
        path = os.path.join(ASSETS_DIR, name)
        if not os.path.exists(path):
            print(f"  [{name}] SKIP — file tidak ditemukan")
            results[name] = {'status': 'file_not_found'}
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\n  Benchmarking: {name} ({size_mb:.1f} MB) ...", end='', flush=True)
        r = benchmark_model(path, cfg['input_shape'], cfg['dtype'])
        results[name] = {**r, 'size_mb': round(size_mb, 2), 'input_shape': list(cfg['input_shape'])}

        if r['status'] == 'ok':
            print(f"\r  [{name}] {r['mean_ms']:.1f} ms/frame avg | "
                  f"min={r['min_ms']:.1f} max={r['max_ms']:.1f} | "
                  f"~{r['est_fps']} FPS (laptop)")
        else:
            print(f"\r  [{name}] ERROR: {r.get('error', '?')}")

    print("\n" + "=" * 60)
    print("  RINGKASAN PERBANDINGAN RELATIF")
    print("  (Angka ini BUKAN performa Android)")
    print("=" * 60)
    ok = {k: v for k, v in results.items() if v.get('status') == 'ok'}
    if ok:
        fastest = min(ok, key=lambda k: ok[k]['mean_ms'])
        for name, r in sorted(ok.items(), key=lambda x: x[1]['mean_ms']):
            ratio = r['mean_ms'] / ok[fastest]['mean_ms']
            print(f"  {name:25s}: {r['mean_ms']:7.1f} ms  ({ratio:.1f}x)")

    # Simpan hasil
    out = {
        'meta': {
            'date'       : time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': 'laptop_cpu',
            'warning'    : 'Hasil ini bukan performa Android. Gunakan untuk perbandingan relatif saja.',
            'warmup_runs': WARMUP_RUNS,
            'bench_runs' : BENCH_RUNS,
        },
        'results': results
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Hasil disimpan: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()


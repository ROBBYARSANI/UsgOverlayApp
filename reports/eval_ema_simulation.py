"""
eval_ema_simulation.py
======================
Evaluasi perbandingan EMA statis vs Adaptive EMA menggunakan trajektori vena
yang disimulasikan secara sintetis.

TUJUAN:
  Karena tidak ada data video/anotasi temporal di repository ini, evaluasi
  dilakukan dengan mensimulasikan berbagai kondisi gerakan probe USG dan
  mengukur respon algoritma smoothing secara matematis.

KONDISI YANG DISIMULASIKAN:
  1. Probe diam (noise kecil)
  2. Probe bergerak pelan
  3. Probe bergerak cepat
  4. Probe bergerak lalu berhenti
  5. Deteksi hilang sementara (miss frames)
  6. Objek berpindah posisi mendadak

METRIK:
  - MAE (Mean Absolute Error) posisi centroid vs ground truth
  - Jitter (std dev perubahan posisi antar frame)
  - Lag (delay rata-rata mengikuti pergerakan)
  - Konvergensi (berapa frame untuk mencapai 95% target)

PERINGATAN: Simulasi ini menggunakan trajektori sintetis, BUKAN data USG nyata.
            Hasil tidak dapat dijadikan klaim akurasi terhadap data klinisi.

Cara menjalankan:
    cd "d:\\D4 TEKNIK KOMPUTER\\TA\\UsgAndroidApp"
    python reports/eval_ema_simulation.py

Output: reports/eval_ema_simulation_results.json + plot (jika matplotlib tersedia).
"""

import json
import math
import os
import time
import random

# ─────────────────────────────────────────────────────────────────
# KONSTANTA EMA (sama persis dengan VeinDetectionService.java)
# ─────────────────────────────────────────────────────────────────
EMA_CONFIGS = {
    'no_ema': {
        'description': 'Tanpa EMA — output raw model',
        'alpha_still': 1.0,
        'alpha_fast' : 1.0,
        'move_thresh': 0.0,
        'jump_thresh': 1.0,
    },
    'static_ema_020': {
        'description': 'EMA statis alpha=0.20 (nilai ALPHA_STILL saat ini)',
        'alpha_still': 0.20,
        'alpha_fast' : 0.20,
        'move_thresh': 0.0,
        'jump_thresh': 9999.0,
    },
    'static_ema_055': {
        'description': 'EMA statis alpha=0.55 (nilai lama sebelum adaptive)',
        'alpha_still': 0.55,
        'alpha_fast' : 0.55,
        'move_thresh': 0.0,
        'jump_thresh': 9999.0,
    },
    'adaptive_ema_current': {
        'description': 'Adaptive EMA — implementasi aktual (ALPHA_STILL=0.20, ALPHA_FAST=1.0, MOVE=12, JUMP=70)',
        'alpha_still': 0.20,
        'alpha_fast' : 1.00,
        'move_thresh': 12.0,
        'jump_thresh': 70.0,
    },
}

MAX_HOLD = 6  # MAX_HOLD dari VeinDetectionService

# ─────────────────────────────────────────────────────────────────
# GENERATOR TRAJEKTORI
# ─────────────────────────────────────────────────────────────────

def gen_still(n_frames, cx0=300.0, cy0=200.0, noise=2.0):
    """Probe diam — posisi relatif konstan dengan noise kecil."""
    rng = random.Random(42)
    traj = []
    for _ in range(n_frames):
        traj.append((
            cx0 + rng.gauss(0, noise),
            cy0 + rng.gauss(0, noise),
        ))
    return traj

def gen_slow_move(n_frames, cx0=300.0, cy0=200.0, vx=1.0, vy=0.5, noise=2.0):
    """Probe bergerak pelan."""
    rng = random.Random(42)
    traj = []
    for i in range(n_frames):
        traj.append((
            cx0 + vx * i + rng.gauss(0, noise),
            cy0 + vy * i + rng.gauss(0, noise),
        ))
    return traj

def gen_fast_move(n_frames, cx0=300.0, cy0=200.0, vx=5.0, vy=2.5, noise=3.0):
    """Probe bergerak cepat."""
    rng = random.Random(42)
    traj = []
    for i in range(n_frames):
        traj.append((
            cx0 + vx * i + rng.gauss(0, noise),
            cy0 + vy * i + rng.gauss(0, noise),
        ))
    return traj

def gen_move_then_stop(n_frames, cx0=300.0, cy0=200.0, vx=4.0, noise=2.0):
    """Probe bergerak lalu berhenti di tengah."""
    rng = random.Random(42)
    traj = []
    half = n_frames // 2
    for i in range(n_frames):
        if i < half:
            cx = cx0 + vx * i + rng.gauss(0, noise)
        else:
            cx = cx0 + vx * half + rng.gauss(0, noise * 0.5)
        traj.append((cx, cy0 + rng.gauss(0, noise)))
    return traj

def gen_with_misses(n_frames, miss_prob=0.3, noise=2.0, cx0=300.0, cy0=200.0):
    """Deteksi sering hilang (confidence rendah / vena tidak terdeteksi)."""
    rng = random.Random(42)
    traj = []
    for _ in range(n_frames):
        if rng.random() < miss_prob:
            traj.append(None)  # miss frame
        else:
            traj.append((cx0 + rng.gauss(0, noise), cy0 + rng.gauss(0, noise)))
    return traj

def gen_sudden_jump(n_frames, cx0=300.0, cy0=200.0, jump_cx=450.0, jump_cy=300.0,
                    jump_at=30, noise=2.0):
    """Vena berpindah posisi mendadak (probe digeser jauh sekaligus)."""
    rng = random.Random(42)
    traj = []
    for i in range(n_frames):
        if i < jump_at:
            traj.append((cx0 + rng.gauss(0, noise), cy0 + rng.gauss(0, noise)))
        else:
            traj.append((jump_cx + rng.gauss(0, noise), jump_cy + rng.gauss(0, noise)))
    return traj

# ─────────────────────────────────────────────────────────────────
# IMPLEMENTASI EMA (Python — mirroring VeinDetectionService.java)
# ─────────────────────────────────────────────────────────────────

def run_ema(trajectory, cfg, max_hold=MAX_HOLD):
    """
    Jalankan EMA + hold frames pada trajektori.
    Trajectory item: (cx, cy) atau None (miss frame).
    Returns: list of smoothed (cx, cy) per frame, atau None jika overlay belum init.
    """
    alpha_still = cfg['alpha_still']
    alpha_fast  = cfg['alpha_fast']
    move_thresh = cfg['move_thresh']
    jump_thresh = cfg['jump_thresh']

    sm_cx, sm_cy = None, None
    last_valid   = None
    hold_count   = 0
    output       = []

    for raw in trajectory:
        if raw is not None:
            if sm_cx is None:
                # Inisialisasi pertama
                sm_cx, sm_cy = raw[0], raw[1]
            else:
                dx   = raw[0] - sm_cx
                dy   = raw[1] - sm_cy
                disp = math.sqrt(dx*dx + dy*dy)
                t    = max(0.0, min(1.0, (disp - move_thresh) / max(1e-6, jump_thresh - move_thresh)))
                alpha = alpha_still + t * (alpha_fast - alpha_still)
                sm_cx += alpha * (raw[0] - sm_cx)
                sm_cy += alpha * (raw[1] - sm_cy)

            last_valid  = (sm_cx, sm_cy)
            hold_count  = max_hold
            output.append((sm_cx, sm_cy))

        else:
            # Miss frame
            if hold_count > 0 and last_valid is not None:
                hold_count -= 1
                output.append(last_valid)
            else:
                sm_cx, sm_cy = None, None
                last_valid   = None
                output.append(None)

    return output


# ─────────────────────────────────────────────────────────────────
# METRIK
# ─────────────────────────────────────────────────────────────────

def compute_metrics(gt_traj, smoothed, name):
    """
    Hitung metrik temporal dari ground truth dan output EMA.
    gt_traj: list (cx, cy) — ground truth posisi per frame
    smoothed: list (cx, cy) atau None — output EMA
    """
    errors   = []
    jitter   = []
    prev     = None
    miss_count = 0

    for gt, sm in zip(gt_traj, smoothed):
        if sm is None:
            miss_count += 1
            prev = None
            continue
        if gt is None:
            # GT frame missing — skip
            continue

        # MAE
        err = math.sqrt((sm[0] - gt[0])**2 + (sm[1] - gt[1])**2)
        errors.append(err)

        # Jitter (perubahan posisi antar frame)
        if prev is not None:
            j = math.sqrt((sm[0] - prev[0])**2 + (sm[1] - prev[1])**2)
            jitter.append(j)
        prev = sm

    def safe_mean(lst): return sum(lst)/len(lst) if lst else float('nan')
    def safe_std(lst):
        if len(lst) < 2: return float('nan')
        m = safe_mean(lst)
        return math.sqrt(sum((x-m)**2 for x in lst) / len(lst))

    n_total = len([g for g in gt_traj if g is not None])
    return {
        'config'       : name,
        'mae_px'       : round(safe_mean(errors), 3),
        'jitter_mean_px': round(safe_mean(jitter), 3),
        'jitter_std_px' : round(safe_std(jitter), 3),
        'miss_frames'  : miss_count,
        'miss_rate_pct': round(100 * miss_count / max(1, len(smoothed)), 1),
        'n_frames_eval': len(errors),
    }


# ─────────────────────────────────────────────────────────────────
# EVALUASI KONVERGENSI
# ─────────────────────────────────────────────────────────────────

def convergence_frames(alpha_still, alpha_fast, move_thresh, jump_thresh,
                       start_cx=300.0, target_cx=450.0, threshold_pct=0.95):
    """
    Berapa frame untuk EMA mencapai 95% dari target setelah lompat mendadak?
    Gunakan displacement besar agar alpha = alpha_fast.
    """
    sm_cx = start_cx
    target = target_cx
    threshold = start_cx + threshold_pct * (target - start_cx)

    for frame in range(200):
        dx = target - sm_cx
        disp = abs(dx)
        t = max(0.0, min(1.0, (disp - move_thresh) / max(1e-6, jump_thresh - move_thresh)))
        alpha = alpha_still + t * (alpha_fast - alpha_still)
        sm_cx += alpha * (target - sm_cx)
        if sm_cx >= threshold:
            return frame + 1
    return '>200'


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  EMA Simulation Evaluation")
    print("  PERINGATAN: Simulasi sintetis — bukan data USG nyata")
    print("=" * 68)

    N = 60  # jumlah frame per skenario

    scenarios = {
        'probe_diam'         : gen_still(N),
        'probe_geser_pelan'  : gen_slow_move(N),
        'probe_geser_cepat'  : gen_fast_move(N),
        'gerak_lalu_berhenti': gen_move_then_stop(N),
        'miss_30pct'         : gen_with_misses(N, miss_prob=0.30),
        'lompat_mendadak'    : gen_sudden_jump(N, jump_at=30),
    }

    all_results = []

    for scenario_name, gt_traj in scenarios.items():
        print(f"\n  Skenario: {scenario_name}")
        print(f"  {'Config':<28} {'MAE':>8} {'Jitter':>8} {'Miss%':>7}")
        print("  " + "-" * 55)

        # Untuk skenario miss, GT adalah posisi statis (bukan None)
        gt_for_metrics = [
            (300.0, 200.0) if g is None else g
            for g in gt_traj
        ]

        for cfg_name, cfg in EMA_CONFIGS.items():
            smoothed = run_ema(gt_traj, cfg)
            m = compute_metrics(gt_for_metrics, smoothed, cfg_name)
            m['scenario'] = scenario_name
            all_results.append(m)
            print(f"  {cfg_name:<28} {m['mae_px']:>8.2f} {m['jitter_mean_px']:>8.2f} {m['miss_rate_pct']:>6.0f}%")

    # Analisis konvergensi
    print("\n" + "=" * 68)
    print("  ANALISIS KONVERGENSI (lompat posisi mendadak, threshold 95%)")
    print(f"  {'Config':<28} {'Frames ke 95%':>15}")
    print("  " + "-" * 45)
    conv_results = {}
    for cfg_name, cfg in EMA_CONFIGS.items():
        frames = convergence_frames(
            cfg['alpha_still'], cfg['alpha_fast'],
            cfg['move_thresh'], cfg['jump_thresh']
        )
        conv_results[cfg_name] = frames
        print(f"  {cfg_name:<28} {str(frames):>15}")

    # Simpan hasil
    output = {
        'meta': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'synthetic_simulation',
            'warning': 'Simulasi sintetis. Bukan data USG nyata. Tidak dapat diklaim sebagai akurasi klinis.',
            'n_frames_per_scenario': N,
            'max_hold_frames': MAX_HOLD,
        },
        'ema_configs': EMA_CONFIGS,
        'scenario_results': all_results,
        'convergence_frames': conv_results,
        'interpretation': {
            'mae_px': 'Rata-rata error posisi centroid (piksel) vs ground truth. Lebih kecil = lebih akurat posisi.',
            'jitter_mean_px': 'Rata-rata perubahan posisi antar frame (piksel). Lebih kecil = lebih halus, tapi bisa berarti lag.',
            'miss_rate_pct': 'Persentase frame di mana overlay hilang (melebihi hold frames). 0% = overlay selalu tampil.',
            'convergence_frames': 'Berapa frame untuk mencapai 95% target setelah posisi berubah mendadak. Lebih kecil = lebih responsif.',
        }
    }
    out_path = os.path.join(os.path.dirname(__file__), 'eval_ema_simulation_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Hasil disimpan: {out_path}")
    print("\n  INTERPRETASI:")
    print("  - MAE kecil + jitter kecil = ideal (akurat DAN stabil)")
    print("  - EMA alpha=1.0 (no_ema): MAE minimal tapi jitter tinggi (mengikuti noise)")
    print("  - EMA alpha kecil: jitter rendah tapi MAE tinggi + lag saat bergerak")
    print("  - Adaptive EMA: kompromi — jitter rendah saat diam, responsif saat bergerak")


if __name__ == '__main__':
    main()


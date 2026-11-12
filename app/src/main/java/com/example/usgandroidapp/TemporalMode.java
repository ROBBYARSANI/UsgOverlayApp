package com.example.usgandroidapp;

/**
 * TemporalMode.java
 * =================
 * Mode pemrosesan temporal untuk stabilisasi deteksi dan pelacakan vena USG.
 *
 * MODES:
 *   RAW             : YOLO mentah tanpa filter temporal (Sample & Hold).
 *   EMA             : YOLO + Adaptive Exponential Moving Average (Baseline lama).
 *   LK_EXPERIMENTAL : YOLO + Pyramidal Lucas-Kanade Optical Flow (Mode eksperimental).
 *   KALMAN          : YOLO + Bounded Constant-Velocity Kalman Filter (Default produksi baru).
 */
public enum TemporalMode {
    RAW,
    MINIMAL_ANTIBLINK,
    EMA,
    LK_EXPERIMENTAL,
    KALMAN
}


package com.example.usgandroidapp;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * VeinOpticalFlowTracker.java
 * ===========================
 * Pyramidal Lucas-Kanade Optical Flow Tracker untuk Continuous Vein Tracking & Flicker-Free Overlay.
 *
 * ARSITEKTUR:
 *   1. Berjalan pada frame kamera di antara inferensi YOLO (~24-30 FPS).
 *   2. Forward-Backward (FB) error check untuk membuang poin pelacakan yang tidak stabil.
 *   3. Robust motion estimation (RANSAC estimateAffinePartial2D dengan fallback ke median delta).
 *   4. Pergeseran kontinyu BBox & Segmentation Mask tanpa visual snapping / teleporting.
 *   5. YOLO sebagai korektor periodik (koreksi halus 3-5 frame untuk drift kecil, reset instan untuk drift besar).
 *   6. State Machine: FRESH_YOLO, TRACKED, DEGRADED, LOST (bebas blinking, tanpa ghost vein).
 */
public class VeinOpticalFlowTracker {

    private static final String TAG = "VeinOFTracker";

    // Static initialization of OpenCV
    private static volatile boolean sOpenCvInitialized = false;

    public static synchronized boolean initOpenCV() {
        if (!sOpenCvInitialized) {
            try {
                if (OpenCVLoader.initLocal()) {
                    sOpenCvInitialized = true;
                    Log.i(TAG, "OpenCV initialized successfully via initLocal()");
                } else if (OpenCVLoader.initDebug()) {
                    sOpenCvInitialized = true;
                    Log.i(TAG, "OpenCV initialized successfully via initDebug()");
                } else {
                    Log.e(TAG, "OpenCV initialization failed!");
                }
            } catch (Throwable t) {
                Log.e(TAG, "OpenCVLoader exception: " + t.getMessage(), t);
            }
        }
        return sOpenCvInitialized;
    }

    public enum TrackState {
        FRESH_YOLO,
        TRACKED,
        DEGRADED,
        LOST
    }

    public static class TrackedVein {
        public TrackState state = TrackState.LOST;
        public final RectF bbox = new RectF();
        public float cx = 0f;
        public float cy = 0f;
        public float confidence = 0f;
        public float depthMm = 0f;
        public Bitmap maskBitmap = null;
        public final Matrix maskMatrix = new Matrix();

        public int numInputPoints = 0;
        public int numValidPoints = 0;
        public float fbErrorPx = 0f;
        public float lkErrorPx = 0f;

        public long captureTimestampNs = 0L;
        public long updateTimestampNs = 0L;
        public float predAgeMs = 0f;
        public boolean isFreshYolo = false;
        public float rawConfidence = 0f;

        public TrackedVein copy() {
            TrackedVein t = new TrackedVein();
            t.state = this.state;
            t.bbox.set(this.bbox);
            t.cx = this.cx;
            t.cy = this.cy;
            t.confidence = this.confidence;
            t.depthMm = this.depthMm;
            t.maskBitmap = this.maskBitmap;
            t.maskMatrix.set(this.maskMatrix);
            t.numInputPoints = this.numInputPoints;
            t.numValidPoints = this.numValidPoints;
            t.fbErrorPx = this.fbErrorPx;
            t.lkErrorPx = this.lkErrorPx;
            t.captureTimestampNs = this.captureTimestampNs;
            t.updateTimestampNs = this.updateTimestampNs;
            t.predAgeMs = this.predAgeMs;
            t.isFreshYolo = this.isFreshYolo;
            t.rawConfidence = this.rawConfidence;
            return t;
        }
    }

    // Parameters
    private static final int MAX_FEATURE_POINTS = 40;
    private static final int MIN_POINTS_TRACKED = 7;
    private static final int MIN_POINTS_DEGRADED = 3;
    private static final float MAX_FB_ERROR_PX = 1.6f;
    private static final float MAX_STEP_TRANSLATION_PX = 45.0f;
    private static final float CORRECTION_DISAGREEMENT_SNAP_PX = 45.0f;
    private static final int CORRECTION_INTERPOLATION_STEPS = 4;
    private static final long MAX_DEGRADED_DURATION_MS = 500L;

    // OpenCV Reusable Objects (Zero Allocation in Tracking Loop)
    private Mat mPrevGray = null;
    private Mat mCurrGray = null;
    private Mat mRoiMask = null;
    private Mat mRgbaMat = null;

    private final MatOfPoint2f mPoints0 = new MatOfPoint2f();
    private final MatOfPoint2f mPoints1 = new MatOfPoint2f();
    private final MatOfPoint2f mPoints0Back = new MatOfPoint2f();
    private final MatOfByte mStatusFwd = new MatOfByte();
    private final MatOfByte mStatusBwd = new MatOfByte();
    private final MatOfFloat mErrFwd = new MatOfFloat();
    private final MatOfFloat mErrBwd = new MatOfFloat();
    private final Mat mInliers = new Mat();

    private final Size mWinSize = new Size(17, 17);
    private final TermCriteria mTermCriteria = new TermCriteria(
            TermCriteria.COUNT + TermCriteria.EPS, 15, 0.03);

    // Current Tracking State
    private TrackState mState = TrackState.LOST;
    private final RectF mCurrentBBox = new RectF();
    private float mCurrentCx = 0f;
    private float mCurrentCy = 0f;
    private float mCurrentConfidence = 0f;
    private float mCurrentDepthMm = 0f;
    private Bitmap mReferenceMask = null;
    private final Matrix mAccumulatedMatrix = new Matrix();

    private long mLastYoloCaptureTsNs = 0L;
    private long mLastDegradedTimeMs = 0L;
    private int mConsecutiveDegradedFrames = 0;

    // Visual Correction Interpolation
    private int mRemainingCorrectionSteps = 0;
    private float mCorrectionStepX = 0f;
    private float mCorrectionStepY = 0f;

    // Quality Stats
    private int mNumInputPoints = 0;
    private int mNumValidPoints = 0;
    private float mFbErrorPx = 0f;
    private float mLkErrorPx = 0f;
    private float mRawYoloConfidence = 0f;

    private int mFrameWidth = 0;
    private int mFrameHeight = 0;

    public VeinOpticalFlowTracker() {
        initOpenCV();
    }

    public synchronized void cleanup() {
        if (mPrevGray != null) { mPrevGray.release(); mPrevGray = null; }
        if (mCurrGray != null) { mCurrGray.release(); mCurrGray = null; }
        if (mRoiMask != null) { mRoiMask.release(); mRoiMask = null; }
        if (mRgbaMat != null) { mRgbaMat.release(); mRgbaMat = null; }
        mPoints0.release();
        mPoints1.release();
        mPoints0Back.release();
        mStatusFwd.release();
        mStatusBwd.release();
        mErrFwd.release();
        mErrBwd.release();
        mInliers.release();
        mState = TrackState.LOST;
    }

    /**
     * Memproses frame kamera baru di antara inferensi YOLO (~24-30 FPS).
     * Mengeksekusi Lucas-Kanade forward-backward optical flow dan memperbarui koordinat.
     */
    public synchronized TrackedVein onCameraFrame(Bitmap frameBitmap, long captureTimestampNs) {
        if (!sOpenCvInitialized || frameBitmap == null || frameBitmap.isRecycled()) {
            return getSnapshot(captureTimestampNs);
        }

        mFrameWidth = frameBitmap.getWidth();
        mFrameHeight = frameBitmap.getHeight();

        // Siapkan Mat penampung
        if (mCurrGray == null || mCurrGray.cols() != mFrameWidth || mCurrGray.rows() != mFrameHeight) {
            if (mCurrGray != null) mCurrGray.release();
            if (mPrevGray != null) mPrevGray.release();
            if (mRoiMask != null) mRoiMask.release();
            if (mRgbaMat != null) mRgbaMat.release();

            mCurrGray = new Mat(mFrameHeight, mFrameWidth, CvType.CV_8UC1);
            mPrevGray = new Mat(mFrameHeight, mFrameWidth, CvType.CV_8UC1);
            mRoiMask = new Mat(mFrameHeight, mFrameWidth, CvType.CV_8UC1);
            mRgbaMat = new Mat(mFrameHeight, mFrameWidth, CvType.CV_8UC4);
        }

        // Convert Bitmap ke Grayscale Mat
        Utils.bitmapToMat(frameBitmap, mRgbaMat);
        Imgproc.cvtColor(mRgbaMat, mCurrGray, Imgproc.COLOR_RGBA2GRAY);

        // Jika tracker belum memiliki track aktif (LOST), tunggu YOLO
        if (mState == TrackState.LOST || mPoints0.empty()) {
            mCurrGray.copyTo(mPrevGray);
            return getSnapshot(captureTimestampNs);
        }

        long t0 = SystemClock.elapsedRealtimeNanos();

        // 1. Forward Optical Flow (mPrevGray -> mCurrGray)
        Video.calcOpticalFlowPyrLK(
                mPrevGray, mCurrGray,
                mPoints0, mPoints1,
                mStatusFwd, mErrFwd,
                mWinSize, 2, mTermCriteria
        );

        // 2. Backward Optical Flow (mCurrGray -> mPrevGray)
        Video.calcOpticalFlowPyrLK(
                mCurrGray, mPrevGray,
                mPoints1, mPoints0Back,
                mStatusBwd, mErrBwd,
                mWinSize, 2, mTermCriteria
        );

        // 3. Forward-Backward Error Filter & Point Validation
        Point[] pts0Arr = mPoints0.toArray();
        Point[] pts1Arr = mPoints1.toArray();
        Point[] pts0BackArr = mPoints0Back.toArray();
        byte[] stFwd = mStatusFwd.toArray();
        byte[] stBwd = mStatusBwd.toArray();
        float[] errFwdArr = mErrFwd.toArray();

        mNumInputPoints = pts0Arr.length;
        List<Point> validPts0 = new ArrayList<>();
        List<Point> validPts1 = new ArrayList<>();
        List<Float> fbErrors = new ArrayList<>();
        List<Float> lkErrors = new ArrayList<>();

        for (int i = 0; i < pts0Arr.length; i++) {
            if (i < stFwd.length && stFwd[i] == 1 && i < stBwd.length && stBwd[i] == 1) {
                double dx = pts0Arr[i].x - pts0BackArr[i].x;
                double dy = pts0Arr[i].y - pts0BackArr[i].y;
                float fbErr = (float) Math.sqrt(dx * dx + dy * dy);

                if (fbErr <= MAX_FB_ERROR_PX) {
                    validPts0.add(pts0Arr[i]);
                    validPts1.add(pts1Arr[i]);
                    fbErrors.add(fbErr);
                    if (i < errFwdArr.length) {
                        lkErrors.add(errFwdArr[i]);
                    }
                }
            }
        }

        mNumValidPoints = validPts1.size();
        mFbErrorPx = calculateMedian(fbErrors);
        mLkErrorPx = calculateMedian(lkErrors);

        // 4. Robust Motion Estimation (RANSAC Affine Partial 2D / Robust Median Translation)
        boolean motionAccepted = false;
        float shiftX = 0f;
        float shiftY = 0f;
        float scale = 1.0f;

        if (mNumValidPoints >= MIN_POINTS_DEGRADED) {
            MatOfPoint2f p0MatValid = new MatOfPoint2f();
            MatOfPoint2f p1MatValid = new MatOfPoint2f();
            p0MatValid.fromList(validPts0);
            p1MatValid.fromList(validPts1);

            Mat affine = null;
            try {
                if (mNumValidPoints >= 6) {
                    affine = Calib3d.estimateAffinePartial2D(p0MatValid, p1MatValid, mInliers, Calib3d.RANSAC, 2.0);
                }
            } catch (Exception e) {
                affine = null;
            }

            if (affine != null && affine.rows() == 2 && affine.cols() == 3) {
                double a00 = affine.get(0, 0)[0];
                double a01 = affine.get(0, 1)[0];
                double tx = affine.get(0, 2)[0];
                double a10 = affine.get(1, 0)[0];
                double a11 = affine.get(1, 1)[0];
                double ty = affine.get(1, 2)[0];

                double s = Math.sqrt(a00 * a00 + a10 * a10);
                double thetaDeg = Math.abs(Math.toDegrees(Math.atan2(a10, a00)));

                // Verifikasi batas fisik pergerakan probe USG
                if (s >= 0.85 && s <= 1.15 && thetaDeg <= 18.0) {
                    // Koreksi transformasi affine relatif terhadap pusat centroid vena (eliminasi bias titik nol)
                    double cxShift = (a00 - 1.0) * mCurrentCx + a01 * mCurrentCy + tx;
                    double cyShift = a10 * mCurrentCx + (a11 - 1.0) * mCurrentCy + ty;
                    double tNorm = Math.sqrt(cxShift * cxShift + cyShift * cyShift);

                    if (tNorm <= MAX_STEP_TRANSLATION_PX) {
                        shiftX = (float) cxShift;
                        shiftY = (float) cyShift;
                        scale = (float) s;
                        motionAccepted = true;
                    }
                }
                affine.release();
            }

            // Fallback ke Median Translation jika affine ditolak atau poin sedikit
            if (!motionAccepted) {
                List<Float> dxList = new ArrayList<>();
                List<Float> dyList = new ArrayList<>();
                for (int i = 0; i < validPts1.size(); i++) {
                    dxList.add((float)(validPts1.get(i).x - validPts0.get(i).x));
                    dyList.add((float)(validPts1.get(i).y - validPts0.get(i).y));
                }
                float medDx = calculateMedian(dxList);
                float medDy = calculateMedian(dyList);
                float tNorm = (float) Math.sqrt(medDx * medDx + medDy * medDy);

                if (tNorm <= MAX_STEP_TRANSLATION_PX) {
                    shiftX = medDx;
                    shiftY = medDy;
                    scale = 1.0f;
                    motionAccepted = true;
                }
            }

            p0MatValid.release();
            p1MatValid.release();
        }

        // 5. Update Bounding Box & Transformation State
        long nowMs = SystemClock.uptimeMillis();

        if (motionAccepted) {
            // Apply visual correction step jika ada pending correction dari YOLO
            if (mRemainingCorrectionSteps > 0) {
                shiftX += mCorrectionStepX;
                shiftY += mCorrectionStepY;
                mRemainingCorrectionSteps--;
            }

            // Geser Bounding Box
            float w = mCurrentBBox.width() * scale;
            float h = mCurrentBBox.height() * scale;
            mCurrentCx += shiftX;
            mCurrentCy += shiftY;
            mCurrentBBox.set(mCurrentCx - w / 2f, mCurrentCy - h / 2f,
                             mCurrentCx + w / 2f, mCurrentCy + h / 2f);

            // Batasi agar tidak keluar layar
            clampBBoxToScreen();

            // Akumulasi transformasi mask
            mAccumulatedMatrix.postTranslate(shiftX, shiftY);

            // Update state
            if (mNumValidPoints >= MIN_POINTS_TRACKED) {
                mState = TrackState.TRACKED;
                mConsecutiveDegradedFrames = 0;
            } else {
                if (mState != TrackState.DEGRADED) {
                    mLastDegradedTimeMs = nowMs;
                }
                mState = TrackState.DEGRADED;
                mConsecutiveDegradedFrames++;
            }

            // Siapkan poin untuk frame berikutnya (recycle valid points)
            mPoints0.fromList(validPts1);

            // Re-seed feature points jika poin mulai menyusut
            if (mNumValidPoints < MAX_FEATURE_POINTS / 2) {
                reseedFeaturesInsideBBox(mCurrGray);
            }

        } else {
            // Motion gagal atau poin hilang
            if (mState != TrackState.DEGRADED) {
                mLastDegradedTimeMs = nowMs;
            }
            mState = TrackState.DEGRADED;
            mConsecutiveDegradedFrames++;

            if (nowMs - mLastDegradedTimeMs > MAX_DEGRADED_DURATION_MS || mConsecutiveDegradedFrames > 12) {
                mState = TrackState.LOST;
                mPoints0.fromList(new ArrayList<>());
            }
        }

        // Simpan frame saat ini sebagai previous frame
        mCurrGray.copyTo(mPrevGray);

        long t1 = SystemClock.elapsedRealtimeNanos();
        float ofDurationMs = (t1 - t0) / 1_000_000f;
        PerformanceProfiler.getInstance().recordOpticalFlowStep(ofDurationMs, mNumValidPoints, mFbErrorPx);

        return getSnapshot(captureTimestampNs);
    }

    /**
     * Menerima deteksi authoritative dari inferensi YOLOv8 (~412 ms).
     * Melakukan koreksi drift, interpolasi visual, atau reinitialization instan.
     */
    public synchronized TrackedVein onYoloResult(YoloDetector.Detection yoloDet, long yoloCaptureTsNs) {
        mLastYoloCaptureTsNs = yoloCaptureTsNs;

        if (yoloDet == null) {
            // YOLO tidak mendeteksi objek: jika tracker aktif, degradasi perlahan
            if (mState == TrackState.TRACKED) {
                mState = TrackState.DEGRADED;
                mLastDegradedTimeMs = SystemClock.uptimeMillis();
            } else if (mState == TrackState.DEGRADED) {
                if (SystemClock.uptimeMillis() - mLastDegradedTimeMs > 450L) {
                    mState = TrackState.LOST;
                    mPoints0.fromList(new ArrayList<>());
                }
            }
            return getSnapshot(yoloCaptureTsNs);
        }

        mRawYoloConfidence = yoloDet.rawConfidence > 0 ? yoloDet.rawConfidence : yoloDet.confidence;
        mCurrentConfidence = yoloDet.confidence;
        mCurrentDepthMm = yoloDet.depthMm;
        mReferenceMask = yoloDet.maskBitmap;

        // Hitung jarak diskrepansi antara deteksi YOLO dengan posisi Tracker saat ini
        boolean isLost = (mState == TrackState.LOST);
        float dist = 9999f;
        if (!isLost) {
            float dx = yoloDet.cx - mCurrentCx;
            float dy = yoloDet.cy - mCurrentCy;
            dist = (float) Math.sqrt(dx * dx + dy * dy);
        }

        if (isLost || dist > CORRECTION_DISAGREEMENT_SNAP_PX || yoloDet.confidence >= 0.65f && dist > 30f) {
            // IMMEDIATE REINITIALIZATION (Phase 10)
            mCurrentBBox.set(yoloDet.left, yoloDet.top, yoloDet.right, yoloDet.bottom);
            mCurrentCx = yoloDet.cx;
            mCurrentCy = yoloDet.cy;
            mAccumulatedMatrix.reset();
            mRemainingCorrectionSteps = 0;
            mCorrectionStepX = 0f;
            mCorrectionStepY = 0f;
            mState = TrackState.FRESH_YOLO;
            mConsecutiveDegradedFrames = 0;

            // Ekstraksi ulang fitur Shi-Tomasi pada ROI YOLO
            if (mCurrGray != null && !mCurrGray.empty()) {
                reseedFeaturesInsideBBox(mCurrGray);
            }
            PerformanceProfiler.getInstance().recordTrackerReset();

        } else {
            // PERIODIC CORRECTION INTERPOLATION (Phase 9)
            // Hindari visual snap: bagi selisih posisi ke 3-5 frame berikutnya
            float dx = yoloDet.cx - mCurrentCx;
            float dy = yoloDet.cy - mCurrentCy;

            mRemainingCorrectionSteps = CORRECTION_INTERPOLATION_STEPS;
            mCorrectionStepX = dx / (float) CORRECTION_INTERPOLATION_STEPS;
            mCorrectionStepY = dy / (float) CORRECTION_INTERPOLATION_STEPS;

            mState = TrackState.FRESH_YOLO;
            mConsecutiveDegradedFrames = 0;

            // Tambahkan fitur baru jika perlu
            if (mCurrGray != null && !mCurrGray.empty() && mNumValidPoints < MAX_FEATURE_POINTS / 2) {
                reseedFeaturesInsideBBox(mCurrGray);
            }
            PerformanceProfiler.getInstance().recordTrackerCorrection(dist);
        }

        return getSnapshot(yoloCaptureTsNs);
    }

    /**
     * Ekstraksi fitur Shi-Tomasi pada ROI BBox + margin 20% (Phase 4).
     */
    private void reseedFeaturesInsideBBox(Mat grayMat) {
        if (grayMat == null || grayMat.empty() || mCurrentBBox.width() <= 0 || mCurrentBBox.height() <= 0) return;

        float marginX = mCurrentBBox.width() * 0.20f;
        float marginY = mCurrentBBox.height() * 0.20f;

        int l = Math.max(0, (int)(mCurrentBBox.left - marginX));
        int t = Math.max(0, (int)(mCurrentBBox.top - marginY));
        int r = Math.min(mFrameWidth - 1, (int)(mCurrentBBox.right + marginX));
        int b = Math.min(mFrameHeight - 1, (int)(mCurrentBBox.bottom + marginY));

        if (r <= l || b <= t) return;

        mRoiMask.setTo(new Scalar(0));
        Rect roiCv = new Rect(l, t, r - l, b - t);
        Mat roiSub = mRoiMask.submat(roiCv);
        roiSub.setTo(new Scalar(255));
        roiSub.release();

        MatOfPoint corners = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(
                grayMat, corners,
                MAX_FEATURE_POINTS,
                0.015, // qualityLevel
                5.0,   // minDistance
                mRoiMask,
                3, false, 0.04
        );

        Point[] cArr = corners.toArray();
        corners.release();

        if (cArr.length > 0) {
            mPoints0.fromArray(cArr);
            mNumInputPoints = cArr.length;
        }
    }

    private void clampBBoxToScreen() {
        if (mFrameWidth <= 0 || mFrameHeight <= 0) return;
        float w = mCurrentBBox.width();
        float h = mCurrentBBox.height();

        if (mCurrentBBox.left < 0) {
            mCurrentBBox.left = 0;
            mCurrentBBox.right = w;
        }
        if (mCurrentBBox.top < 0) {
            mCurrentBBox.top = 0;
            mCurrentBBox.bottom = h;
        }
        if (mCurrentBBox.right > mFrameWidth) {
            mCurrentBBox.right = mFrameWidth;
            mCurrentBBox.left = mFrameWidth - w;
        }
        if (mCurrentBBox.bottom > mFrameHeight) {
            mCurrentBBox.bottom = mFrameHeight;
            mCurrentBBox.top = mFrameHeight - h;
        }
        mCurrentCx = mCurrentBBox.centerX();
        mCurrentCy = mCurrentBBox.centerY();
    }

    private float calculateMedian(List<Float> list) {
        if (list == null || list.isEmpty()) return 0f;
        Collections.sort(list);
        int mid = list.size() / 2;
        if (list.size() % 2 == 1) return list.get(mid);
        return (list.get(mid - 1) + list.get(mid)) / 2f;
    }

    public synchronized TrackedVein getSnapshot(long captureTimestampNs) {
        TrackedVein snap = new TrackedVein();
        snap.state = mState;
        snap.bbox.set(mCurrentBBox);
        snap.cx = mCurrentCx;
        snap.cy = mCurrentCy;
        snap.confidence = mCurrentConfidence;
        snap.depthMm = mCurrentDepthMm;
        snap.maskBitmap = mReferenceMask;
        snap.maskMatrix.set(mAccumulatedMatrix);

        snap.numInputPoints = mNumInputPoints;
        snap.numValidPoints = mNumValidPoints;
        snap.fbErrorPx = mFbErrorPx;
        snap.lkErrorPx = mLkErrorPx;

        snap.captureTimestampNs = (captureTimestampNs > 0) ? captureTimestampNs : mLastYoloCaptureTsNs;
        snap.updateTimestampNs = SystemClock.elapsedRealtimeNanos();
        if (snap.captureTimestampNs > 0) {
            snap.predAgeMs = (snap.updateTimestampNs - snap.captureTimestampNs) / 1_000_000f;
        }
        snap.isFreshYolo = (mState == TrackState.FRESH_YOLO);
        snap.rawConfidence = mRawYoloConfidence;
        return snap;
    }
}

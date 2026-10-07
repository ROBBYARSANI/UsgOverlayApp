package com.example.usgandroidapp;

import android.Manifest;
import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private static final int REQ_OVERLAY    = 1001;
    private static final int REQ_CAPTURE    = 1002;
    private static final int REQ_NOTIF_PERM = 1003;

    public static final String ACTION_SERVICE_STOPPED = "com.example.usgandroidapp.SERVICE_STOPPED";
    public static final String ACTION_SERVICE_STARTED = "com.example.usgandroidapp.SERVICE_STARTED";

    private MediaProjectionManager mProjectionManager;
    private TextView               mStatusText;
    private Button                 mStartButton;
    private Button                 mStopButton;

    private boolean mInPermissionFlow   = false;
    private boolean mReceiverRegistered = false;

    private final BroadcastReceiver mServiceReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent == null || intent.getAction() == null) return;
            switch (intent.getAction()) {
                case ACTION_SERVICE_STARTED:
                    Log.d(TAG, "BR: service started");
                    mInPermissionFlow = false;
                    runOnUiThread(() -> setUIState(true, "Status: Deteksi berjalan..."));
                    break;
                case ACTION_SERVICE_STOPPED:
                    Log.d(TAG, "BR: service stopped");
                    mInPermissionFlow = false;
                    runOnUiThread(() -> setUIState(false, "Status: Siap dimulai."));
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mStatusText  = findViewById(R.id.tv_status);
        mStartButton = findViewById(R.id.btn_start);
        mStopButton  = findViewById(R.id.btn_stop);

        mProjectionManager = (MediaProjectionManager)
                getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        mStartButton.setOnClickListener(v -> onStartClicked());
        mStopButton.setOnClickListener(v -> onStopClicked());

        boolean running = VeinDetectionService.isRunning();
        setUIState(running, running ? "Status: Deteksi berjalan..." : "Status: Siap dimulai.");
    }

    @Override
    protected void onStart() {
        super.onStart();
        if (!mReceiverRegistered) {
            IntentFilter filter = new IntentFilter();
            filter.addAction(ACTION_SERVICE_STOPPED);
            filter.addAction(ACTION_SERVICE_STARTED);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(mServiceReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
            } else {
                registerReceiver(mServiceReceiver, filter);
            }
            mReceiverRegistered = true;
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (mReceiverRegistered) {
            try { unregisterReceiver(mServiceReceiver); } catch (Exception ignored) {}
            mReceiverRegistered = false;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!mInPermissionFlow) {
            boolean running = VeinDetectionService.isRunning();
            setUIState(running, running ? "Status: Deteksi berjalan..." : "Status: Siap dimulai.");
        }
        if (VeinDetectionService.isRunning()) {
            VeinDetectionService.setOverlayVisible(false);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (VeinDetectionService.isRunning()) {
            VeinDetectionService.setOverlayVisible(true);
        }
    }

    private void onStartClicked() {
        if (mInPermissionFlow || VeinDetectionService.isRunning()) return;
        mInPermissionFlow = true;
        setUIState(false, "Memeriksa izin...");
        mStartButton.setEnabled(false);
        checkOverlayPermission();
    }

    private void onStopClicked() {
        Log.d(TAG, "Stop clicked");
        mInPermissionFlow = false;
        Intent serviceIntent = new Intent(this, VeinDetectionService.class);
        serviceIntent.setAction(VeinDetectionService.ACTION_STOP);
        startService(serviceIntent);
        setUIState(false, "Status: Menghentikan...");
    }

    private void checkOverlayPermission() {
        if (!Settings.canDrawOverlays(this)) {
            mStatusText.setText("Langkah 1/3: Izinkan tampilan overlay...");
            startActivityForResult(
                    new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                            Uri.parse("package:" + getPackageName())),
                    REQ_OVERLAY
            );
        } else {
            checkNotificationPermission();
        }
    }

    private void checkNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                mStatusText.setText("Langkah 2/3: Izinkan notifikasi...");
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.POST_NOTIFICATIONS},
                        REQ_NOTIF_PERM);
                return;
            }
        }
        requestScreenCapture();
    }

    private void requestScreenCapture() {
        mStatusText.setText("Langkah 3/3: Izinkan rekam layar...");
        startActivityForResult(mProjectionManager.createScreenCaptureIntent(), REQ_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQ_OVERLAY) {
            if (Settings.canDrawOverlays(this)) {
                checkNotificationPermission();
            } else {
                showError("Izin overlay diperlukan. Aktifkan di Settings lalu coba lagi.");
            }

        } else if (requestCode == REQ_CAPTURE) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                Log.d(TAG, "Screen capture granted. Starting service...");
                
                mStatusText.setText("Status: Memulai deteksi...");
                setUIState(true, "Status: Memulai deteksi...");
                startDetectionService(resultCode, data);

            } else {
                showError("Izin rekam layar dibatalkan.");
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_NOTIF_PERM) {
            requestScreenCapture();
        }
    }

    private void startDetectionService(int resultCode, Intent projectionData) {
        Intent serviceIntent = new Intent(this, VeinDetectionService.class);
        serviceIntent.setAction(VeinDetectionService.ACTION_START);
        serviceIntent.putExtra(VeinDetectionService.EXTRA_RESULT_CODE, resultCode);
        serviceIntent.putExtra(VeinDetectionService.EXTRA_PROJECTION_DATA, projectionData);
        ContextCompat.startForegroundService(this, serviceIntent);
        Log.d(TAG, "startForegroundService called.");
    }

    private void setUIState(boolean isRunning, String statusMsg) {
        mStatusText.setText(statusMsg);
        boolean canStart = !isRunning && !mInPermissionFlow;
        mStartButton.setEnabled(canStart);
        mStopButton.setEnabled(isRunning);
        mStartButton.setAlpha(canStart ? 1.0f : 0.5f);
        mStopButton.setAlpha(isRunning ? 1.0f : 0.5f);
    }

    private void showError(String message) {
        Log.e(TAG, "showError: " + message);
        mInPermissionFlow = false;
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
        setUIState(false, "Error: " + message);
    }
}

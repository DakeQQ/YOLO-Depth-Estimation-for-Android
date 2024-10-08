package com.example.myapplication;

import static com.example.myapplication.GLRender.FPS;
import static com.example.myapplication.GLRender.camera_height;
import static com.example.myapplication.GLRender.camera_width;
import static com.example.myapplication.GLRender.central_depth;
import static com.example.myapplication.GLRender.focus_area;


import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    @SuppressLint("StaticFieldLeak")
    public static GLSurfaceView mGLSurfaceView;
    @SuppressLint("StaticFieldLeak")
    private static Context mContext;
    @SuppressLint("StaticFieldLeak")
    private static GLRender mGLRender;
    private static CameraManager mCameraManager;
    private static CameraDevice mCameraDevice;
    private static CameraCaptureSession mCaptureSession;
    private static CaptureRequest mPreviewRequest;
    private static CaptureRequest.Builder mPreviewRequestBuilder;
    private static String mCameraId;
    private static Handler mBackgroundHandler;
    private static HandlerThread mBackgroundThread;
    public static final int REQUEST_CAMERA_PERMISSION = 1;
    public static final String file_name_class = "class.txt";
    public static final List<String> labels = new ArrayList<>();
    @SuppressLint("StaticFieldLeak")
    public static TextView FPS_view;
    @SuppressLint("StaticFieldLeak")
    public static TextView class_view;
    @SuppressLint("StaticFieldLeak")
    public static TextView depth_view;
    public static StringBuilder class_result = new StringBuilder();

    static {
        System.loadLibrary("myapplication");
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mContext = this;
        AssetManager mgr = getAssets();
        Read_Assets(file_name_class, mgr);
        FPS_view = findViewById(R.id.fps);
        class_view = findViewById(R.id.class_list);
        depth_view = findViewById(R.id.depth);
        if (!Load_Models_A(mgr,false,false,false,false,false,false)) {
            FPS_view.setText("YOLO failed.");
        }
        // Close the load code if you don't need it.
        if (!Load_Models_B(mgr,false,false,false,false,false,false)) {
            depth_view.setText("Depth failed.");
        }
        if (!Load_Models_C(mgr,false,false,false,false,false,false)) {
            FPS_view.setText("TwinLite failed.");
        }
        setWindowFlag();
        initView();
    }
    private void setWindowFlag() {
        Window window = getWindow();
        window.addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }
    private void initView() {
        mGLSurfaceView = findViewById(R.id.glSurfaceView);
        mGLSurfaceView.setEGLContextClientVersion(3);
        mGLRender = new GLRender(mContext);
        mGLSurfaceView.setRenderer(mGLRender);
    }
    @Override
    public void onResume() {
        super.onResume();
    }
    @Override
    public void onPause() {
        super.onPause();
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        closeCamera();
        stopBackgroundThread();
    }
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }
    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @SuppressLint("SetTextI18n")
    private void requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            FPS_view.setText("Camera permission failed");
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }
    public void openCamera() {
        if (ContextCompat.checkSelfPermission(mContext, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }
        startBackgroundThread();
        setUpCameraOutputs();

        try {
            mCameraManager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
    private void setUpCameraOutputs() {
        mCameraManager = (CameraManager) mContext.getSystemService(Context.CAMERA_SERVICE);
        mCameraId = CameraUtils.getInstance().getCameraId();
    }
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @SuppressLint({"ResourceType", "DefaultLocale", "SetTextI18n"})
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            try {
                SurfaceTexture surfaceTexture = mGLRender.getSurfaceTexture();
                if (surfaceTexture == null) {
                    return;
                }
                surfaceTexture.setDefaultBufferSize(camera_width, camera_height);
                surfaceTexture.setOnFrameAvailableListener(surfaceTexture1 -> {
                    mGLSurfaceView.requestRender();
                    FPS_view.setText("FPS: " + String.format("%.1f", FPS));
                    depth_view.setText("Central\nDepth: " + String.format("%.2f", central_depth) + " m");
                    class_view.setText(class_result);
                    class_result.setLength(0);
                });
                Surface surface = new Surface(surfaceTexture);
                mCameraDevice = cameraDevice;
                mPreviewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                mPreviewRequestBuilder.addTarget(surface);
                mPreviewRequest = mPreviewRequestBuilder.build();
                cameraDevice.createCaptureSession(List.of(surface), sessionsStateCallback, null);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }
        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            cameraDevice.close();
            mCameraDevice = null;
        }
        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            mCameraDevice = null;
            finish();
        }
    };
    CameraCaptureSession.StateCallback sessionsStateCallback = new CameraCaptureSession.StateCallback() {
        @Override
        public void onConfigured(@NonNull CameraCaptureSession session) {
            if (null == mCameraDevice) return;
            mCaptureSession = session;
            try {
                // Turn off for processing speed (lower power consumption). / Turn on for image quality.
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_CAPTURE_INTENT, CaptureRequest.CONTROL_CAPTURE_INTENT_PREVIEW);
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON);
                mPreviewRequestBuilder.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY);
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_REGIONS, focus_area);
                mPreviewRequest = mPreviewRequestBuilder.build();
                mCaptureSession.setRepeatingRequest(mPreviewRequest, mCaptureCallback, mBackgroundHandler);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }
        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
        }
    };
    private final CameraCaptureSession.CaptureCallback mCaptureCallback
            = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureProgressed(@NonNull CameraCaptureSession session,
                                        @NonNull CaptureRequest request,
                                        @NonNull CaptureResult partialResult) {
        }

        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                       @NonNull CaptureRequest request,
                                       @NonNull TotalCaptureResult result) {
        }
    };
    private void closeCamera() {
        if (null != mCaptureSession) {
            mCaptureSession.close();
            mCaptureSession = null;
        }
        if (null != mCameraDevice) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
    }
    private void Read_Assets(String file_name, AssetManager mgr) {
        switch (file_name) {
            case file_name_class -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_class)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        labels.add(line);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    public static native void Process_Init(int textureId);
    public static native float[] Process_Texture();
    private native boolean Load_Models_A(AssetManager assetManager, boolean USE_GPU, boolean FP16, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    private native boolean Load_Models_B(AssetManager assetManager, boolean USE_GPU, boolean FP16, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    private native boolean Load_Models_C(AssetManager assetManager, boolean USE_GPU, boolean FP16, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    public static native float[] Run_YOLO(float[] pixel_values);
    public static native float[] Run_Depth(float[] pixel_values);
    public static native float[] Run_TwinLite(float[] pixel_values);
}

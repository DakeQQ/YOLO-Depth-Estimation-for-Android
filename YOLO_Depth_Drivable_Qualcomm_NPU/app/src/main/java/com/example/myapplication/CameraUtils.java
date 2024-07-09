/**
 * Create By Shawn.xiao at 2023/05/01
 */
package com.example.myapplication;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.params.StreamConfigurationMap;

import android.util.Size;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import java.util.List;

public class CameraUtils {
    private static final String BACK_MAIN_CAMERA = "0";
    private static final String FRONT_CAMERA = "1";
    private static final String BACK_WIDTH_CAMERA = "4";

    private static final class SInstanceHolder {
        private static final CameraUtils sInstance = new CameraUtils();
    }

    public static CameraUtils getInstance() {
        return SInstanceHolder.sInstance;
    }

    public String getCameraId() {
        //先写死返回后摄
        return BACK_MAIN_CAMERA;
    }
    public List<Size> getCameraOutputSizes(CameraCharacteristics characteristics, Class clz) {
        StreamConfigurationMap configs = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        assert configs != null;
        List<Size> sizes = Arrays.asList(configs.getOutputSizes(clz));
        sizes.sort(Comparator.comparingInt(s -> s.getWidth() * s.getHeight()));
        Collections.reverse(sizes);
        return sizes;
    }
}

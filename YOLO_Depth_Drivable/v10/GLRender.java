package com.example.myapplication;

import static com.example.myapplication.MainActivity.Process_Init;
import static com.example.myapplication.MainActivity.Process_Texture;
import static com.example.myapplication.MainActivity.Run_Depth;
import static com.example.myapplication.MainActivity.Run_TwinLite;
import static com.example.myapplication.MainActivity.Run_YOLO;
import static com.example.myapplication.MainActivity.class_result;
import static com.example.myapplication.MainActivity.labels;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.params.MeteringRectangle;
import android.opengl.GLES11Ext;
import android.opengl.GLES32;
import android.opengl.GLSurfaceView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class GLRender implements GLSurfaceView.Renderer {
    private static final ExecutorService executorService = Executors.newFixedThreadPool(4);
    @SuppressLint("StaticFieldLeak")
    private static Context mContext;
    private static int mVertexLocation;
    private static int mTextureLocation;
    private static int mUTextureLocation;
    private static int mVMatrixLocation;
    private static int box_position;
    private static int box_color;
    private static int ShaderProgram_Camera;
    private static int ShaderProgram_YOLO;
    private static final int BYTES_FLOAT_2 = 8;
    public static final int camera_width = 1280;  // Please modify the project.h file simultaneously when editing these values.
    public static final int camera_height = 720;
    private static final int yolo_width = 512;
    private static final int yolo_height = 288;
    private static final int depth_width = 518;
    private static final int depth_height = 294;
    private static final int yolo_output_length = 1800;  // 300 * 6
    private static final int camera_pixels = camera_height * camera_width;
    private static final int camera_pixels_2 = camera_pixels * 2;
    private static final int depth_pixels = depth_width * depth_height;
    private static final int depth_height_offset = 25;
    private static final int depth_width_offset = depth_height_offset * depth_width;
    private static final int depth_central_position_5 = (depth_pixels - depth_width) >> 1;
    private static final int depth_central_position_8 = depth_central_position_5 - depth_width_offset;
    private static final int depth_central_position_2 = depth_central_position_5 + depth_width_offset;
    private static long sum_t = 0;
    private static long count_t = 1000;
    private static final int[] mTextureId = new int[1];
    private static final int[] depth_central_area = new int[]{depth_central_position_2 - depth_height_offset, depth_central_position_2, depth_central_position_2 + depth_height_offset, depth_central_position_5 - depth_height_offset, depth_central_position_5, depth_central_position_5 + depth_height_offset, depth_central_position_8 - depth_height_offset, depth_central_position_8, depth_central_position_8 + depth_height_offset};
    private static int[] imageRGBA = new int[camera_pixels];
    public static final MeteringRectangle[] focus_area = new MeteringRectangle[]{new MeteringRectangle(camera_width >> 1, camera_height >> 1, 100, 100, MeteringRectangle.METERING_WEIGHT_MAX)};
    public static final float depth_adjust_factor = 1.f;  // Please adjust it by yourself to get more depth accuracy. This factor should be optimized by making it a function of focal distance rather than maintaining it as a constant.
    private static final float depth_adjust_bias = 0.f;  // Please adjust it by yourself to get more depth accuracy. This factor should be optimized by making it a function of focal distance rather than maintaining it as a constant.
    private static final float yolo_detect_threshold = 0.3f;
    private static final float color_factor = 1.f / (1.f - yolo_detect_threshold);
    private static final float line_width = 6.f;  // draw boxes
    private static final float depth_w_factor = 0.5f * depth_width / yolo_width;
    private static final float depth_h_factor = 0.5f * depth_height / yolo_height;
    private static final float inv_yolo_width = 2.f / (float) yolo_width;
    private static final float inv_yolo_height = 2.f / (float) yolo_height;
    private static final float inv_255 = 1.f / 255.f;
    public static float FPS;
    public static float central_depth;
    private static final float[] lowColor = {1.0f, 1.0f, 0.0f}; // Yellow for low confidence.
    private static final float[] highColor = {1.0f, 0.0f, 0.0f}; // Red for high confidence.
    private static final float[] image_rgb = new float[camera_pixels * 3];
    private static float[] depth_results = new float[depth_pixels];
    private static final float[] vMatrix = new float[16];
    private static final String VERTEX_ATTRIB_POSITION = "aPosVertex";
    private static final String VERTEX_ATTRIB_TEXTURE_POSITION = "aTexVertex";
    private static final String UNIFORM_TEXTURE = "camera_texture";
    private static final String UNIFORM_VMATRIX = "vMatrix";
    private static final String BOX_POSITION = "box_position";
    private static final String BOX_COLOR = "box_color";
    private static final String camera_vertex_shader_name = "camera_vertex_shader.glsl";
    private static final String camera_fragment_shader_name = "camera_fragment_shader.glsl";
    private static final String yolo_vertex_shader_name = "yolo_vertex_shader.glsl";
    private static final String yolo_fragment_shader_name = "yolo_fragment_shader.glsl";
    public static SurfaceTexture mSurfaceTexture;
    private static boolean run_yolo = true;  // true for turn on the function.
    private static boolean run_depth = false;  // true for turn on the function. Enabling both YOLO and depth estimation simultaneously decrease performance by 30+%.
    private static boolean run_twinLite = false;
    private static final LinkedList<LinkedList<Classifier.Recognition>> draw_queue_yolo = new LinkedList<>();
    private static final LinkedList<float[]> draw_queue_twinLite = new LinkedList<>();
    public GLRender(Context context) {
        mContext = context;
    }
    public SurfaceTexture getSurfaceTexture() {
        return mSurfaceTexture;
    }
    // Center cross +
    private static final FloatBuffer cross_float_buffer = getFloatBuffer(new float[]{
            0.f, -0.04f,
            0.f, 0.04f,
            0.f, 0.f,
            0.06f, 0.f,
            -0.06f, 0.f
    });
    private static final FloatBuffer mVertexCoord_buffer = getFloatBuffer(new float[]{
            -1f, -1f,
            1f, -1f,
            -1f, 1f,
            1f, 1f
    });
    private static final FloatBuffer mTextureCoord_buffer = getFloatBuffer(new float[]{
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f
    });
    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES32.glEnable(GLES32.GL_BLEND);
        GLES32.glBlendFunc(GLES32.GL_SRC_ALPHA, GLES32.GL_ONE_MINUS_SRC_ALPHA);
        GLES32.glClearColor(0.f, 0.0f, 0.0f, 1.0f);
        ShaderProgram_Camera = createAndLinkProgram(camera_vertex_shader_name, camera_fragment_shader_name);
        ShaderProgram_YOLO = createAndLinkProgram(yolo_vertex_shader_name, yolo_fragment_shader_name);
        initTexture();
        initAttribLocation();
        Process_Init(mTextureId[0]);
        ((MainActivity) mContext).openCamera();
    }
    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES32.glViewport(0, 0, camera_height, camera_width);
    }
    @Override
    public void onDrawFrame(GL10 gl) {
        mSurfaceTexture.updateTexImage();
        mSurfaceTexture.getTransformMatrix(vMatrix);
        Draw_Camera_Preview();
        if (!run_yolo && !run_depth && !run_twinLite) {
            imageRGBA = Process_Texture();
            // Choose CPU normalization over GPU, as GPU float32 buffer access is much slower than int8 buffer access. 
            // Therefore, use a new thread to parallelize the normalization process.
            executorService.execute(() -> {
                for (int i = 0; i < camera_pixels; i++) {
                    int rgba = imageRGBA[i];
                    image_rgb[i] = (float) ((rgba >> 16) & 0xFF) * inv_255;
                    image_rgb[i + camera_pixels] = (float) ((rgba >> 8) & 0xFF) * inv_255;
                    image_rgb[i + camera_pixels_2] = (float) (rgba & 0xFF) * inv_255;
                }
            });
        }
        if (run_yolo) {
            run_yolo = false;
            executorService.execute(() -> {
                long t = System.currentTimeMillis();
                draw_queue_yolo.add(Post_Process_Yolo(Run_YOLO(image_rgb)));
                sum_t += System.currentTimeMillis() - t;
                FPS = (float) count_t / sum_t;
                if (count_t > 99999) {  // Reset
                    count_t >>= 1;
                    sum_t >>= 1;
                }
                count_t += 1000;
                run_yolo = true;
            });
        }
        if (run_depth) {
            run_depth = false;
            executorService.execute(() -> {
                depth_results = Run_Depth(image_rgb);
                float center_area = 0.f;
                for (int i = 0; i < 9; i++) {
                    center_area += depth_results[depth_central_area[i]];
                }
                central_depth = depth_adjust_factor * (center_area * 0.111111111f + central_depth) * 0.5f + depth_adjust_bias;
                run_depth = true;
            });
        }
        if (run_twinLite) {
            run_twinLite = false;
            executorService.execute(() -> {
                draw_queue_twinLite.add(Run_TwinLite(image_rgb));
                run_twinLite = true;
            });
        }
        if (!draw_queue_yolo.isEmpty()) {
            drawBox(draw_queue_yolo.removeFirst());
        }
        if (!draw_queue_twinLite.isEmpty()) {
            draw_area_lane(draw_queue_twinLite.removeFirst());
        }
    }
    private static LinkedList<Classifier.Recognition> Post_Process_Yolo(float[] outputs) {
        LinkedList<Classifier.Recognition> detections = new LinkedList<>();
        for (int i = 0; i < yolo_output_length; i+=6) {
            float maxScore = outputs[i + 4];
            if (maxScore >= yolo_detect_threshold) {
                RectF rect = new RectF(
                        Math.max(0.f, outputs[i]),
                        Math.max(0.f, outputs[i + 1]),
                        Math.min(yolo_width, outputs[i + 2]),
                        Math.min(yolo_height, outputs[i + 3])
                );
                detections.add(new Classifier.Recognition("", labels.get((int) outputs[i + 5]), maxScore, rect));
            }
        }
        return detections;
    }

    @SuppressLint("DefaultLocale")
    private static void drawBox(LinkedList<Classifier.Recognition> nmsList) {
        GLES32.glUseProgram(ShaderProgram_YOLO);
        for (Classifier.Recognition draw_target : nmsList) {
            RectF box = draw_target.getLocation();
            if (run_depth) {
                float depth_avg = Get_Depth_Central_5_Points(box);  // Disable it, if no depth model.
                class_result.append(draw_target.getTitle()).append(" / ").append(String.format("%.1f", 100.f * draw_target.getConfidence())).append("% / ").append(String.format("%.1f", depth_avg)).append(" m\n");
            } else {
                class_result.append(draw_target.getTitle()).append(" / ").append(String.format("%.1f", 100.f * draw_target.getConfidence())).append("%\n");
            }
            box.top = 1.f - box.top * inv_yolo_height;
            box.bottom = 1.f - box.bottom * inv_yolo_height;
            box.left = 1.f - box.left * inv_yolo_width;
            box.right = 1.f - box.right * inv_yolo_width;
            float[] rotatedVertices = {
                    box.top, box.left,
                    box.top, box.right,
                    box.bottom, box.right,
                    box.bottom, box.left
            };
            float[] color = getColorFromConfidence(draw_target.getConfidence());
            GLES32.glUniform4f(box_color, color[0], color[1], color[2], 1.f);
            GLES32.glVertexAttribPointer(box_position, 2, GLES32.GL_FLOAT, false, BYTES_FLOAT_2, getFloatBuffer(rotatedVertices));
            GLES32.glDrawArrays(GLES32.GL_LINE_LOOP, 0, 4);
        }
        // Draw center cross mark.
        GLES32.glUniform4f(box_color, 1.f, 1.f, 1.f, 1.f);
        GLES32.glVertexAttribPointer(box_position, 2, GLES32.GL_FLOAT, false, BYTES_FLOAT_2, cross_float_buffer);
        GLES32.glDrawArrays(GLES32.GL_LINE_STRIP, 0, 5);
    }
    private static float Get_Depth_Central_5_Points(RectF box) {
        int target_position =  ((int) ((box.top + box.bottom) * depth_h_factor) - 1) * depth_width + (int) ((box.left + box.right) * depth_w_factor);
        if (target_position >= depth_pixels) {
            target_position = depth_pixels - 1;
        } else if (target_position < 0) {
            target_position = 0;
        }
        int target_position_left = target_position - depth_height_offset;
        if (target_position_left < 0) {
            target_position_left = 0;
        }
        int target_position_right = target_position + depth_height_offset;
        if (target_position_right >= depth_pixels) {
            target_position_right = depth_pixels - 1;
        }
        int target_position_up = target_position - depth_width_offset;
        if (target_position_up < 0) {
            target_position_up = 0;
        }
        int target_position_bottom = target_position + depth_width_offset;
        if (target_position_bottom >= depth_pixels) {
            target_position_bottom = depth_pixels - 1;
        }
        return depth_adjust_factor * (depth_results[target_position] + depth_results[target_position_left] + depth_results[target_position_right] + depth_results[target_position_up] + depth_results[target_position_bottom]) * 0.2f + depth_adjust_bias;
    }
    private static void draw_area_lane(float[] twinLite_output) {
        int area_len = ((int) twinLite_output[0]) >> 1;
        int lane_len = ((int) twinLite_output[1]) >> 1;
        if (area_len != 0 || lane_len != 0) {
            FloatBuffer output_buffer = getFloatBuffer(twinLite_output);
            GLES32.glUseProgram(ShaderProgram_YOLO);
            if (area_len != 0) {
                GLES32.glUniform4f(box_color, 0.f, 0.f, 1.f, 0.5f);
                GLES32.glVertexAttribPointer(box_position, 2, GLES32.GL_FLOAT, false, BYTES_FLOAT_2, output_buffer);
                GLES32.glDrawArrays(GLES32.GL_POINTS, 1, area_len);
            }
            if (lane_len != 0) {
                GLES32.glUniform4f(box_color, 0.f, 1.f, 0.f, 0.5f);
                GLES32.glVertexAttribPointer(box_position, 2, GLES32.GL_FLOAT, false, BYTES_FLOAT_2, output_buffer);
                GLES32.glDrawArrays(GLES32.GL_POINTS, 1 + area_len, lane_len);
            }
        }
    }
    private static void Draw_Camera_Preview() {
        GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT);
        GLES32.glUseProgram(ShaderProgram_Camera);
        GLES32.glVertexAttribPointer(mVertexLocation, 2, GLES32.GL_FLOAT, false, 0, mVertexCoord_buffer);
        GLES32.glVertexAttribPointer(mTextureLocation, 2, GLES32.GL_FLOAT, false, 0, mTextureCoord_buffer);
        GLES32.glEnableVertexAttribArray(mVertexLocation);
        GLES32.glEnableVertexAttribArray(mTextureLocation);
        GLES32.glUniformMatrix4fv(mVMatrixLocation, 1, false, vMatrix, 0);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_STRIP, 0, 4);  // mVertexCoord.length / 2
    }
    private static void initAttribLocation() {
        GLES32.glLineWidth(line_width);
        mVertexLocation = GLES32.glGetAttribLocation(ShaderProgram_Camera, VERTEX_ATTRIB_POSITION);
        mTextureLocation = GLES32.glGetAttribLocation(ShaderProgram_Camera, VERTEX_ATTRIB_TEXTURE_POSITION);
        mUTextureLocation = GLES32.glGetUniformLocation(ShaderProgram_Camera, UNIFORM_TEXTURE);
        mVMatrixLocation = GLES32.glGetUniformLocation(ShaderProgram_Camera, UNIFORM_VMATRIX);
        box_position = GLES32.glGetAttribLocation(ShaderProgram_YOLO, BOX_POSITION);
        box_color = GLES32.glGetUniformLocation(ShaderProgram_YOLO, BOX_COLOR);
    }
    private static void initTexture() {
        GLES32.glGenTextures(mTextureId.length, mTextureId, 0);
        GLES32.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, mTextureId[0]);
        GLES32.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES32.GL_TEXTURE_MIN_FILTER, GLES32.GL_LINEAR);
        GLES32.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES32.GL_TEXTURE_MAG_FILTER, GLES32.GL_LINEAR);
        GLES32.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES32.GL_TEXTURE_WRAP_S, GLES32.GL_CLAMP_TO_EDGE);
        GLES32.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES32.GL_TEXTURE_WRAP_T, GLES32.GL_CLAMP_TO_EDGE);
        mSurfaceTexture = new SurfaceTexture(mTextureId[0]);
        mSurfaceTexture.setDefaultBufferSize(camera_width, camera_height);
        GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
        GLES32.glUniform1i(mUTextureLocation, 0);
    }
    private static int createAndLinkProgram(String vertexShaderFN, String fragShaderFN) {
        int shaderProgram = GLES32.glCreateProgram();
        if (shaderProgram == 0) {
            return 0;
        }
        AssetManager mgr = mContext.getResources().getAssets();
        int vertexShader = loadShader(GLES32.GL_VERTEX_SHADER, loadShaderSource(mgr, vertexShaderFN));
        if (0 == vertexShader) {
            return 0;
        }
        int fragmentShader = loadShader(GLES32.GL_FRAGMENT_SHADER, loadShaderSource(mgr, fragShaderFN));
        if (0 == fragmentShader) {
            return 0;
        }
        GLES32.glAttachShader(shaderProgram, vertexShader);
        GLES32.glAttachShader(shaderProgram, fragmentShader);
        GLES32.glLinkProgram(shaderProgram);
        int[] linked = new int[1];
        GLES32.glGetProgramiv(shaderProgram, GLES32.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            GLES32.glDeleteProgram(shaderProgram);
            return 0;
        }
        return shaderProgram;
    }
    private static int loadShader(int type, String shaderSource) {
        int shader = GLES32.glCreateShader(type);
        if (shader == 0) {
            return 0;
        }
        GLES32.glShaderSource(shader, shaderSource);
        GLES32.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES32.glGetShaderiv(shader, GLES32.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            GLES32.glDeleteShader(shader);
            return 0;
        }
        return shader;
    }
    private static String loadShaderSource(AssetManager mgr, String file_name) {
        StringBuilder strBld = new StringBuilder();
        String nextLine;
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(mgr.open(file_name)));
            while ((nextLine = br.readLine()) != null) {
                strBld.append(nextLine).append('\n');
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return strBld.toString();
    }
    private static FloatBuffer getFloatBuffer(float[] array) {
        FloatBuffer buffer = ByteBuffer.allocateDirect((array.length << 2)).order(ByteOrder.nativeOrder()).asFloatBuffer();
        buffer.put(array).position(0);
        return buffer;
    }
    private static float[] getColorFromConfidence(float confidence) {
        float[] color = new float[3];
        float factor = (confidence - yolo_detect_threshold) * color_factor;
        for (int i = 0; i < 3; ++i) {
            color[i] = lowColor[i] + (highColor[i] - lowColor[i]) * factor;
        }
        return color;
    }

}

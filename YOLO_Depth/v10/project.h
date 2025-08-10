#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include <GLES3/gl32.h>

const char* computeShaderSource = "#version 320 es\n"
                                  "#extension GL_OES_EGL_image_external_essl3 : require\n"
                                  "precision mediump float;\n"
                                  "layout(local_size_x = 16, local_size_y = 16) in;\n"
                                  "const int camera_width = 1280;\n"    // Please set this value to match the exported model.
                                  "const int camera_height = 720;\n"    // Please set this value to match the exported model.
                                  "const uint pixel_count = uint(camera_width * camera_height);\n"
                                  "layout(binding = 0) uniform samplerExternalOES yuvTex;\n"
                                  "layout(std430, binding = 1) buffer Output {\n"
                                  "    uint result[];\n"
                                  "} outputData;\n"
                                  "const vec3 bias = vec3(-0.15, -0.5, -0.2);\n"
                                  "const mat3 YUVtoRGBMatrix = mat3(127.5, 0.0, 1.402 * 127.5,\n"
                                  "                                 127.5, -0.344136 * 127.5, -0.714136 * 127.5,\n"
                                  "                                 127.5, 1.772 * 127.5, 0.0);\n"
                                  "void main() {\n"
                                  "    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);\n"
                                  "    if (gid.x >= camera_width || gid.y >= camera_height) return;\n"
                                  "\n"
                                  "    // Process 4 horizontally adjacent pixels per invocation starting at x aligned to 4\n"
                                  "    int baseX = gid.x & ~3; // floor to multiple of 4\n"
                                  "    if (gid.x != baseX) return; // only 1/4 of threads do useful work\n"
                                  "\n"
                                  "    // Precompute base linear index for the first pixel of the 4-pack\n"
                                  "    uint base_pix_idx = uint(gid.y) * uint(camera_width) + uint(baseX);\n"
                                  "    uint out_idx_uint = base_pix_idx >> 2; // one uint holds 4 bytes\n"
                                  "    const uint plane_stride_uint = pixel_count >> 2; // per-channel stride in uints\n"
                                  "\n"
                                  "    // Fetch and convert 4 pixels\n"
                                  "    ivec2 p0 = ivec2(baseX + 0, gid.y);\n"
                                  "    ivec2 p1 = ivec2(baseX + 1, gid.y);\n"
                                  "    ivec2 p2 = ivec2(baseX + 2, gid.y);\n"
                                  "    ivec2 p3 = ivec2(baseX + 3, gid.y);\n"
                                  "\n"
                                  "    ivec3 rgb0 = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, p0, 0).rgb + bias)), -128, 127) + 128;\n"
                                  "    ivec3 rgb1 = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, p1, 0).rgb + bias)), -128, 127) + 128;\n"
                                  "    ivec3 rgb2 = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, p2, 0).rgb + bias)), -128, 127) + 128;\n"
                                  "    ivec3 rgb3 = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, p3, 0).rgb + bias)), -128, 127) + 128;\n"
                                  "\n"
                                  "    // Pack B, R, G channels into 3 uints (little-endian byte packing)\n"
                                  "    uint bVal = (uint(rgb0.b)      ) | (uint(rgb1.b) << 8) | (uint(rgb2.b) << 16) | (uint(rgb3.b) << 24);\n"
                                  "    uint rVal = (uint(rgb0.r)      ) | (uint(rgb1.r) << 8) | (uint(rgb2.r) << 16) | (uint(rgb3.r) << 24);\n"
                                  "    uint gVal = (uint(rgb0.g)      ) | (uint(rgb1.g) << 8) | (uint(rgb2.g) << 16) | (uint(rgb3.g) << 24);\n"
                                  "\n"
                                  "    // Write directly (no atomics, no pre-clear needed)\n"
                                  "    outputData.result[out_idx_uint] = bVal;\n"
                                  "    outputData.result[out_idx_uint + plane_stride_uint] = rVal;\n"
                                  "    outputData.result[out_idx_uint + (plane_stride_uint << 1)] = gVal;\n"
                                  "}";

// --- Globals for Optimization ---
const int NUM_BUFFERS = 3;
GLuint pbos[NUM_BUFFERS] = {0};
GLsync fences[NUM_BUFFERS] = {0};
int current_index = 0;

// --- GL Program Handles ---
GLuint processProgram;
GLint yuvTexLoc = 0;
const GLsizei camera_width = 1280;
const GLsizei camera_height = 720;
const GLsizei pixelCount = camera_width * camera_height;
const int output_size_A = 6 * 300;              // [left, top, right, bottom, max_score, max_indices] * yolo_num_boxes
const int output_size_B = 294 * 518;        // depth_pixels
const int pixelCount_rgb = 3 * pixelCount;
const int gpu_num_group = 16;               // Customize it to fit your device's specifications.
const GLsizei rgbSize_i8 = pixelCount_rgb * sizeof(uint8_t);
const GLsizei workGroupCountX = camera_width / gpu_num_group;
const GLsizei workGroupCountY = camera_height / gpu_num_group;

const OrtApi *ort_runtime_A;
OrtSession *session_model_A;
OrtRunOptions *run_options_A;
std::vector<const char *> input_names_A;
std::vector<const char *> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue *> input_tensors_A;
std::vector<OrtValue *> output_tensors_A;
const OrtApi *ort_runtime_B;
OrtSession *session_model_B;
OrtRunOptions *run_options_B;
std::vector<const char *> input_names_B;
std::vector<const char *> output_names_B;
std::vector<std::vector<std::int64_t>> input_dims_B;
std::vector<std::vector<std::int64_t>> output_dims_B;
std::vector<ONNXTensorElementDataType> input_types_B;
std::vector<ONNXTensorElementDataType> output_types_B;
std::vector<OrtValue *> input_tensors_B;
std::vector<OrtValue *> output_tensors_B;
const std::string file_name_A = "Model_Yolo_v10s.ort";
const std::string file_name_B = "Depth_Anything_Metric_V2.ort";
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";


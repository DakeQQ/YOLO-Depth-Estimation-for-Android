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
                                  "const int camera_width = 1280;\n"
                                  "const int camera_height = 720;\n"
                                  "const uint pixel_count = uint(camera_width * camera_height);\n"
                                  "layout(binding = 0) uniform samplerExternalOES yuvTex;\n"
                                  "layout(std430, binding = 1) buffer Output {\n"
                                  "    uint result[];\n"
                                  "} outputData;\n"
                                  "const vec3 bias = vec3(-0.15, -0.5, -0.2);\n"
                                  "const mat3 YUVtoRGBMatrix = mat3(127.5, 0.0, 1.402 * 127.5, "
                                  "                                 127.5, -0.344136 * 127.5, -0.714136 * 127.5, "
                                  "                                 127.5, 1.772 * 127.5, 0.0);\n"
                                  "void main() {\n"
                                  "    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);\n"
                                  "    if (texelPos.x >= camera_width || texelPos.y >= camera_height) return;\n"
                                  "    ivec3 rgb = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, texelPos, 0).rgb + bias)), -128, 127) + 128;\n"
                                  "    uint pixel_idx = uint(texelPos.y * camera_width + texelPos.x);\n"
                                  // Planar BGR layout: B...BR...RG...G, requires atomic operations to write bytes into uint buffer.\n"
                                  // B channel\n"
                                  "    uint b_idx = pixel_idx / 4u;\n"
                                  "    uint b_shift = (pixel_idx % 4u) * 8u;\n"
                                  "    atomicOr(outputData.result[b_idx], uint(rgb.b) << b_shift);\n"
                                  // R channel\n"
                                  "    uint r_idx = (pixel_idx + pixel_count) / 4u;\n"
                                  "    uint r_shift = ((pixel_idx + pixel_count) % 4u) * 8u;\n"
                                  "    atomicOr(outputData.result[r_idx], uint(rgb.r) << r_shift);\n"
                                  // G channel\n"
                                  "    uint g_idx = (pixel_idx + 2u * pixel_count) / 4u;\n"
                                  "    uint g_shift = ((pixel_idx + 2u * pixel_count) % 4u) * 8u;\n"
                                  "    atomicOr(outputData.result[g_idx], uint(rgb.g) << g_shift);\n"
                                  "}";
GLuint pbo_A = 0;
GLuint computeProgram = 0;
GLint yuvTexLoc = 0;
const GLsizei camera_width = 1280;
const GLsizei camera_height = 720;
const GLsizei pixelCount = camera_width * camera_height;
const int output_size_A = 6 * 300;              // [left, top, right, bottom, max_score, max_indices] * yolo_num_boxes
const int output_size_B = 294 * 518;            // depth_pixels
const int pixelCount_rgb = 3 * pixelCount;
const int gpu_num_group = 16;                   // Customize it to fit your device's specifications.
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


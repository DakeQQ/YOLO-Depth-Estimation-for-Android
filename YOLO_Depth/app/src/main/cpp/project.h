#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include <GLES3/gl32.h>

const char* computeShaderSource = "#version 320 es\n"
                                  "#extension GL_OES_EGL_image_external_essl3 : require\n"
                                  "precision mediump float;\n"
                                  "layout(local_size_x = 16, local_size_y = 16) in;\n"  // gpu_num_group=16, Customize it to fit your device's specifications.
                                  "const int camera_width = 1280;\n"                    // camera_width
                                  "const int camera_height = 720;\n"                    // camera_height
                                  "layout(binding = 0) uniform samplerExternalOES yuvTex;\n"
                                  "layout(std430, binding = 1) buffer Output {\n"
                                  "    int result[camera_height * camera_width];\n"     // pixelCount
                                  "} outputData;\n"
                                  "const vec3 bias = vec3(-0.15, -0.5, -0.2);\n"
                                  "const mat3 YUVtoRGBMatrix = mat3(127.5, 0.0, 1.402 * 127.5, "
                                  "                                 127.5, -0.344136 * 127.5, -0.714136 * 127.5, "
                                  "                                 127.5, 1.772 * 127.5, 0.0);\n"
                                  "void main() {\n"
                                  "    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);\n"
                                  "    ivec3 rgb = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, texelPos, 0).rgb + bias)), -128, 127) + 128;\n"
                                  // Use uint8 packing the pixels, it would be 1.6 times faster than using float32 buffer.
                                  "    outputData.result[texelPos.y * camera_width + texelPos.x] = (rgb.b << 16) | (rgb.r << 8) | (rgb.g);\n"
                                  "}";
GLuint pbo_A = 0;
GLuint computeProgram = 0;
GLint yuvTexLoc = 0;
const GLsizei camera_width = 1280;
const GLsizei camera_height = 720;
const GLsizei pixelCount = camera_width * camera_height;
const int output_size_A = 6 * 3024;         // [x, y, w, h, max_score, max_indices] * yolo_num_boxes
const int output_size_B = 294 * 518;        // depth_pixels
const int pixelCount_rgb = 3 * pixelCount;
const int gpu_num_group = 16;               // Customize it to fit your device's specifications.
const GLsizei rgbSize = pixelCount_rgb * sizeof(float);
const GLsizei rgbSize_int = pixelCount * sizeof(int);
const GLsizei rgbSize_i8 = pixelCount_rgb * sizeof(uint8_t);
const GLsizei workGroupCountX = camera_width / gpu_num_group;
const GLsizei workGroupCountY = camera_height / gpu_num_group;

const OrtApi* ort_runtime_A;
OrtSession* session_model_A;
OrtRunOptions *run_options_A;
std::vector<const char*> input_names_A;
std::vector<const char*> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue*> input_tensors_A;
std::vector<OrtValue*> output_tensors_A;
const OrtApi* ort_runtime_B;
OrtSession* session_model_B;
OrtRunOptions *run_options_B;
std::vector<const char*> input_names_B;
std::vector<const char*> output_names_B;
std::vector<std::vector<std::int64_t>> input_dims_B;
std::vector<std::vector<std::int64_t>> output_dims_B;
std::vector<ONNXTensorElementDataType> input_types_B;
std::vector<ONNXTensorElementDataType> output_types_B;
std::vector<OrtValue*> input_tensors_B;
std::vector<OrtValue*> output_tensors_B;
const std::string file_name_A = "Model_YOLO_v12_n_f32.ort";
const std::string file_name_B = "Depth_Anything_Metric_V2.ort";
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";

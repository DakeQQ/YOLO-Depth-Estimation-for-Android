#include "project.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_A;
    OrtSessionOptions *session_options_A;
    {
        std::vector<char> fileBuffer;
        off_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_A.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_A, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_A);
        ort_runtime_A->CreateSessionOptions(&session_options_A);
        ort_runtime_A->CreateRunOptions(&run_options_A);
        ort_runtime_A->AddRunConfigEntry(run_options_A, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_A->AddRunConfigEntry(run_options_A, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_A->DisableProfiling(session_options_A);
        ort_runtime_A->EnableCpuMemArena(session_options_A);
        ort_runtime_A->EnableMemPattern(session_options_A);
        ort_runtime_A->SetSessionExecutionMode(session_options_A, ORT_SEQUENTIAL);
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 2);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "1");
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,2");                                                            // It is the best cost/performance (C/P) value setting for running the Yolo on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 2);                                              // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);                     // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_EXTENDED instead of ORT_ENABLE_BLL.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdq_matmulnbits_accuracy_level",
                                             "4");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);                                                 // Keep the same value as above.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning", "0");           // Set to 0.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "1");                // Set to 1.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities", "1,2,3,4");  // Use ',' to split the #core
            ort_runtime_A->SetIntraOpNumThreads(session_options_A, 1);  // Set to 1.
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_A->CreateSessionFromArray(ort_env_A, fileBuffer.data(), fileSize, session_options_A, &session_model_A);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_A->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_A->SessionGetInputCount(session_model_A, &amount_of_input);
    input_names_A.resize(amount_of_input);
    input_dims_A.resize(amount_of_input);
    input_types_A.resize(amount_of_input);
    input_tensors_A.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetInputName(session_model_A, i, allocator, &name);
        input_names_A[i] = name;
        ort_runtime_A->SessionGetInputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        input_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, input_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetOutputName(session_model_A, i, allocator, &name);
        output_names_A[i] = name;
        ort_runtime_A->SessionGetOutputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        output_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, output_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1B(JNIEnv *env, jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_B;
    OrtSessionOptions *session_options_B;
    {
        std::vector<char> fileBuffer;
        off_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_B.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_B, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_B = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_B->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_B);
        ort_runtime_B->CreateSessionOptions(&session_options_B);
        ort_runtime_B->CreateRunOptions(&run_options_B);
        ort_runtime_B->AddRunConfigEntry(run_options_B, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_B->AddRunConfigEntry(run_options_B, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_B->DisableProfiling(session_options_B);
        ort_runtime_B->EnableCpuMemArena(session_options_B);
        ort_runtime_B->EnableMemPattern(session_options_B);
        ort_runtime_B->SetSessionExecutionMode(session_options_B, ORT_SEQUENTIAL);
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base",
                                             "2");                                                              // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "3,5;4,6");                                                        // It is the best cost/performance (C/P) value setting for running the Depth on my device.
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 3);                                              // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);                     // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_EXTENDED instead of ORT_ENABLE_BLL.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");                                                              //
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdq_matmulnbits_accuracy_level",
                                             "4");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);                                                 // Keep the same value as above.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning", "0");           // Set to 0.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "1");                // Set to 1.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities", "1,2,3,4");  // Use ',' to split the #core
            ort_runtime_B->SetIntraOpNumThreads(session_options_B, 1);  // Set to 1.
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_B->CreateSessionFromArray(ort_env_B, fileBuffer.data(), fileSize, session_options_B, &session_model_B);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_B->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_B->SessionGetInputCount(session_model_B, &amount_of_input);
    input_names_B.resize(amount_of_input);
    input_dims_B.resize(amount_of_input);
    input_types_B.resize(amount_of_input);
    input_tensors_B.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetInputName(session_model_B, i, allocator, &name);
        input_names_B[i] = name;
        ort_runtime_B->SessionGetInputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        input_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, input_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_B->SessionGetOutputCount(session_model_B, &amount_of_output);
    output_names_B.resize(amount_of_output);
    output_dims_B.resize(amount_of_output);
    output_types_B.resize(amount_of_output);
    output_tensors_B.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetOutputName(session_model_B, i, allocator, &name);
        output_names_B[i] = name;
        ort_runtime_B->SessionGetOutputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        output_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, output_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Process_1Texture(JNIEnv *env, jclass clazz) {
    glUseProgram(computeProgram);
    glDispatchCompute(workGroupCountX, workGroupCountY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    jintArray final_results = env->NewIntArray(pixelCount);
    env->SetIntArrayRegion(final_results, 0, pixelCount, (jint*) glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, rgbSize_int, GL_MAP_READ_BIT));
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    return final_results;
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Process_1Init(JNIEnv *env, jclass clazz, jint texture_id) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSource, nullptr);
    glCompileShader(computeShader);
    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    glDeleteShader(computeShader);
    yuvTexLoc = glGetUniformLocation(computeProgram, "yuvTex");
    glUniform1i(yuvTexLoc, 0);
    glGenBuffers(1, &pbo_A);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_A);
    glBufferData(GL_PIXEL_PACK_BUFFER, rgbSize, nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pbo_A);
    glBindImageTexture(0, static_cast<GLuint> (texture_id), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA);
}
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1YOLO(JNIEnv *env, jclass clazz,
                                                       jbyteArray pixel_values) {
    jbyte* pixels = env->GetByteArrayElements(pixel_values, nullptr);
    OrtMemoryInfo *memory_info;
    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info, reinterpret_cast<void*>(pixels), rgbSize_i8,
            input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0], &input_tensors_A[0]);
    ort_runtime_A->ReleaseMemoryInfo(memory_info);
    ort_runtime_A->Run(session_model_A, nullptr, input_names_A.data(), (const OrtValue* const*) input_tensors_A.data(),
                       input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                       output_tensors_A.data());
    void* output_tensors_buffer_A;
    ort_runtime_A->GetTensorMutableData(output_tensors_A[0], &output_tensors_buffer_A);
    jfloatArray final_results = env->NewFloatArray(output_size_A);
    env->SetFloatArrayRegion(final_results, 0, output_size_A, reinterpret_cast<float*> (output_tensors_buffer_A));
    return final_results;
}


extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1Depth(JNIEnv *env, jclass clazz,
                                                       jbyteArray pixel_values) {
    jbyte* pixels = env->GetByteArrayElements(pixel_values, nullptr);
    OrtMemoryInfo *memory_info;
    ort_runtime_B->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info, reinterpret_cast<void*>(pixels), rgbSize_i8,
            input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0], &input_tensors_B[0]);
    ort_runtime_B->ReleaseMemoryInfo(memory_info);
    ort_runtime_B->Run(session_model_B, nullptr, input_names_B.data(), (const OrtValue* const*) input_tensors_B.data(),
                       input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                       output_tensors_B.data());
    void* output_tensors_buffer_B;
    ort_runtime_B->GetTensorMutableData(output_tensors_B[0], &output_tensors_buffer_B);
    jfloatArray final_results = env->NewFloatArray(output_size_B);
    env->SetFloatArrayRegion(final_results, 0, output_size_B, reinterpret_cast<float*> (output_tensors_buffer_B));
    return final_results;
}

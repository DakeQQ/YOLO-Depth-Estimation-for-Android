import os
import gc
import onnx
import subprocess
import onnx.version_converter
from onnxslim import slim
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model


# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Quanted"                   # The quanted folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_quanted.onnx")     # The quanted model stored path.
use_gpu = False                                                                  # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                # ['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider']


# Preprocess, it also cost alot of memory during preprocess, you can close this command and keep quanting. Call subprocess may get permission failed on Windows system.
# (optional process)
# subprocess.run([f'python -m onnxruntime.quantization.preprocess --auto_merge --all_tensors_to_one_file --input {model_path} --output {quanted_folder_path}'], shell=True)


# Start Quantize
quantize_dynamic(
    model_input=model_path,
    model_output=quanted_model_path,
    per_channel=True,                                        # True for model accuracy but cost a lot of time during quanting process.
    reduce_range=False,                                      # True for some x86_64 platform.
    weight_type=QuantType.QUInt8,                            # It is recommended using uint8 + Symmetric False
    extra_options={'ActivationSymmetric': False,             # True for inference speed. False may keep more accuracy.
                   'WeightSymmetric': False,                 # True for inference speed. False may keep more accuracy.
                   'EnableSubgraph': True,                   # True for more quant.
                   'ForceQuantizeNoInputCheck': False,       # True for more quant.
                   'MatMulConstBOnly': False                 # False for more quant. Sometime, the inference speed may get worse.
                   },
    nodes_to_exclude=None,                                   # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
    use_external_data_format=True                            # Save the model into two parts.
)


# onnxslim
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,                                     # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# transformers.optimizer
model = optimize_model(quanted_model_path,
                       use_gpu=use_gpu,
                       opt_level=2,
                       num_heads=4 if "12" in model_path else 0,      # For v12 series
                       hidden_size=144 if "12" in model_path else 0,  # For v12 series
                       provider=provider,
                       verbose=False,
                       model_type='bert')
model.save_model_to_file(quanted_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,                                       # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# Upgrade the Opset version. (optional process)
model = onnx.load(quanted_model_path)
model = onnx.version_converter.convert_version(model, 21)
onnx.save(model, quanted_model_path, save_as_external_data=False)
del model
gc.collect()


# Convert the simplified model to ORT format.
optimization_style = "Runtime"          # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
target_platform = "arm"                 # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
# Call subprocess may get permission failed on Windows system.
subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)


import os
import gc
import onnx
import subprocess
import onnx.version_converter
from onnxslim import slim
from onnxconverter_common import float16
from onnxruntime.transformers.optimizer import optimize_model


# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Quanted"                   # The quanted folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_quanted.onnx")     # The quanted model stored path.
use_gpu = True                                                                   # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                # ['CPUExecutionProvider', 'CUDAExecutionProvider']


# Preprocess, it also cost alot of memory during preprocess, you can close this command and keep quanting. Call subprocess may get permission failed on Windows system.
# (optional process)
# subprocess.run([f'python -m onnxruntime.quantization.preprocess --auto_merge --all_tensors_to_one_file --input {model_path} --output {quanted_folder_path}'], shell=True)


# Start Weight-Only Quantize
block_size = 256            # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
symmetric = False           # False may get more accuracy.
accuracy_level = 2          # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
bits = 4                    # [2, 4, 8]
quant_method = 'default'    # ["default", "hqq", "rtn", "gptq"]; default is recommended, or you will get errors.
quant_format = 'QOperator'  # ["QOperator", "QDQ"]; QOperator format quantizes the model with quantized operators directly.  QDQ format quantize the model by inserting DeQuantizeLinear before the MatMul.,
nodes_to_exclude = None     # Specify the unsupported op type, for example: ReduceMean
# Call subprocess may get permission failed on Windows system.
subprocess.run([f'python -m onnxruntime.quantization.matmul_4bits_quantizer --input_model {model_path} --output_model {quanted_model_path} --block_size {block_size} --symmetric {symmetric} --accuracy_level {accuracy_level} --bits {bits} --quant_method {quant_method} --quant_format {quant_format} --nodes_to_exclude {nodes_to_exclude}'], shell=True)


# Convert the fp32 to fp16
model = onnx.load(quanted_model_path)
model = float16.convert_float_to_float16(model,
                                         min_positive_val=1e-7,
                                         max_finite_val=65504,
                                         keep_io_types=True,         # True for keep original input format.
                                         disable_shape_infer=False,  # False for more optimize.
                                         op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'Resize'],  # The op type list for skip the conversion. These are known unsupported op type for fp16.
                                         node_block_list=None)       # The node name list for skip the conversion.

# onnxslim
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,                                             # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# transformers.optimizer
model = optimize_model(quanted_model_path,
                       use_gpu=use_gpu,
                       opt_level=2,                                   # If use NPU-HTP, opt_level <=1
                       num_heads=4 if "12" in model_path else 0,      # For v12 series
                       hidden_size=144 if "12" in model_path else 0,  # For v12 series
                       provider=provider,
                       verbose=False,
                       model_type='bert')
model.convert_float_to_float16(
    keep_io_types=True,
    force_fp16_initializers=True,
    use_symbolic_shape_infer=True,                                    # True for more optimize but may get errors.
    op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range']
)
model.save_model_to_file(quanted_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,                                             # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# Upgrade the Opset version. (optional process)
model = onnx.load(quanted_model_path)
model = onnx.version_converter.convert_version(model, 21)
onnx.save(model, quanted_model_path, save_as_external_data=False)

# It is not recommended to convert an FP16 ONNX model to the ORT format because this process adds a Cast operation to convert the FP16 process back to FP32.

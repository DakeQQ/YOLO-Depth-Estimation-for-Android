import torch
import shutil
from depth_config import EXPORT_MODEL_ENCODER_TYPE, EXPORT_DEPTH_INPUT_SIZE, MAX_DEPTH
import sys
import os

# Set the path
model_path = "C:/Users/dake/Desktop/Depth_Anything_Metric_V2/depth_anything_v2_metric_hypersim_vits.pth"  # Specify the path where the depth model stored. Must match the "EXPORT_MODEL_ENCODER_TYPE".
output_path = "./Depth_Anything_Metric_V2.onnx"
depth_metric_path = "C:/Users/Downloads/Depth-Anything-V2-main/metric_depth/depth_anything_v2/"  # Specify the path where the Depth-Anything-V2 github project downloaded.
config_path = "./depth_config.py"  # The depth_config.py path.
modeifid_path = "./modeling_modified/"  # The modeling_modified folder path.

# Replace the original source code.
shutil.copy(modeifid_path + "dpt.py", depth_metric_path + "dpt.py")
shutil.copy(modeifid_path + "dinov2.py", depth_metric_path + "dinov2.py")
shutil.copy(modeifid_path + "mlp.py", depth_metric_path + "dinov2_layers/mlp.py")
shutil.copy(modeifid_path + "patch_embed.py", depth_metric_path + "dinov2_layers/patch_embed.py")
shutil.copy(modeifid_path + "attention.py", depth_metric_path + "dinov2_layers/attention.py")
shutil.copy(config_path, modeifid_path + "depth_config.py")
sys.path.append(os.path.dirname(os.path.abspath(depth_metric_path)))

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

from depth_anything_v2.dpt import DepthAnythingV2
model = DepthAnythingV2(**{**model_configs[EXPORT_MODEL_ENCODER_TYPE], 'max_depth': MAX_DEPTH})
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to('cpu').eval()
images = torch.ones(EXPORT_DEPTH_INPUT_SIZE, dtype=torch.float32)
torch.onnx.export(
    model,
    images,
    output_path,
    input_names=['images'],
    output_names=['outputs'],
    do_constant_folding=True,
    opset_version=17)

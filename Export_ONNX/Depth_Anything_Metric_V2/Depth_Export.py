import torch
import shutil
from depth_config import EXPORT_MODEL_ENCODER_TYPE, EXPORT_DEPTH_INPUT_SIZE, MAX_DEPTH
import sys
import os
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =================================================================================
# 1. SETUP AND ONNX EXPORT (Your original code)
# =================================================================================

# --- Paths and Configuration ---
# Specify the path where the depth model is stored. Must match "EXPORT_MODEL_ENCODER_TYPE".
model_path = "/home/DakeQQ/Downloads/depth_anything_v2_metric_hypersim_vits.pth"
# Specify the path where the Depth-Anything-V2 GitHub project was downloaded.
depth_metric_path = "/home/DakeQQ/Downloads/Depth-Anything-V2-main/metric_depth/depth_anything_v2"
# Path for the output ONNX model.
output_path = "./Depth_Anything_Metric_V2.onnx"
# Path to the image you want to test inference on.
test_image_path = "./test.jpg"

# --- Code Modification and Model Loading ---
# Replace the original source code.
print("Copying modified model files...")
shutil.copy("./modeling_modified/dpt.py", os.path.join(depth_metric_path, "dpt.py"))
shutil.copy("./modeling_modified/dinov2.py", os.path.join(depth_metric_path, "dinov2.py"))
shutil.copy("./modeling_modified/mlp.py", os.path.join(depth_metric_path, "dinov2_layers/mlp.py"))
shutil.copy("./modeling_modified/patch_embed.py", os.path.join(depth_metric_path, "dinov2_layers/patch_embed.py"))
shutil.copy("./modeling_modified/attention.py", os.path.join(depth_metric_path, "dinov2_layers/attention.py"))
shutil.copy("./depth_config.py", "./modeling_modified/depth_config.py")
sys.path.append(os.path.dirname(os.path.abspath(depth_metric_path)))
print("Finished copying files.")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

from depth_anything_v2.dpt import DepthAnythingV2

print("Loading PyTorch model...")
model = DepthAnythingV2(**{**model_configs[EXPORT_MODEL_ENCODER_TYPE], 'max_depth': MAX_DEPTH})
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to('cpu').eval()
print("PyTorch model loaded.")

# --- ONNX Export ---
# Create a dummy input tensor with the correct shape and data type.
# The `uint8` dtype is important as the model expects a raw image tensor.
dummy_input = torch.ones(EXPORT_DEPTH_INPUT_SIZE, dtype=torch.uint8)
print(f"Exporting model to ONNX at: {output_path}")
with torch.inference_mode():
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['images'],
        output_names=['depth'],
        do_constant_folding=True,
        opset_version=17
    )
print("✅ ONNX export complete!")


# =================================================================================
# 2. ONNX RUNTIME INFERENCE TEST
# =================================================================================

print("\nStarting ONNX Runtime inference test...")

# --- Load the ONNX model and prepare the session ---
try:
    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    print("ONNX Runtime session created successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sys.exit()

# --- Preprocess the input image ---
# Get the expected input shape from the config [N, C, H, W]
_, _, input_height, input_width = EXPORT_DEPTH_INPUT_SIZE

# Load the raw image using OpenCV
raw_img = cv2.imread(test_image_path)
if raw_img is None:
    print(f"Error: Could not read image at {test_image_path}")
    sys.exit()

# The model expects RGB, but OpenCV loads images in BGR format, so we convert.
rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# Resize the image to the exact size the model expects.
# cv2.resize expects (width, height).
resized_img = cv2.resize(rgb_img, (input_width, input_height))

# The model's ONNX graph expects a uint8 tensor with the shape (N, C, H, W).
# 1. Add a batch dimension: (H, W, C) -> (1, H, W, C)
# 2. Change layout to (1, C, H, W)
input_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2)

print(f"Input tensor prepared with shape: {input_tensor.shape} and dtype: {input_tensor.dtype}")

# --- Run Inference ---
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

print(f"Running inference on ONNX model...")
onnx_result = ort_session.run([output_name], {input_name: input_tensor})

# The output is a list containing one array. We extract the array.
depth_map_onnx = onnx_result[0]

# The output depth map has a shape of (1, H, W), so we remove the batch dimension.
depth_map_onnx = np.squeeze(depth_map_onnx)
print(f"✅ ONNX inference complete! Output depth map shape: {depth_map_onnx.shape}")


# =================================================================================
# 3. VISUALIZE THE RESULT (Method 2: Matplotlib)
# =================================================================================

print("\nVisualizing results using Matplotlib...")

plt.figure(figsize=(14, 7))

# --- Display Original Image ---
plt.subplot(1, 2, 1)
# Use the RGB image we converted earlier for correct color display in Matplotlib
plt.imshow(rgb_img)
plt.title('Original Image')
plt.axis('off')

# --- Display Depth Heatmap from ONNX Inference ---
plt.subplot(1, 2, 2)
# Use the 'viridis' colormap for a perceptually uniform heatmap.
# imshow can directly handle the floating-point depth array.
plt.imshow(depth_map_onnx, cmap='viridis')
plt.title('Depth Heatmap (from ONNX model)')
# Add a color bar to show the mapping of colors to depth values.
plt.colorbar(label='Depth Metric')
plt.axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

print("✅ Visualization complete.")

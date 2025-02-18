import shutil
import torch
import site


# Import the yolo_config
ultralytics_path = site.getsitepackages()[-1] + "/ultralytics"
shutil.copyfile("./yolo_config.py", ultralytics_path + "/engine/yolo_config.py")
shutil.copyfile("./yolo_config.py", ultralytics_path + "/nn/yolo_config.py")
shutil.copyfile("./yolo_config.py", ultralytics_path + "/nn/modules/yolo_config.py")

# Replace the original source code file.
shutil.copyfile("./modeling_modified/exporter.py", ultralytics_path + "/engine/exporter.py")
shutil.copyfile("./modeling_modified/tasks.py", ultralytics_path + "/nn/tasks.py")
shutil.copyfile("./modeling_modified/head.py", ultralytics_path + "/nn/modules/head.py")

from ultralytics import YOLO

# Load a model, The version number must match EXPORT_YOLO_VERSION.
model = YOLO("yolo11n.pt")   # Load an official model,
# model = YOLO("/Users/Downloads/yolo11n.pt") # Or specify your own model path.

# Export the model
with torch.inference_mode():
  model.export(format='onnx')  # The exported model will save at the current folder.

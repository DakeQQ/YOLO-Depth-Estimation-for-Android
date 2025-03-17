import shutil
import torch
import site

# Import the yolo_config
python_package_path = site.getsitepackages()[-1]
shutil.copyfile("./yolo_config.py", python_package_path + "/ultralytics/engine/yolo_config.py")
shutil.copyfile("./yolo_config.py", python_package_path + "/ultralytics/nn/yolo_config.py")
shutil.copyfile("./yolo_config.py", python_package_path + "/ultralytics/nn/modules/yolo_config.py")

# Replace the original source code file.
shutil.copyfile("./modeling_modified/exporter.py", python_package_path + "/ultralytics/engine/exporter.py")
shutil.copyfile("./modeling_modified/tasks.py", python_package_path + "/ultralytics/nn/tasks.py")
shutil.copyfile("./modeling_modified/head.py", python_package_path + "/ultralytics/nn/modules/head.py")

from ultralytics import YOLO

# Load a model, The version number must match EXPORT_YOLO_VERSION.
model = YOLO("yolo12n.pt")                                 # Load an official model,
# model = YOLO("/Users/Downloads/custom_yolo12n.pt")       # Or specify your own model path. Currently, do not support yolo12-turbo series.

# Export the model
with torch.inference_mode():
    model.export(format='onnx')                 # The exported model will save at the current folder.

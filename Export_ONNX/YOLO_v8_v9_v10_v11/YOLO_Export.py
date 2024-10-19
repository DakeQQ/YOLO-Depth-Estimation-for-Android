import shutil

config_path = './yolo_config.py'  # The path where the yolo_config.py stored.
modified_exporter_path = './modeling_modified/exporter.py'  # The path where the modified exporter.py stored.
modified_tasks_path = './modeling_modified/tasks.py'  # The path where the modified tasks.py stored.
modified_head_path = './modeling_modified/head.py'  # The path where the modified head.py stored.
ultralytics_path = '/Users/PycharmProjects/.venv/lib/python3.11/site-packages/ultralytics/'  # The original ultralytics path which was stored in the python packages.

# Import the yolo_config
shutil.copyfile(config_path, ultralytics_path + "engine/yolo_config.py")
shutil.copyfile(config_path, ultralytics_path + "nn/yolo_config.py")
shutil.copyfile(config_path, ultralytics_path + "nn/modules/yolo_config.py")

# Replace the original source code file.
shutil.copyfile(modified_exporter_path, ultralytics_path + "engine/exporter.py")
shutil.copyfile(modified_tasks_path, ultralytics_path + "nn/tasks.py")
shutil.copyfile(modified_head_path, ultralytics_path + "nn/modules/head.py")

from ultralytics import YOLO

# Load a model, The version number must match EXPORT_YOLO_VERSION.
model = YOLO("yolo11n.pt")   # Load an official model,

# model = YOLO("/Users/Downloads/yolo11n.pt") # Or specify your own model path.

# Export the model
with torch.no_grad():
  model.export(format='onnx')  # The exported model will save at the current folder.

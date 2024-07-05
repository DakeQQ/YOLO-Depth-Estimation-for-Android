import shutil

config_path = './yolo_config.py'  # The path where the yolo_config.py stored.
modified_exporter_path = './modeling_modified/exporter.py'  # The path where the modified exporter.py stored.
modified_tasks_path = './modeling_modified/tasks.py'  # The path where the modified tasks.py stored.
ultralytics_engine_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/ultralytics/engine/'  # The original ultralyticss/engine pathwhich was stored in the python packages.
ultralytics_nn_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/ultralytics/nn/'  # The original ultralytics/nn path which was stored in the python packages.

# Import the yolo_config
shutil.copyfile(config_path, ultralytics_engine_path + "yolo_config.py")
shutil.copyfile(config_path, ultralytics_nn_path + "yolo_config.py")

# Replace the original source code file.
shutil.copyfile(modified_exporter_path, ultralytics_engine_path + "exporter.py")
shutil.copyfile(modified_tasks_path, ultralytics_nn_path + "tasks.py")

from ultralytics import YOLO
# Load a model
model = YOLO("yolov10n.pt")  # Load an official model, or specify your own model path. The version number must match EXPORT_YOLO_VERSION.

# Export the model
model.export(format='onnx')  # The exported model will save at the current folder.

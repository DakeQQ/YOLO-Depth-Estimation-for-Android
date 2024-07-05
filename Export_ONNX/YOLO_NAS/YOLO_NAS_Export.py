import torch
import shutil
from yolo_config import EXPORT_YOLO_INPUT_SIZE


config_path = './yolo_config.py'  # The path where the yolo_config.py stored.
modified_path = './modeling_modified/'
output_path = "./yolo_nas_s.onnx"
super_gradients_path = "C:/Users/dake/.conda/envs/python_39/Lib/site-packages/super_gradients/"

# Replace the original source code file.
path_0 = super_gradients_path + "modules/qarepvgg_block.py"
path_1 = super_gradients_path + "training/models/detection_models/yolo_nas/yolo_stages.py"
path_2 = super_gradients_path + "training/models/detection_models/customizable_detector.py"
path_3 = super_gradients_path + "training/models/detection_models/yolo_nas/dfl_heads.py"
path_4 = super_gradients_path + "training/utils/bbox_utils.py"
path_5 = super_gradients_path + "training/models/detection_models/yolo_nas/panneck.py"
shutil.copyfile(modified_path + "qarepvgg_block.py", path_0)
shutil.copyfile(modified_path + "yolo_stages.py", path_1)
shutil.copyfile(modified_path + "customizable_detector.py", path_2)
shutil.copyfile(modified_path + "dfl_heads.py", path_3)
shutil.copyfile(modified_path + "bbox_utils.py", path_4)
shutil.copyfile(modified_path + "panneck.py", path_5)
shutil.copyfile(config_path, modified_path + config_path)

from super_gradients.training import models
from super_gradients.common.object_names import Models

# Load a model
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").eval()

# Export the model
dummy_input = torch.ones(EXPORT_YOLO_INPUT_SIZE, dtype=torch.float32)
torch.onnx.export(model,
                  dummy_input,
                  output_path,
                  input_names=['images'],
                  output_names=['output'],
                  opset_version=17,
                  do_constant_folding=True)

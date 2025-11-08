import torch
import shutil
import site
from yolo_config import EXPORT_YOLO_INPUT_SIZE

output_path = "./yolo_nas.onnx"

# Replace the original source code file.
python_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/qarepvgg_block.py",
                python_package_path + "/super_gradients/modules/qarepvgg_block.py")
shutil.copyfile("./modeling_modified/yolo_stages.py",
                python_package_path + "/super_gradients/training/models/detection_models/yolo_nas/yolo_stages.py")
shutil.copyfile("./modeling_modified/customizable_detector.py",
                python_package_path + "/super_gradients/training/models/detection_models/customizable_detector.py")
shutil.copyfile("./modeling_modified/dfl_heads.py",
                python_package_path + "/super_gradients/training/models/detection_models/yolo_nas/dfl_heads.py")
shutil.copyfile("./modeling_modified/bbox_utils.py",
                python_package_path + "/super_gradients/training/utils/bbox_utils.py")
shutil.copyfile("./modeling_modified/panneck.py",
                python_package_path + "/super_gradients/training/models/detection_models/yolo_nas/panneck.py")
shutil.copyfile("./yolo_config.py", "./modeling_modified/yolo_config.py")

from super_gradients.training import models
from super_gradients.common.object_names import Models

# Load a model
model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco").eval()

# Export the model
dummy_input = torch.ones(EXPORT_YOLO_INPUT_SIZE, dtype=torch.uint8)
print("Export Start")
with torch.inference_mode():
    torch.onnx.export(model,
                      (dummy_input,),
                      output_path,
                      input_names=['images'],
                      output_names=['output'],
                      opset_version=17,
                      dynamo=False,
                      do_constant_folding=True)
print("Export Done!")


EXPORT_YOLO_VERSION = 11  # Specify the YOLO version to export. The version must match the loaded model.
EXPORT_YOLO_INPUT_SIZE = [1, 3, 720, 1280]  # Input image shape. Batch size must be 1. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
EXPORT_YOLO_RESIZE = (288, 512)  # Resize input to this shape to accelerate YOLO inference. Maintain the same W/H ratio as the input images. Should be a multiple of 32.



EXPORT_MODEL_ENCODER_TYPE = 'vits'
EXPORT_DEPTH_INPUT_SIZE = [1, 3, 720, 1280]     # Input image shape. Batch size must be 1. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
EXPORT_DEPTH_RESIZE = (294, 518)                # Must be a multiple of "14" and less than 518. Resize input to this shape to accelerate Depth inference. Maintain the same W/H ratio as the input images.
MAX_DEPTH = 20                                  # 20 for indoor model, 80 for outdoor model

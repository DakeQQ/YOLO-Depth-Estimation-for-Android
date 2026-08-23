# YOLO Series ONNX Export

This repository is dedicated to exporting supported YOLO checkpoints to ONNX. It no longer contains an Android application, Android deployment code, or a general-purpose monocular depth-estimation pipeline.

The main entry point is [Export_ONNX/Export_YOLO.py](Export_ONNX/Export_YOLO.py). The repository also includes optional tools for optimizing exported models and validating them with task-aware ONNX Runtime inference.

## Supported Models

| Series | Detect | Segment | Pose | OBB | Classify | Semantic | Depth |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ultralytics YOLOv8 | Yes | Yes | Yes | Yes | Yes | - | - |
| Ultralytics YOLOv9 | Yes | Yes | Yes | Yes | Yes | - | - |
| Ultralytics YOLOv10 | Yes | Yes | Yes | Yes | Yes | - | - |
| Ultralytics YOLOv11 | Yes | Yes | Yes | Yes | Yes | - | - |
| Ultralytics YOLOv12 | Yes | Yes | Yes | Yes | Yes | - | - |
| Ultralytics YOLO26 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| YOLO-NAS S/M/L | Yes | - | - | - | - | - | - |

Semantic and depth export are supported only when those heads are part of a YOLO26 checkpoint. They do not represent a separate depth-model export pipeline.

## Repository Layout

```text
Export_ONNX/
├── Export_YOLO.py              # Export PyTorch YOLO checkpoints to ONNX
├── Optimize_ONNX.py            # Optimize or quantize exported ONNX models
├── Inference_YOLO_ONNX.py      # Validate and render ONNX Runtime results
├── demo_images/                # Task-specific inference inputs and results
└── models/                     # Checkpoints and generated ONNX artifacts by task
```

## Requirements

Use a separate environment for each route when their dependency constraints conflict.

### Ultralytics export

- Python 3.11 or newer
- `torch`
- `onnx`
- `ultralytics==8.4.123` (verified version)

### YOLO-NAS export

- Python 3.11
- `torch`
- `onnx`
- `super-gradients==3.7.1` (verified version)

SuperGradients 3.7.1 pins older `numpy`, `onnx`, and `onnxruntime` releases, so its exact dependency set should not share an environment with the modern optimizer stack.

### Optimization

- `numpy`
- `onnx`
- `onnxruntime`
- `onnxslim`

### Inference validation

- `numpy`
- `onnxruntime`
- `opencv-python`

The scripts do not install packages or modify files in `site-packages`.

## Export a Model

Edit the `USER EXPORT CONFIGURATION` block near the top of [Export_ONNX/Export_YOLO.py](Export_ONNX/Export_YOLO.py):

| Setting | Purpose |
| --- | --- |
| `MODEL_FAMILY` | Select `"ultralytics"` or `"yolo_nas"`. |
| `MODEL_PATH` | Select the source `.pt` checkpoint. |
| `MODEL_TASK` | Select a supported task, or use `"auto"` to read it from the checkpoint. |
| `YOLO_VERSION` | Override filename-based version detection for custom names such as `best.pt`. |
| `INPUT_SHAPE` | Set the public static ONNX input shape. |
| `RESIZE_SHAPE` | Set the internal model resize shape. |
| `PUBLIC_INPUT_DTYPE` | Select the public `uint8` or `float32` input contract. |
| `MODEL_PRECISION` | Select the model export precision. |
| `OUTPUT_PATH` | Set an explicit destination, or leave `None` to export beside the checkpoint. |
| `OPSET` | Set the ONNX opset passed to `torch.onnx.export`. |
| `RUN_INFERENCE_DEMO` | Run task-aware inference after a successful export. |

Then run:

```bash
python Export_ONNX/Export_YOLO.py
```

For an official Ultralytics filename such as `yolo26n.pt`, `AUTO_DOWNLOAD_MODEL = True` resolves the selected task suffix and stores a missing checkpoint under `Export_ONNX/models/<task>/`. Explicit custom paths and YOLO-NAS checkpoints must be supplied locally.

The exporter executes the checkpoint's static layer graph instead of relying on a fixed layer count. Package-supported P2/P6 profiles and custom topologies can therefore be exported when their final head implements one of the supported task contracts.

## Default ONNX Contract

The default public input is a static `images` tensor:

```text
dtype: uint8
shape: [1, 3, 720, 1280]
```

Input conversion, bilinear resize, and normalization are embedded in the graph. Each artifact also stores metadata such as the model family, task, resize geometry, output names, class labels, end-to-end box semantics, and pose keypoint shape.

| Task | ONNX outputs |
| --- | --- |
| Ultralytics detect | `output0`: `[1, anchors, 6]` containing box, confidence, and class index |
| Ultralytics segment | `output0`: detection rows plus mask coefficients; `output1`: mask prototypes |
| Ultralytics pose | `output0`: detection rows plus keypoint values |
| Ultralytics OBB | `output0`: `[1, anchors, 7]`, including rotation |
| Ultralytics classify | `output0`: `[1, classes]` probabilities |
| Ultralytics semantic/depth | `output0`: the checkpoint's static dense output map |
| YOLO-NAS detect | `output`: `[1, anchors, 6]` |

Official YOLOv10 and YOLO26 end-to-end heads preserve their native top-k, NMS-free ordering and `xyxy` boxes. Other detection-like heads reduce class channels to confidence and class index while preserving task-specific values.

## Optimize an Export

[Export_ONNX/Optimize_ONNX.py](Export_ONNX/Optimize_ONNX.py) supports these methods:

| Method | Result |
| --- | --- |
| `Q4` | Blocked logical `UINT4` weights using `DequantizeLinear -> Reshape -> Conv` |
| `Q8` | Blocked `UINT8` weights using the same Q/DQ Conv representation |
| `DYNAMIC` | Runtime activation quantization with symmetric per-output-channel weights and `ConvInteger` |
| `F16` | Convert floating-point graph values and weights to float16 |
| `F32` | Run graph optimization without quantization |

Run the synthetic optimizer test, then optimize an exported model:

```bash
python Export_ONNX/Optimize_ONNX.py --self-test
python Export_ONNX/Optimize_ONNX.py --input path/to/model.onnx --method F16
python Export_ONNX/Optimize_ONNX.py --input path/to/model.onnx --method Q8
```

The default output is `<source-stem>_<method>.onnx`. Q4 and Q8 require ONNX opset 21 for blocked `DequantizeLinear`. The optimizer validates rewrite coverage, checks the graph with `onnx.checker`, optionally compares outputs with ONNX Runtime, and publishes the result atomically only after all enabled checks pass.

## Validate an Export

[Export_ONNX/Inference_YOLO_ONNX.py](Export_ONNX/Inference_YOLO_ONNX.py) loads the ONNX artifact and its embedded metadata without importing PyTorch, Ultralytics, SuperGradients, or the exporter.

```bash
python Export_ONNX/Inference_YOLO_ONNX.py --self-test
python Export_ONNX/Inference_YOLO_ONNX.py \
  --model path/to/model.onnx \
  --image path/to/image.jpg \
  --output path/to/result.jpg \
  --no-open
```

When no image is specified, the script selects a prepared image for the model task. It renders boxes and labels for detection, masks for segmentation, keypoints for pose, rotated boxes for OBB, ranked classes for classification, and dense maps for semantic or YOLO26 depth outputs.

## Notes

- Export, optimization, and inference are separate stages. The exporter does not quantize or rewrite the published ONNX graph after export.
- Source checkpoints are never overwritten. Generated artifacts are written through process-owned temporary files and then published atomically.
- `OUTPUT_PATH = None` writes the ONNX model beside the resolved checkpoint.
- Set `RUN_INFERENCE_DEMO = False` when only the exported artifact is required.

## License

See [LICENSE](LICENSE). Portions of the graph behavior are adapted from Ultralytics (AGPL-3.0) and SuperGradients (Apache-2.0); consult those projects' licenses when distributing derived artifacts or software.
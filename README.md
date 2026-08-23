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

---

# YOLO 系列 ONNX 导出

本仓库专用于将受支持的 YOLO 检查点导出为 ONNX。仓库不再包含 Android 应用、Android 部署代码或通用的单目深度估计流水线。

主要入口为 [Export_ONNX/Export_YOLO.py](Export_ONNX/Export_YOLO.py)。仓库还提供了一些可选工具，用于优化已导出的模型，以及通过面向具体任务的 ONNX Runtime 推理进行验证。

## 支持的模型

| 系列 | 检测 | 分割 | 姿态 | OBB | 分类 | 语义分割 | 深度 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ultralytics YOLOv8 | 是 | 是 | 是 | 是 | 是 | - | - |
| Ultralytics YOLOv9 | 是 | 是 | 是 | 是 | 是 | - | - |
| Ultralytics YOLOv10 | 是 | 是 | 是 | 是 | 是 | - | - |
| Ultralytics YOLOv11 | 是 | 是 | 是 | 是 | 是 | - | - |
| Ultralytics YOLOv12 | 是 | 是 | 是 | 是 | 是 | - | - |
| Ultralytics YOLO26 | 是 | 是 | 是 | 是 | 是 | 是 | 是 |
| YOLO-NAS S/M/L | 是 | - | - | - | - | - | - |

仅当 YOLO26 检查点包含语义分割头或深度头时，才支持对应的导出。这些功能并不构成独立的深度模型导出流水线。

## 仓库结构

```text
Export_ONNX/
├── Export_YOLO.py              # 将 PyTorch YOLO 检查点导出为 ONNX
├── Optimize_ONNX.py            # 优化或量化已导出的 ONNX 模型
├── Inference_YOLO_ONNX.py      # 验证并渲染 ONNX Runtime 推理结果
├── demo_images/                # 面向具体任务的推理输入与结果
└── models/                     # 按任务存放检查点和生成的 ONNX 文件
```

## 环境要求

当不同路线的依赖约束存在冲突时，请分别使用独立环境。

### Ultralytics 导出

- Python 3.11 或更高版本
- `torch`
- `onnx`
- `ultralytics==8.4.123`（已验证版本）

### YOLO-NAS 导出

- Python 3.11
- `torch`
- `onnx`
- `super-gradients==3.7.1`（已验证版本）

SuperGradients 3.7.1 固定使用较旧版本的 `numpy`、`onnx` 和 `onnxruntime`，因此不应将其完整依赖集与现代优化工具链安装在同一环境中。

### 优化

- `numpy`
- `onnx`
- `onnxruntime`
- `onnxslim`

### 推理验证

- `numpy`
- `onnxruntime`
- `opencv-python`

这些脚本不会安装软件包，也不会修改 `site-packages` 中的文件。

## 导出模型

编辑 [Export_ONNX/Export_YOLO.py](Export_ONNX/Export_YOLO.py) 顶部附近的 `USER EXPORT CONFIGURATION` 配置块：

| 配置项 | 用途 |
| --- | --- |
| `MODEL_FAMILY` | 选择 `"ultralytics"` 或 `"yolo_nas"`。 |
| `MODEL_PATH` | 选择源 `.pt` 检查点。 |
| `MODEL_TASK` | 选择受支持的任务，或使用 `"auto"` 从检查点中读取。 |
| `YOLO_VERSION` | 对 `best.pt` 等自定义名称覆盖基于文件名的版本检测结果。 |
| `INPUT_SHAPE` | 设置公开的静态 ONNX 输入形状。 |
| `RESIZE_SHAPE` | 设置模型内部的缩放形状。 |
| `PUBLIC_INPUT_DTYPE` | 选择公开输入使用的 `uint8` 或 `float32` 数据约定。 |
| `MODEL_PRECISION` | 选择模型导出精度。 |
| `OUTPUT_PATH` | 设置明确的输出位置，或保留为 `None` 以导出到检查点所在目录。 |
| `OPSET` | 设置传递给 `torch.onnx.export` 的 ONNX opset。 |
| `RUN_INFERENCE_DEMO` | 成功导出后运行面向具体任务的推理。 |

然后运行：

```bash
python Export_ONNX/Export_YOLO.py
```

对于 `yolo26n.pt` 这类官方 Ultralytics 文件名，`AUTO_DOWNLOAD_MODEL = True` 会解析所选任务对应的后缀，并将缺失的检查点保存到 `Export_ONNX/models/<task>/` 下。自定义路径所指向的检查点和 YOLO-NAS 检查点必须在本地提供。

导出器会执行检查点的静态层图，而不是依赖固定的层数。因此，只要最终的模型头实现了某个受支持的任务约定，就可以导出软件包支持的 P2/P6 配置和自定义拓扑。

## 默认 ONNX 约定

默认公开输入是静态的 `images` 张量：

```text
dtype: uint8
shape: [1, 3, 720, 1280]
```

输入转换、双线性缩放和归一化均嵌入图中。每个导出文件还会存储模型系列、任务、缩放几何信息、输出名称、类别标签、端到端边界框语义和姿态关键点形状等元数据。

| 任务 | ONNX 输出 |
| --- | --- |
| Ultralytics 检测 | `output0`：`[1, anchors, 6]`，包含边界框、置信度和类别索引 |
| Ultralytics 分割 | `output0`：检测行及掩码系数；`output1`：掩码原型 |
| Ultralytics 姿态 | `output0`：检测行及关键点值 |
| Ultralytics OBB | `output0`：`[1, anchors, 7]`，包含旋转角度 |
| Ultralytics 分类 | `output0`：`[1, classes]` 概率 |
| Ultralytics 语义分割/深度 | `output0`：检查点的静态稠密输出图 |
| YOLO-NAS 检测 | `output`：`[1, anchors, 6]` |

官方 YOLOv10 和 YOLO26 端到端模型头会保留其原生的 top-k、无 NMS 排序方式和 `xyxy` 边界框。其他检测类模型头会将类别通道归并为置信度和类别索引，同时保留任务特有的值。

## 优化导出模型

[Export_ONNX/Optimize_ONNX.py](Export_ONNX/Optimize_ONNX.py) 支持以下方法：

| 方法 | 结果 |
| --- | --- |
| `Q4` | 使用 `DequantizeLinear -> Reshape -> Conv` 的分块逻辑 `UINT4` 权重 |
| `Q8` | 使用相同 Q/DQ Conv 表示形式的分块 `UINT8` 权重 |
| `DYNAMIC` | 对激活进行运行时量化，并使用对称的逐输出通道权重和 `ConvInteger` |
| `F16` | 将浮点图中的值和权重转换为 float16 |
| `F32` | 执行图优化而不进行量化 |

先运行合成优化器测试，再优化已导出的模型：

```bash
python Export_ONNX/Optimize_ONNX.py --self-test
python Export_ONNX/Optimize_ONNX.py --input path/to/model.onnx --method F16
python Export_ONNX/Optimize_ONNX.py --input path/to/model.onnx --method Q8
```

默认输出路径为 `<source-stem>_<method>.onnx`。Q4 和 Q8 的分块 `DequantizeLinear` 需要 ONNX opset 21。优化器会验证重写覆盖情况，使用 `onnx.checker` 检查计算图，按需通过 ONNX Runtime 比较输出，并且仅在所有已启用的检查通过后以原子方式发布结果。

## 验证导出模型

[Export_ONNX/Inference_YOLO_ONNX.py](Export_ONNX/Inference_YOLO_ONNX.py) 无需导入 PyTorch、Ultralytics、SuperGradients 或导出器，即可加载 ONNX 文件及其中嵌入的元数据。

```bash
python Export_ONNX/Inference_YOLO_ONNX.py --self-test
python Export_ONNX/Inference_YOLO_ONNX.py \
  --model path/to/model.onnx \
  --image path/to/image.jpg \
  --output path/to/result.jpg \
  --no-open
```

未指定图像时，脚本会为模型任务选择一张准备好的图像。它会为检测任务渲染边界框和标签，为分割任务渲染掩码，为姿态任务渲染关键点，为 OBB 任务渲染旋转框，为分类任务渲染排序后的类别，并为语义分割或 YOLO26 深度输出渲染稠密图。

## 说明

- 导出、优化和推理是彼此独立的阶段。导出器不会在导出后量化或重写已发布的 ONNX 图。
- 源检查点绝不会被覆盖。生成的文件会先写入进程专用的临时文件，再以原子方式发布。
- `OUTPUT_PATH = None` 会将 ONNX 模型写入解析后的检查点所在目录。
- 仅需要导出文件时，请设置 `RUN_INFERENCE_DEMO = False`。

## 许可证

请参阅 [LICENSE](LICENSE)。部分计算图行为改编自 Ultralytics（AGPL-3.0）和 SuperGradients（Apache-2.0）；分发派生文件或软件时，请查阅这些项目的许可证。
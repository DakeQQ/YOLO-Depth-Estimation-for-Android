---

# YOLO-Depth-Drivable Estimation for Android

## Overview

This project enables running YOLO series, monocular depth estimation, and drivable area estimation on Android devices. It is optimized for peak performance using models converted from HuggingFace, ModelScope, and GitHub.

## Key Features

1. **Demo Models Available**: 
   - Download demo models from [Google Drive](https://drive.google.com/drive/folders/1MPUvCQCNkjBiHtMjx-eTywetKbkTK7VA?usp=sharing).
   - Place model files in the `assets` folder and unzip `.so` files in `libs/arm64-v8a`.

2. **Supported Models**:
   - **YOLO**: Versions 8, 9, 10, 11, and NAS series.
   - **Depth Estimation**: Depth Anything V2-Metric-Small.
   - **Drivable Area**: TwinLiteNet.

3. **Image Input**:
   - Models accept images with resolution h720*w1280. Hold your phone horizontally.

4. **ONNX Runtime Compatibility**:
   - Models are exported without dynamic axes for better Android compatibility, which may affect x86_64 performance.

5. **GPU Utilization**:
   - Utilizes OpenGL ES 3.x for efficient camera frame processing and GPU compute shaders for YUV to RGB conversion.

6. **Asynchronous Inference**:
   - Processes previous camera frame data for inference, improving efficiency.

7. **YOLO Bounding Box Rendering**:
   - Efficiently rendered using OpenGL ES.

8. **Performance Notes**:
   - Running both YOLO and depth estimation reduces FPS by about 30%.

9. **Export and Configuration**:
   - Update model export methods and ensure configuration files match model weights. Use `UInt8` for quantization to avoid ONNX Runtime errors.

10. **Qualcomm NPU Support**:
    - Only YOLO v9 & NAS series support Qualcomm NPU. NPU libraries must be obtained independently.

11. **Quantization Methods**:
    - Avoid q4(uint4) due to poor ONNX Runtime performance.

12. **Image Preprocessing**:
    - Updated on 2024/10/12 for optimized performance. Set `EXPORT_YOLO_INPUT_SIZE` appropriately for high-resolution screens.

## Project Resources

- [More about the project](https://dakeqq.github.io/overview/)

## 演示结果 Demo Results

- **YOLOv8-n & Depth Anything-Small**
  ![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/yolo_depth.gif?raw=true)

- **YOLOv8-s & Depth Anything-Small**
  ![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/yolo_depth2.gif?raw=true)

- **TwinLiteNet**
  ![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/drivable.gif?raw=true)

---

# 安卓本地运行YOLO+深度(距离)+可行驶区域估计

## 概述

该项目支持在Android设备上运行YOLO系列、单目深度估计和可行驶区域估计。通过从HuggingFace、ModelScope和GitHub转换的模型进行优化以实现最佳性能。

## 主要功能

1. **演示模型**:
   - 从[Google Drive](https://drive.google.com/drive/folders/1MPUvCQCNkjBiHtMjx-eTywetKbkTK7VA?usp=sharing)下载演示模型。
   - 将模型文件放入`assets`文件夹，解压`libs/arm64-v8a`中的`.so`文件。

2. **支持的模型**:
   - **YOLO**: v8, v9, v10, v11, NAS系列。
   - **深度估计**: Depth Anything V2-Metric-Small。
   - **可行驶区域**: TwinLiteNet。

3. **图像输入**:
   - 模型接收分辨率为h720*w1280的图像，需横置手机。

4. **ONNX Runtime兼容性**:
   - 为了更好地适配Android，导出时未使用dynamic-axes，这可能影响x86_64的性能。

5. **GPU利用**:
   - 使用OpenGL ES 3.x高效处理相机帧并通过GPU计算着色器进行YUV到RGB的转换。

6. **异步推理**:
   - 使用前一帧相机数据进行推理，提高效率。

7. **YOLO框渲染**:
   - 通过OpenGL ES高效渲染YOLO框。

8. **性能提示**:
   - 同时运行YOLO和深度估计会使FPS降低约30%。

9. **导出与配置**:
   - 更新模型导出方法并确保配置文件与模型权重匹配。使用`UInt8`进行量化以避免ONNX Runtime错误。

10. **高通NPU支持**:
    - 只有YOLO v9 & NAS系列支持高通NPU。NPU库需自行获取。

11. **量化方法**:
    - 由于ONNX Runtime表现不佳，不建议使用q4(uint4)。

12. **图像预处理**:
    - 2024/10/12更新以优化性能。为高分辨率屏幕设置适当的`EXPORT_YOLO_INPUT_SIZE`。

## 项目资源

- [查看更多项目](https://dakeqq.github.io/overview/)

## GPU Image Preprocess - 图像预处理性能

| OS | Device | Backend | Pixels: h720*w1280<br>Time Cost: ms | Pixels: h1088*w1920<br>Time Cost: ms | 
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-GPU<br>Adreno-740 | 3.5 | 6.5 |
| Harmony 4 | P40 | Kirin_990_5G-GPU<br>Mali-G76 MP16 | 9 | 17 |

## YOLO - 性能 Performance

| OS | Device | Backend | Model | FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-x<br>q8f32 | 3.5 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-l<br>q8f32 | 6 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-m<br>q8f32 | 8 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-s<br>q8f32 | 18 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-n<br>q8f32 | 36 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v10-m<br>q8f32 | 9.5 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v10-s<br>q8f32 | 17.5 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v10-n<br>q8f32 | 35 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v9-C<br>q8f32 | 7 |
| Android 13 | Nubia Z50 | 8_Gen2-NPU<br>(HTPv73) | v9-C<br>f16 | 50+ |
| Android 13 | Nubia Z50 | 8_Gen2-NPU<br>(HTPv73) | v9-M<br>f16 | 60+ |
| Android 13 | Nubia Z50 | 8_Gen2-NPU<br>(HTPv73) | v9-S<br>f16 | 90+ |
| Android 13 | Nubia Z50 | 8_Gen2-NPU<br>(HTPv73) | v9-T<br>f16 | 110+ |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v8-s<br>q8f32 | 21 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v8-n<br>q8f32 | 43 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | NAS-m<br>q8f32 | 9 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | NAS-s<br>q8f32 | 19 |
| Android 13 | Nubia Z50 | 8_Gen2-NPU<br>(HTPv73) | NAS-m<br>f16 | 75+ |
| Android 13 | Nubia Z50 | 8_Gen2-NPU<br>(HTPv73) | NAS-s<br>f16 | 95+ |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v11-x<br>q8f32 | 2.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v11-l<br>q8f32 | 3.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v11-m<br>q8f32 | 5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v11-s<br>q8f32 | 11.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v11-n<br>q8f32 | 23 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v10-m<br>q8f32 | 5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v10-s<br>q8f32 | 9.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v10-n<br>q8f32 | 18.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v9-C<br>q8f32 | 3.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v8-s<br>q8f32 | 10.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v8-n<br>q8f32 | 22 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | NAS-m<br>q8f32 | 5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | NAS-s<br>q8f32 | 9.5 |

## Depth - 性能 Performance

| OS | Device | Backend | Model | FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | Depth Anything-Small<br>q8f32<br>(previous version) | 22 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Depth Anything-Small<br>q8f32<br>(previous version) | 11 |

## YOLO+Depth - 性能 Performance

| OS | Device | Backend | Model | YOLO FPS<br>Camera: h720*w1280 | Depth FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | YOLOv8-n & <br>Depth Anything-Small<br>q8f32<br>(previous version) | 28 | 16.7 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | YOLOv8-n & <br>Depth Anything-Small<br>q8f32<br>(previous version) | 16 | 7.7 |

## Drivable Area - 性能 Performance

| OS | Device | Backend | Model | FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | TwinLiteNet<br>q8f32 | 56 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | TwinLiteNet<br>q8f32 | 28 |

---

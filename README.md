# YOLO-Depth Estimation-for-Android
1. Running YOLO series and monocular depth (distance) estimation on Android devices.
2. Demo models have been uploaded to Google Drive: https://drive.google.com/drive/folders/1MPUvCQCNkjBiHtMjx-eTywetKbkTK7VA?usp=sharing
3. No space left on Baidu Cloud.
4. After downloading, please place the model files in the assets folder.
5. Remember to unzip the *.so compressed files stored in the libs/arm64-v8a folder.
6. Demo models include YOLO: v8, v9 series / Depth Estimation: Depth Anything-small. They are converted from HuggingFace or ModelScope and optimized for peak execution speed.
7. Hence, the input/output of the demo models slightly differs from the original models.
8. Demo models are set to accept 1280x720 image input; hence, they work when the phone is held horizontally.
9. For better compatibility with ONNX Runtime-Android, export does not utilize dynamic-axes. Therefore, the exported ONNX model may not be optimal for x86_64.
10. This project uses OpenGL ES 3.x to directly capture camera frame textures and pass them to the GPU, minimizing image data copying and transfer. It uses GPU compute shaders to convert YUV to RGB and normalize before feeding to the model, minimizing CPU usage.
11. Dual-buffering and asynchronous inference are used to boost FPS. (Using previous camera frame data for inference, not waiting for current image processing)
12. Finally, YOLO bounding boxes are efficiently rendered using the OpenGL ES.
13. Enabling both YOLO and depth estimation simultaneously drops FPS by about 30% (compared to YOLO-only tasks).
14. Estimation accuracy at the focus point is higher due to the influence of the current camera focal length.
15. Based on depth model principles, precision is lower for smooth, luminescent objects, scenes without light and shadow changes, and image edges. For more details, refer to papers on monocular depth estimation.
16. See more about the project: https://dakeqq.github.io/overview/
# 安卓本地运行YOLO与深度(距离)估计
1. 在Android设备上运行YOLO系列与单目深度(距离)估计。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1MPUvCQCNkjBiHtMjx-eTywetKbkTK7VA?usp=sharing
3. 百度云盘没空间了。
4. 下载后，请将模型文件放入assets文件夹。
5. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
6. 演示模型是YOLO: v8, v9 系列 / 深度(距离)估计: Depth Anything-small。 它们是从HuggingFace或ModelScope转换来的，并经过代码优化，以实现极致执行速度。
7. 因此，演示模型的输入输出与原始模型略有不同。
8. 演示模型导出设定为接收1280*720的图象输入，因此"横置手机"才能使用。
9. 为了更好的适配ONNXRuntime-Android，导出时未使用dynamic-axes. 因此导出的ONNX模型对x86_64而言不一定是最优解。
10. 本项目使用OpenGL ES 3.x，直接获取相机帧纹理后传递给GPU，尽可能减少图象数据的复制与传输. 再利用GPU计算着色器将YUV转成RGB并完成归一化后传递给模型, 尽量降低CPU占用。
11. 采用双重缓冲+异步推理來提升FPS。 (推理时使用前一刻相机帧数据，不等待当前的图像处理)
12. 最后使用OpenGL ES来高效的渲染YOLO框线。
13. 同时启用YOLO与距离估计，FPS会下降约30%。(与单YOLO任务时相比)
14. 受到当前相机焦距影响，因此对焦位置的估计精度较高。
15. 根据深度模型原理，光滑物体，发光物体，无光线阴影变化场景，画面边缘等等的精度不高，详细请参阅单目深度估计的相关论文.
16. 看更多項目: https://dakeqq.github.io/overview/
# YOLO - 性能 Performance
| OS | Device | Backend | Model | FPS<br>Camera: 1280*720 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | v9-C<br>q8f32 | 6.5 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | v8-s<br>q8f32 | 21 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | v8-n<br>q8f32 | 43 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v9-C<br>q8f32 | 3.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v8-s<br>q8f32 | 10.5 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | v8-n<br>q8f32 | 23 |

# Depth - 性能 Performance
| OS | Device | Backend | Model | FPS<br>Camera: 1280*720 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | Depth Anything-small<br>q8f32 | 22 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Depth Anything-small<br>q8f32 | 11 |

# YOLO+Depth - 性能 Performance
| OS | Device | Backend | Model | YOLO FPS<br>Camera: 1280*720 | Depth FPS<br>Camera: 1280*720 |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | YOLOv8-n & <br>Depth Anything-small<br>q8f32 | 28 | 16.7 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | YOLOv8-n & <br>Depth Anything-small<br>q8f32 | 16 | 7.7 |
# 演示结果 Demo Results
(YOLOv8-n & Depth Anything-small)<br>
<br>
![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/yolo_depth.gif?raw=true?raw=true)


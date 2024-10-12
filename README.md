# YOLO-Depth-Drivable Estimation-for-Android
1. Running YOLO series, monocular depth (distance) and drivable area estimation on Android devices.
2. Demo models have been uploaded to Google Drive: https://drive.google.com/drive/folders/1MPUvCQCNkjBiHtMjx-eTywetKbkTK7VA?usp=sharing
3. After downloading, please place the model files in the assets folder.
4. Remember to unzip the *.so compressed files stored in the libs/arm64-v8a folder.
5. Demo models include YOLO: v8, v9, v10, v11, NAS series / Depth Estimation: Depth Anything V2-Metric-Small / Drivable Area: TwinLiteNet. They are converted from HuggingFace or ModelScope or Github and optimized for peak execution speed.
6. Hence, the input/output of the demo models slightly differs from the original models.
7. Demo models are set to accept h720*w1280 image input; hence, they work when the phone is held horizontally.
8. For better compatibility with ONNX Runtime-Android, export does not utilize dynamic-axes. Therefore, the exported ONNX model may not be optimal for x86_64.
9. This project uses OpenGL ES 3.x to directly capture camera frame textures and pass them to the GPU, minimizing image data copying and transfer. It uses GPU compute shaders to convert YUV to RGB before feeding to the model, minimizing CPU usage.
10. Asynchronous inference are used. (Using previous camera frame data for inference, not waiting for current image processing)
11. Finally, YOLO bounding boxes are efficiently rendered using the OpenGL ES.
12. If using YOLO-v10 series, replace the original GLRender.java and project.h with the one in the "YOLO_Depth_Drivable/v10" folder.
13. Enabling both YOLO and depth estimation simultaneously drops FPS by about 30% (compared to YOLO-only tasks).
14. Based on depth model principles, precision is lower for smooth, luminescent objects, scenes without light and shadow changes, and image edges. For more details, refer to papers on monocular depth estimation.
15. TwinLiteNet's lane detection accuracy is low, with the cause currently unclear.
16. Drivable area detection is less accurate due to water, reflections, shadows, and low light.
17. For high-resolution screens, use GL_LINE_STRIP to draw the drivable area, or you won't see anything.
18. Updated model export method on July 6, 2024, in the folder Export_ONNX. The Depth model has been updated to DepthAnythingV2-Metric; for more details, please search for Github DepthAnythingV2.
19. Implemented the Depth-Metric model that directly outputs distance, which is more accurate than the previous approximation method. It is normal to observe fluctuations or inaccuracies in detected values.
20. Before exporting, make sure to complete the configuration file *_config.py, ensuring it matches the model weight file.
21. If using self-exported models, remember to modify the corresponding height/width values in GLRenderer.java and project.h files.
22. Since image processing models typically involve convolutional operators, remember to use UInt8 for quantization; otherwise, ONNX Runtime will generate errors.
23. Currently, only the Yolo v9 & NAS series can utilize the Qualcomm NPU (HTP). Other series and Depth models are unsupported, either failing to compile or crashing upon execution. Waiting for updates from Qualcomm and ONNX.
24. The export of v9-E fails and is currently unusable.
25. The demo code states, "When YOLO inference is to be executed, the GPU bypasses processing the current frame's data." To utilize NPU with v9-S,T and NAS effectively (since have high FPS, leading to continuous image processing skips), slight modifications to this logic are necessary.
26. Configuration code for the Qualcomm NPU (HTP) is in the folder YOLO_Depth_Drivable_Qualcomm_NPU.
27. Due to Qualcomm's regulations, you must obtain the NPU libraries on your own.
28. The quantization method for the model can be seen in the folder "Do_Quantize".
29. The q4(uint4) quantization method is not currently recommended because the "MatMulNBits" operator in ONNX Runtime is performing poorly.
30. See more about the project: https://dakeqq.github.io/overview/
# 安卓本地运行YOLO+深度(距离)+可行驶区域估计
1. 在Android设备上运行YOLO系列, 单目深度(距离), 可行驶区域估计。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1MPUvCQCNkjBiHtMjx-eTywetKbkTK7VA?usp=sharing
3. 百度云盘: https://pan.baidu.com/s/1WzRPiV9EL_ijpkgCJaZRTg?pwd=dake 提取码: dake。
4. 下载后，请将模型文件放入assets文件夹。
5. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
6. 演示模型是YOLO: v8, v9, v10, v11, NAS 系列 / 深度(距离)估计: Depth Anything V2-Metric-Small / 可行驶区域: TwinLiteNet。 它们是从HuggingFace或ModelScope或Github转换来的，并经过代码优化，以实现极致执行速度。
7. 因此，演示模型的输入输出与原始模型略有不同。
8. 演示模型导出设定为接收h720*w1280的图象输入，因此"横置手机"才能使用。
9. 为了更好的适配ONNXRuntime-Android，导出时未使用dynamic-axes. 因此导出的ONNX模型对x86_64而言不一定是最优解。
10. 本项目使用OpenGL ES 3.x，直接获取相机帧纹理后传递给GPU，尽可能减少图象数据的复制与传输. 再利用GPU计算着色器将YUV转成RGB后传递给模型, 尽量降低CPU占用。
11. 采用异步推理。 (推理时使用前一刻相机帧数据，不等待当前的图像处理)
12. 最后使用OpenGL ES来高效的渲染YOLO框线。
13. 若使用YOLO-v10系列, 请将"YOLO_Depth_Drivable/v10"文件夹里的GLRender.java & project.h替换原文件。
14. 同时启用YOLO与距离估计，FPS会下降约30%。(与单YOLO任务时相比)
15. 根据深度模型原理，光滑物体，发光物体，无光线阴影变化场景，画面边缘等等的精度不高，详细请参阅单目深度估计的相关论文.
16. TwinLiteNet的车道线估计准确率不高，原因暂时不明。
17. 可行驶区域检测会受到积水或光滑地面的反光影响，阴影交界处, 昏暗地区等等的精确度也较低。
18. 对于高屏幕分辨率的手机，请改使用GL_LINE_STRIP来绘制可行驶区域, 否則你啥也看不見。
19. 2024/07/06更新模型导出方法, 文件夾Export_ONNX。更新Depth模型為DepthAnythingV2-Metric, 详情请搜索Github DepthAnythingV2。
20. 使用了直出距离的Depth-Metric模型，比之前的近似法准一些. 检测数值跳动或不准仍是正常现象。
21. 导出前记得填写配置*_config.py，务必跟模型权重档吻合。
22. 使用自己导出的模型记得修改对应的GLRenfer.java和project.h代码中相关height/width数值。
23. 由于图像处理模型普遍包含卷积算子，因此量化记得使用UInt8, 否则ONNX Runtime会报错。
24. 目前只有Yolo v9 & NAS系列能使用高通NPU（HTP），其他的系列和Depth模型暂时皆不能用，要嘛编译不通过，要嘛编译通过后一跑就崩，坐等高通和ONNX更新。
25. v9-E会导出失败，暂时不能用。
26. Demo代码中写道：“当要执行YOLO推理，GPU略过处理当前帧数据”，因此需要稍微修改此逻辑，才能正常使用NPU + (v9-S,T / NAS)。（因為FPS太高，会一直跳过图像处理）
27. 高通NPU（HTP）的配置代码，在文件夾YOLO_Depth_Drivable_Qualcomm_NPU。
28. 根据高通的规定，您需要自行申請获取NPU库。
29. 模型的量化方法可以在文件夹 "Do_Quantize" 中查看。
30. 现在不建议使用q4(uint4)量化方法, 因为ONNX Runtime的运算符"MatMulNBits"表现不佳。
31. 看更多項目: https://dakeqq.github.io/overview/

# GPU Image Preprocess - 图像预处理性能
| OS | Device | Backend | Pixels: h720*w1280<br>Time Cost: ms | 
|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-GPU<br>Aderno-740 | 4 |
| Harmony 4 | P40 | Kirin_990_5G-GPU<br>Mali-G76 MP16 | 8 |
# YOLO - 性能 Performance
| OS | Device | Backend | Model | FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-x<br>q8f32 | 4 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-l<br>q8f32 | 7 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-m<br>q8f32 | 9 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-s<br>q8f32 | 19 |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | v11-n<br>q8f32 | 38 |
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

# Depth - 性能 Performance
| OS | Device | Backend | Model | FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | Depth Anything-Small<br>q8f32<br>(previous version) | 22 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Depth Anything-Small<br>q8f32<br>(previous version) | 11 |

# YOLO+Depth - 性能 Performance
| OS | Device | Backend | Model | YOLO FPS<br>Camera: h720*w1280 | Depth FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | YOLOv8-n & <br>Depth Anything-Small<br>q8f32<br>(previous version) | 28 | 16.7 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | YOLOv8-n & <br>Depth Anything-Small<br>q8f32<br>(previous version) | 16 | 7.7 |

# Drivable Area - 性能 Performance
| OS | Device | Backend | Model | FPS<br>Camera: h720*w1280 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | TwinLiteNet<br>q8f32 | 56 |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | TwinLiteNet<br>q8f32 | 28 |

# 演示结果 Demo Results
YOLOv8-n & Depth Anything-Small-previous_version)<br>
<br>
![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/yolo_depth.gif?raw=true?raw=true)
<br>
<br>
(YOLOv8-s & Depth Anything-Small-h294_w518-previous_version)<br>
<br>
![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/yolo_depth2.gif?raw=true?raw=true)
<br>
<br>
(TwinLiteNet)<br>
<br>
![Demo Animation](https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android/blob/main/drivable.gif?raw=true?raw=true)




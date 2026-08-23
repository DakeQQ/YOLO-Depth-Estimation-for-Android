#!/usr/bin/env python3
"""Run task-aware ONNX Runtime inference for standalone Android YOLO models."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


# ============================================================================
# USER INFERENCE CONFIGURATION
# Edit this section to select an exported ONNX model and input image.
# ============================================================================
MODEL_PATH = Path(__file__).with_name("models") / "detect" / "yolo26n-det_f32.onnx"
IMAGE_PATH = None                  # None selects a prepared image for the model task.
OUTPUT_PATH = None                 # None writes demo_images/results/<model-stem>_inference.jpg.
PROVIDERS = ("CPUExecutionProvider",)
ORT_LOG = False
ORT_FP16 = False                 # True preserves native FP16 graph transforms.
MAX_THREADS = 0                  # 0 lets ONNX Runtime choose the thread count.
DEVICE_ID = 0
MODEL_TASK = None                  # None reads model_task from ONNX metadata.
MODEL_FAMILY = None                # None reads model_family from ONNX metadata.
RESIZE_SHAPE = None                # None reads resize_shape from ONNX metadata.
END_TO_END = None                  # None reads metadata; old v10/26 detection models use a fallback.
KEYPOINT_SHAPE = None              # None reads keypoint_shape from ONNX metadata.
CLASS_NAMES = None                 # None reads class_names from ONNX metadata.
CONFIDENCE_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
AUTO_OPEN_RESULT = True
OVERWRITE_OUTPUT = True


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEMO_INPUT_DIRECTORY = _SCRIPT_DIR / "demo_images" / "inputs"
_DEMO_RESULT_DIRECTORY = _SCRIPT_DIR / "demo_images" / "results"
_DEFAULT_DEMO_IMAGE_NAMES = {
    "detect": "office.jpg",
    "segment": "street.jpg",
    "pose": "street.jpg",
    "obb": "aerial.jpg",
    "classify": "stadium.jpg",
    "semantic": "street.jpg",
    "depth": "office.jpg",
}
_DETECTION_LIKE_TASKS = frozenset({"detect", "segment", "pose", "obb"})
_SUPPORTED_TASKS = _DETECTION_LIKE_TASKS | {"classify", "semantic", "depth"}
_SUPPORTED_FAMILIES = frozenset({"ultralytics", "yolo_nas"})
_PROVIDER_CHAINS = {
    "cpu": ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "openvino": ("OpenVINOExecutionProvider", "CPUExecutionProvider"),
    "dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
}
_POSE_KEYPOINT_CONFIDENCE_THRESHOLD = 0.25
_COCO_POSE_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
_ORT_TO_NUMPY_DTYPES = {
    "tensor(bool)": "bool",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
    "tensor(uint16)": "uint16",
    "tensor(int16)": "int16",
    "tensor(uint32)": "uint32",
    "tensor(int32)": "int32",
    "tensor(uint64)": "uint64",
    "tensor(int64)": "int64",
    "tensor(float)": "float32",
}
_SUPPORTED_PUBLIC_INPUT_TYPES = frozenset({"tensor(uint8)", "tensor(float)"})


class InferenceConfigurationError(ValueError):
    """Raised when the selected artifact or runtime configuration is invalid."""


class InferenceValidationError(RuntimeError):
    """Raised when ONNX Runtime output violates the exported task contract."""


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    image_path: Path | None
    output_path: Path | None
    providers: tuple[str, ...]
    ort_log: bool
    ort_fp16: bool
    max_threads: int
    device_id: int
    model_task: str | None
    model_family: str | None
    resize_shape: tuple[int, int] | None
    end_to_end: bool | None
    keypoint_shape: tuple[int, int] | None
    class_names: tuple[str, ...] | None
    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int
    auto_open_result: bool
    overwrite_output: bool

    @classmethod
    def from_user_configuration(cls) -> "InferenceConfig":
        return cls(
            model_path=Path(MODEL_PATH),
            image_path=Path(IMAGE_PATH) if IMAGE_PATH is not None else None,
            output_path=Path(OUTPUT_PATH) if OUTPUT_PATH is not None else None,
            providers=tuple(PROVIDERS),
            ort_log=ORT_LOG,
            ort_fp16=ORT_FP16,
            max_threads=MAX_THREADS,
            device_id=DEVICE_ID,
            model_task=MODEL_TASK,
            model_family=MODEL_FAMILY,
            resize_shape=tuple(RESIZE_SHAPE) if RESIZE_SHAPE is not None else None,
            end_to_end=END_TO_END,
            keypoint_shape=tuple(KEYPOINT_SHAPE) if KEYPOINT_SHAPE is not None else None,
            class_names=tuple(CLASS_NAMES) if CLASS_NAMES is not None else None,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            max_detections=MAX_DETECTIONS,
            auto_open_result=AUTO_OPEN_RESULT,
            overwrite_output=OVERWRITE_OUTPUT,
        )


@dataclass(frozen=True)
class ArtifactContract:
    model_family: str
    model_task: str
    yolo_version: int | None
    input_name: str
    input_shape: tuple[int, int, int, int]
    input_dtype: str
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    resize_shape: tuple[int, int]
    end_to_end: bool
    keypoint_shape: tuple[int, int] | None
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class StaticIOBuffers:
    input_array: Any
    output_arrays: tuple[Any, ...]
    resize_array: Any
    input_ortvalue: Any
    output_ortvalues: tuple[Any, ...]
    binding: Any


def _import_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        from onnxruntime.capi import _pybind_state as C
    except ModuleNotFoundError as error:
        raise InferenceConfigurationError(
            "Inference requires numpy, onnxruntime, and opencv-python in the active environment."
        ) from error
    return cv2, np, ort, C


def _metadata_literal(metadata: dict[str, str], key: str) -> Any | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise InferenceValidationError(
            f"ONNX metadata {key!r} is not a valid literal: {value!r}."
        ) from error


def _positive_shape(name: str, values: Any, rank: int) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise InferenceConfigurationError(
            f"{name} must contain {rank} positive integers; received {values!r}."
        ) from error
    if len(shape) != rank or any(value <= 0 for value in shape):
        raise InferenceConfigurationError(
            f"{name} must contain {rank} positive integers; received {values!r}."
        )
    return shape


def _metadata_bool(metadata: dict[str, str], key: str) -> bool | None:
    value = metadata.get(key)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise InferenceValidationError(
        f"ONNX metadata {key!r} must be true or false; received {value!r}."
    )


def _metadata_class_names(metadata: dict[str, str]) -> tuple[str, ...]:
    value = metadata.get("class_names")
    if not value:
        return ()
    try:
        names = json.loads(value)
    except json.JSONDecodeError as error:
        raise InferenceValidationError("ONNX class_names metadata is not valid JSON.") from error
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise InferenceValidationError("ONNX class_names metadata must be a JSON string array.")
    return tuple(names)


def _resolve_contract(session: Any, config: InferenceConfig) -> ArtifactContract:
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise InferenceValidationError(
            f"Expected one public image input, found {[value.name for value in inputs]!r}."
        )
    image_input = inputs[0]
    if image_input.name != "images":
        raise InferenceValidationError(
            f"Expected public input name 'images', found {image_input.name!r}."
        )
    if any(not isinstance(dimension, int) or dimension <= 0 for dimension in image_input.shape):
        raise InferenceValidationError(
            f"Inference requires a fully static input shape, found {image_input.shape!r}."
        )
    input_shape = _positive_shape("ONNX input shape", image_input.shape, 4)
    if input_shape[0] != 1 or input_shape[1] != 3:
        raise InferenceValidationError(
            f"Expected static BCHW input [1, 3, H, W], found {input_shape!r}."
        )
    if image_input.type not in _SUPPORTED_PUBLIC_INPUT_TYPES:
        raise InferenceValidationError(
            f"Unsupported public input dtype {image_input.type!r}; expected uint8 or float32."
        )
    input_dtype = _ORT_TO_NUMPY_DTYPES[image_input.type]

    model_task = config.model_task or metadata.get("model_task")
    model_family = config.model_family or metadata.get("model_family")
    if model_task not in _SUPPORTED_TASKS:
        raise InferenceConfigurationError(
            f"MODEL_TASK must be one of {sorted(_SUPPORTED_TASKS)} or present in ONNX metadata; "
            f"received {model_task!r}."
        )
    if model_family not in _SUPPORTED_FAMILIES:
        raise InferenceConfigurationError(
            f"MODEL_FAMILY must be one of {sorted(_SUPPORTED_FAMILIES)} or present in ONNX metadata; "
            f"received {model_family!r}."
        )

    metadata_resize_shape = _metadata_literal(metadata, "resize_shape")
    resize_shape_value = config.resize_shape or metadata_resize_shape
    if resize_shape_value is None:
        raise InferenceConfigurationError(
            "RESIZE_SHAPE is required when the ONNX artifact has no resize_shape metadata."
        )
    resize_shape = _positive_shape("RESIZE_SHAPE", resize_shape_value, 2)

    yolo_version_value = metadata.get("yolo_version")
    try:
        yolo_version = int(yolo_version_value) if yolo_version_value is not None else None
    except ValueError as error:
        raise InferenceValidationError(
            f"ONNX yolo_version metadata is invalid: {yolo_version_value!r}."
        ) from error

    metadata_end_to_end = _metadata_bool(metadata, "end_to_end")
    fallback_end_to_end = (
        model_family == "ultralytics"
        and model_task in _DETECTION_LIKE_TASKS
        and yolo_version in {10, 26}
    )
    end_to_end = (
        config.end_to_end
        if config.end_to_end is not None
        else metadata_end_to_end
        if metadata_end_to_end is not None
        else fallback_end_to_end
    )

    metadata_keypoint_shape = _metadata_literal(metadata, "keypoint_shape")
    keypoint_shape_value = config.keypoint_shape or metadata_keypoint_shape
    keypoint_shape = (
        _positive_shape("KEYPOINT_SHAPE", keypoint_shape_value, 2)
        if keypoint_shape_value is not None
        else None
    )
    if model_task == "pose" and keypoint_shape is None:
        raise InferenceConfigurationError(
            "KEYPOINT_SHAPE is required for pose artifacts without keypoint_shape metadata."
        )

    class_names = config.class_names or _metadata_class_names(metadata)
    output_names = tuple(output.name for output in outputs)
    output_shapes = tuple(
        _positive_shape(
            f"ONNX output {output.name!r} shape",
            output.shape,
            len(output.shape),
        )
        for output in outputs
    )
    try:
        output_dtypes = tuple(_ORT_TO_NUMPY_DTYPES[output.type] for output in outputs)
    except KeyError as error:
        raise InferenceValidationError(
            f"Unsupported public output dtype {error.args[0]!r}."
        ) from error
    metadata_output_names = tuple(
        name for name in metadata.get("output_names", "").split(",") if name
    )
    if metadata_output_names and metadata_output_names != output_names:
        raise InferenceValidationError(
            f"ONNX metadata output_names={metadata_output_names!r} disagrees with graph outputs "
            f"{output_names!r}."
        )
    expected_primary = "output" if model_family == "yolo_nas" else "output0"
    if not output_names or output_names[0] != expected_primary:
        raise InferenceValidationError(
            f"Expected primary output {expected_primary!r}, found {output_names!r}."
        )
    if model_task == "segment" and len(output_names) < 2:
        raise InferenceValidationError("Segmentation inference requires predictions and prototypes.")
    if model_task != "segment" and len(output_names) != 1:
        raise InferenceValidationError(
            f"Task {model_task!r} expects one output, found {output_names!r}."
        )

    return ArtifactContract(
        model_family=model_family,
        model_task=model_task,
        yolo_version=yolo_version,
        input_name=image_input.name,
        input_shape=input_shape,
        input_dtype=input_dtype,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        resize_shape=resize_shape,
        end_to_end=bool(end_to_end),
        keypoint_shape=keypoint_shape,
        class_names=class_names,
    )


def _resolve_paths(config: InferenceConfig, contract: ArtifactContract) -> tuple[Path, Path]:
    image_path = config.image_path
    if image_path is None:
        image_path = _DEMO_INPUT_DIRECTORY / _DEFAULT_DEMO_IMAGE_NAMES[contract.model_task]
    output_path = config.output_path
    if output_path is None:
        output_path = _DEMO_RESULT_DIRECTORY / f"{config.model_path.stem}_inference.jpg"
    if not image_path.is_file():
        raise InferenceConfigurationError(f"IMAGE_PATH does not exist or is not a file: {image_path}.")
    if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise InferenceConfigurationError("OUTPUT_PATH must end in .jpg, .jpeg, or .png.")
    if output_path.resolve() == image_path.resolve():
        raise InferenceConfigurationError("OUTPUT_PATH must not overwrite IMAGE_PATH.")
    if output_path.exists() and not config.overwrite_output:
        raise InferenceConfigurationError(
            f"OUTPUT_PATH already exists: {output_path}. Set OVERWRITE_OUTPUT=True or choose another path."
        )
    return image_path, output_path


def validate_config(config: InferenceConfig, available_providers: list[str]) -> None:
    if not config.model_path.is_file():
        raise InferenceConfigurationError(
            f"MODEL_PATH does not exist or is not a file: {config.model_path}."
        )
    if config.model_path.suffix.lower() != ".onnx":
        raise InferenceConfigurationError("MODEL_PATH must end in .onnx.")
    if not config.providers:
        raise InferenceConfigurationError("PROVIDERS must contain at least one execution provider.")
    unavailable = [provider for provider in config.providers if provider not in available_providers]
    if unavailable:
        raise InferenceConfigurationError(
            f"Unavailable ONNX Runtime provider(s) {unavailable!r}; available={available_providers!r}."
        )
    for name, value in (
        ("CONFIDENCE_THRESHOLD", config.confidence_threshold),
        ("NMS_IOU_THRESHOLD", config.nms_iou_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise InferenceConfigurationError(f"{name} must be a number in [0, 1].")
    if isinstance(config.max_detections, bool) or not isinstance(config.max_detections, int):
        raise InferenceConfigurationError("MAX_DETECTIONS must be a positive integer.")
    if config.max_detections <= 0:
        raise InferenceConfigurationError("MAX_DETECTIONS must be a positive integer.")
    if isinstance(config.max_threads, bool) or not isinstance(config.max_threads, int):
        raise InferenceConfigurationError("MAX_THREADS must be a non-negative integer.")
    if config.max_threads < 0:
        raise InferenceConfigurationError("MAX_THREADS must be a non-negative integer.")
    if isinstance(config.device_id, bool) or not isinstance(config.device_id, int):
        raise InferenceConfigurationError("DEVICE_ID must be a non-negative integer.")
    if config.device_id < 0:
        raise InferenceConfigurationError("DEVICE_ID must be a non-negative integer.")


def create_session_options(ort: Any, config: InferenceConfig) -> Any:
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 0 if config.ort_log else 4
    session_options.log_verbosity_level = 4
    session_options.inter_op_num_threads = config.max_threads
    session_options.intra_op_num_threads = config.max_threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    disabled_optimizers = (
        "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
        if config.ort_fp16
        else ""
    )
    session_config_entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if config.ort_fp16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": disabled_optimizers,
    }
    for key, value in session_config_entries.items():
        session_options.add_session_config_entry(key, value)
    return session_options


def create_run_options(ort: Any, config: InferenceConfig) -> Any:
    run_options = ort.RunOptions()
    run_options.log_severity_level = 0 if config.ort_log else 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return run_options


def resolve_execution_provider(
    C: Any,
    config: InferenceConfig,
) -> tuple[str, Any, list[dict[str, Any]] | None]:
    primary_provider = config.providers[0]
    options_by_provider: dict[str, dict[str, Any]] = {}
    if primary_provider == "OpenVINOExecutionProvider":
        device_type = "cpu"
        ort_device_type = C.OrtDevice.cpu()
        options_by_provider[primary_provider] = {
            "device_type": "CPU",
            "precision": "ACCURACY",
            "num_of_threads": config.max_threads if config.max_threads != 0 else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }
    elif primary_provider == "CUDAExecutionProvider":
        device_type = "cuda"
        ort_device_type = C.OrtDevice.cuda()
        options_by_provider[primary_provider] = {
            "device_id": config.device_id,
            "gpu_mem_limit": 24 * (1024 ** 3),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "sdpa_kernel": "2",
            "use_tf32": "1",
            "fuse_conv_bias": "1",
            "cudnn_conv_use_max_workspace": "1",
            "cudnn_conv1d_pad_to_nc1d": "0",
            "tunable_op_enable": "0",
            "tunable_op_tuning_enable": "0",
            "tunable_op_max_tuning_duration_ms": 10,
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",
            "prefer_nhwc": "0",
            "enable_skip_layer_norm_strict_mode": "0",
            "use_ep_level_unified_stream": "0",
        }
    elif primary_provider == "DmlExecutionProvider":
        device_type = "dml"
        ort_device_type = C.OrtDevice.dml()
        options_by_provider[primary_provider] = {
            "device_id": config.device_id,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
            "enable_graph_serialization": "false",
        }
    else:
        device_type = "cpu"
        ort_device_type = C.OrtDevice.cpu()

    provider_options = [options_by_provider.get(provider, {}) for provider in config.providers]
    return (
        device_type,
        ort_device_type,
        provider_options if any(provider_options) else None,
    )


def _ov(ort: Any, array: Any) -> Any:
    if not array.flags.c_contiguous:
        raise InferenceValidationError("Static ONNX Runtime buffers must be C-contiguous.")
    return ort.OrtValue.ortvalue_from_numpy(array)


def _create_static_io_buffers(
    np: Any,
    ort: Any,
    session: Any,
    contract: ArtifactContract,
) -> StaticIOBuffers:
    try:
        input_array = np.empty(contract.input_shape, dtype=np.dtype(contract.input_dtype))
        output_arrays = tuple(
            np.empty(shape, dtype=np.dtype(dtype))
            for shape, dtype in zip(contract.output_shapes, contract.output_dtypes)
        )
        _, _, input_height, input_width = contract.input_shape
        resize_array = np.empty((input_height, input_width, 3), dtype=np.uint8)
    except (MemoryError, ValueError) as error:
        raise InferenceConfigurationError(
            "Could not allocate the static ONNX Runtime input/output buffers."
        ) from error

    input_ortvalue = _ov(ort, input_array)
    output_ortvalues = tuple(_ov(ort, array) for array in output_arrays)
    binding = session.io_binding()
    binding.bind_ortvalue_input(contract.input_name, input_ortvalue)
    for name, ortvalue in zip(contract.output_names, output_ortvalues):
        binding.bind_ortvalue_output(name, ortvalue)
    return StaticIOBuffers(
        input_array=input_array,
        output_arrays=output_arrays,
        resize_array=resize_array,
        input_ortvalue=input_ortvalue,
        output_ortvalues=output_ortvalues,
        binding=binding,
    )


def run_iobinding(session: Any, binding: Any, run_options: Any) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _class_aware_nms(
    boxes: Any,
    scores: Any,
    class_indices: Any,
    iou_threshold: float,
    max_detections: int,
) -> Any:
    import numpy as np

    selected: list[int] = []
    for class_index in np.unique(class_indices):
        order = np.flatnonzero(class_indices == class_index)
        order = order[np.argsort(scores[order])[::-1]]
        while order.size:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            intersection_top_left = np.maximum(boxes[current, :2], boxes[remaining, :2])
            intersection_bottom_right = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
            intersection_size = np.maximum(
                intersection_bottom_right - intersection_top_left,
                np.float32(0.0),
            )
            intersection_area = intersection_size[:, 0] * intersection_size[:, 1]
            current_area = (boxes[current, 2] - boxes[current, 0]) * (
                boxes[current, 3] - boxes[current, 1]
            )
            remaining_area = (
                (boxes[remaining, 2] - boxes[remaining, 0])
                * (boxes[remaining, 3] - boxes[remaining, 1])
            )
            union_area = current_area + remaining_area - intersection_area
            iou = np.divide(
                intersection_area,
                union_area,
                out=np.zeros_like(intersection_area),
                where=union_area > 0,
            )
            order = remaining[iou <= iou_threshold]

    selected.sort(key=lambda index: float(scores[index]), reverse=True)
    return np.asarray(selected[:max_detections], dtype=np.int32)


def _postprocess_detections(
    output: Any,
    contract: ArtifactContract,
    config: InferenceConfig,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    detections = np.asarray(output)
    if detections.ndim == 3 and detections.shape[0] == 1:
        detections = detections[0]
    if detections.ndim != 2 or detections.shape[1] < 6:
        raise InferenceValidationError(
            f"Detection output must be [1, detections, 6 + extras], received {detections.shape}."
        )

    valid = np.isfinite(detections).all(axis=1)
    if not valid.all():
        raise InferenceValidationError("Detection output contains NaN or Inf values.")
    valid &= detections[:, 4] >= np.float32(config.confidence_threshold)
    valid &= detections[:, 5] == np.rint(detections[:, 5])
    detections = detections[valid].astype(np.float32, copy=False)
    if not detections.size:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, max(0, detections.shape[-1] - 6)), dtype=np.float32),
        )

    scores = detections[:, 4]
    class_indices = detections[:, 5].astype(np.int32)
    extras = detections[:, 6:]
    if contract.end_to_end and contract.model_task != "obb":
        boxes = detections[:, :4].copy()
    else:
        centers = detections[:, :2]
        half_sizes = detections[:, 2:4] * np.float32(0.5)
        boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)

    resize_height, resize_width = contract.resize_shape
    boxes[:, (0, 2)] = np.clip(
        boxes[:, (0, 2)],
        np.float32(0.0),
        np.float32(resize_width),
    )
    boxes[:, (1, 3)] = np.clip(
        boxes[:, (1, 3)],
        np.float32(0.0),
        np.float32(resize_height),
    )
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    class_indices = class_indices[valid_boxes]
    extras = extras[valid_boxes]

    if contract.end_to_end:
        selected = np.argsort(scores)[::-1][: config.max_detections].astype(
            np.int32,
            copy=False,
        )
    else:
        selected = _class_aware_nms(
            boxes,
            scores,
            class_indices,
            config.nms_iou_threshold,
            config.max_detections,
        )
    return boxes[selected], scores[selected], class_indices[selected], extras[selected]


def _open_result(result_path: Path) -> bool:
    if sys.platform == "win32":
        try:
            os.startfile(result_path)  # type: ignore[attr-defined]
        except OSError as error:
            print(f"[yolo-inference] warning: could not open {result_path}: {error}")
            return False
        return True
    command = ["open" if sys.platform == "darwin" else "xdg-open", str(result_path)]
    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        print(f"[yolo-inference] warning: could not open {result_path}: {error}")
        return False
    return True


def _write_image_atomically(cv2: Any, output_path: Path, image: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(output_path.suffix.lower(), image)
    if not encoded:
        raise InferenceValidationError(f"OpenCV could not encode {output_path}.")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=f".tmp{output_path.suffix.lower()}",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(memoryview(payload))
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _color(class_index: int) -> tuple[int, int, int]:
    return (
        64 + (class_index * 53) % 192,
        64 + (class_index * 97) % 192,
        64 + (class_index * 151) % 192,
    )


def _draw_pose(
    cv2: Any,
    image: Any,
    keypoints: Any,
    x_scale: float,
    y_scale: float,
    color: tuple[int, int, int],
    line_thickness: int,
) -> None:
    keypoint_count, keypoint_dimensions = keypoints.shape
    points: list[tuple[int, int] | None] = []
    for keypoint in keypoints:
        if (
            keypoint_dimensions == 3
            and keypoint[2] < _POSE_KEYPOINT_CONFIDENCE_THRESHOLD
        ):
            points.append(None)
            continue
        points.append(
            (
                min(image.shape[1] - 1, max(0, int(round(float(keypoint[0]) * x_scale)))),
                min(image.shape[0] - 1, max(0, int(round(float(keypoint[1]) * y_scale)))),
            )
        )
    if keypoint_count == 17:
        for first_index, second_index in _COCO_POSE_SKELETON:
            first_point = points[first_index]
            second_point = points[second_index]
            if first_point is not None and second_point is not None:
                cv2.line(image, first_point, second_point, color, line_thickness, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(image, point, line_thickness + 1, color, cv2.FILLED, cv2.LINE_AA)


def _draw_text(cv2: Any, image: Any, lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(image.shape[:2]) / 1200)
    thickness = max(1, round(min(image.shape[:2]) / 500))
    measurements = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(height for _, height in measurements) + 10
    panel_width = min(image.shape[1], max(width for width, _ in measurements) + 20)
    panel_height = min(image.shape[0], line_height * len(lines) + 10)
    cv2.rectangle(image, (0, 0), (panel_width, panel_height), (20, 20, 20), cv2.FILLED)
    for index, line in enumerate(lines):
        baseline = min(panel_height - 5, 10 + (index + 1) * line_height - 5)
        cv2.putText(
            image,
            line,
            (10, baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _overlay_masks(
    cv2: Any,
    np: Any,
    image: Any,
    prototypes: Any,
    coefficients: Any,
    boxes: Any,
    class_indices: Any,
    x_scale: float,
    y_scale: float,
) -> None:
    prototypes = np.asarray(prototypes)
    if prototypes.ndim != 4 or prototypes.shape[0] != 1:
        raise InferenceValidationError(
            f"Segmentation prototypes must be [1, masks, H, W], received {prototypes.shape}."
        )
    mask_channels, mask_height, mask_width = prototypes.shape[1:]
    if coefficients.shape[1] < mask_channels:
        raise InferenceValidationError(
            f"Detections contain {coefficients.shape[1]} mask coefficients, but prototypes require "
            f"{mask_channels}."
        )
    if np.issubdtype(prototypes.dtype, np.floating) and not np.isfinite(prototypes).all():
        raise InferenceValidationError("Segmentation prototypes contain NaN or Inf values.")
    logits = coefficients[:, :mask_channels] @ prototypes[0].reshape(mask_channels, -1)
    masks = (
        np.float32(1.0)
        / (
            np.float32(1.0)
            + np.exp(
                -np.clip(logits, np.float32(-30.0), np.float32(30.0))
            )
        )
    ).reshape(
        -1, mask_height, mask_width
    )
    resized_mask = np.empty(image.shape[:2], dtype=np.float32)
    for mask, box, class_index in zip(masks[::-1], boxes[::-1], class_indices[::-1]):
        cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            dst=resized_mask,
            interpolation=cv2.INTER_LINEAR,
        )
        left = max(0, int(round(float(box[0]) * x_scale)))
        top = max(0, int(round(float(box[1]) * y_scale)))
        right = min(image.shape[1], int(round(float(box[2]) * x_scale)))
        bottom = min(image.shape[0], int(round(float(box[3]) * y_scale)))
        mask_roi = resized_mask[top:bottom, left:right]
        image_roi = image[top:bottom, left:right]
        selected = mask_roi >= np.float32(0.5)
        if selected.any():
            color = np.asarray(_color(int(class_index)), dtype=np.float32)
            image_roi[selected] = (
                image_roi[selected].astype(np.float32) * np.float32(0.55)
                + color * np.float32(0.45)
            ).astype(np.uint8)


def _render_dense(cv2: Any, np: Any, image: Any, output: Any, task: str) -> tuple[Any, dict[str, Any]]:
    dense_output = np.asarray(output)
    if np.issubdtype(dense_output.dtype, np.floating) and not np.isfinite(dense_output).all():
        raise InferenceValidationError(f"{task.title()} output contains NaN or Inf values.")
    if task == "semantic":
        if dense_output.ndim == 4:
            labels = dense_output.argmax(axis=1)[0]
        elif dense_output.ndim == 3 and dense_output.shape[0] == 1:
            labels = dense_output[0]
        elif dense_output.ndim == 2:
            labels = dense_output
        else:
            raise InferenceValidationError(
                f"Semantic output must be [1, classes, H, W] or [1, H, W], received "
                f"{dense_output.shape}."
            )
        labels = cv2.resize(
            np.rint(labels).astype(np.int32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        color_map = np.empty((*labels.shape, 3), dtype=np.uint8)
        color_map[..., 0] = (64 + labels * 53) % 256
        color_map[..., 1] = (64 + labels * 97) % 256
        color_map[..., 2] = (64 + labels * 151) % 256
        cv2.addWeighted(image, 0.4, color_map, 0.6, 0, dst=image)
        return image, {"classes_present": int(np.unique(labels).size)}

    depth = np.squeeze(dense_output).astype(np.float32, copy=False)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise InferenceValidationError(
            f"Depth output must contain one finite 2-D map, received {dense_output.shape}."
        )
    depth = cv2.resize(
        depth,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    lower, upper = np.percentile(depth, (2.0, 98.0))
    lower = np.float32(lower)
    scale = np.float32(max(float(upper - lower), 1e-6))
    normalized = np.clip(
        (depth - lower) / scale * np.float32(255.0),
        np.float32(0.0),
        np.float32(255.0),
    ).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    cv2.addWeighted(image, 0.25, color_map, 0.75, 0, dst=image)
    return image, {
        "minimum_depth": float(depth.min()),
        "maximum_depth": float(depth.max()),
    }


def _prepare_input(
    cv2: Any,
    np: Any,
    image: Any,
    contract: ArtifactContract,
    buffers: StaticIOBuffers,
) -> None:
    _, _, input_height, input_width = contract.input_shape
    if image.shape[:2] == (input_height, input_width):
        resized_bgr = image
    else:
        cv2.resize(
            image,
            (input_width, input_height),
            dst=buffers.resize_array,
            interpolation=cv2.INTER_LINEAR,
        )
        resized_bgr = buffers.resize_array
    np.copyto(
        buffers.input_array[0],
        resized_bgr[..., ::-1].transpose(2, 0, 1),
        casting="unsafe",
    )


def _validate_outputs(np: Any, outputs: tuple[Any, ...], contract: ArtifactContract) -> None:
    if len(outputs) != len(contract.output_names):
        raise InferenceValidationError(
            f"ONNX Runtime returned {len(outputs)} outputs, expected {len(contract.output_names)}."
        )
    for name, output, expected_shape, expected_dtype in zip(
        contract.output_names,
        outputs,
        contract.output_shapes,
        contract.output_dtypes,
    ):
        array = np.asarray(output)
        if array.shape != expected_shape or array.dtype != np.dtype(expected_dtype):
            raise InferenceValidationError(
                f"Output {name!r} changed contract: received shape={array.shape}, "
                f"dtype={array.dtype}; expected shape={expected_shape}, dtype={expected_dtype}."
            )
        if not array.size:
            raise InferenceValidationError(f"Output {name!r} is empty.")


def run_inference(config: InferenceConfig) -> dict[str, Any]:
    cv2, np, ort, C = _import_runtime_dependencies()
    validate_config(config, ort.get_available_providers())
    session_options = create_session_options(ort, config)
    run_options = create_run_options(ort, config)
    device_type, _, provider_options = resolve_execution_provider(C, config)
    disabled_optimizers = (
        ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
        if config.ort_fp16
        else None
    )
    try:
        session = ort.InferenceSession(
            str(config.model_path.resolve()),
            sess_options=session_options,
            providers=list(config.providers),
            provider_options=provider_options,
            disabled_optimizers=disabled_optimizers,
        )
    except Exception as error:
        raise InferenceValidationError(
            f"ONNX Runtime could not load {config.model_path}: {error}"
        ) from error

    contract = _resolve_contract(session, config)
    image_path, output_path = _resolve_paths(config, contract)
    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise InferenceValidationError(f"OpenCV could not read {image_path}.")
    buffers = _create_static_io_buffers(np, ort, session, contract)
    _prepare_input(cv2, np, source_image, contract, buffers)
    try:
        run_iobinding(session, buffers.binding, run_options)
    except Exception as error:
        raise InferenceValidationError(
            f"ONNX Runtime inference failed for {config.model_path}: {error}"
        ) from error
    outputs = buffers.output_arrays
    _validate_outputs(np, outputs, contract)

    original_height, original_width = source_image.shape[:2]
    details: dict[str, Any]
    if contract.model_task in _DETECTION_LIKE_TASKS:
        boxes, scores, class_indices, extras = _postprocess_detections(
            outputs[0], contract, config
        )
        rendered = source_image
        resize_height, resize_width = contract.resize_shape
        x_scale = original_width / resize_width
        y_scale = original_height / resize_height
        if contract.model_task == "segment":
            _overlay_masks(
                cv2,
                np,
                rendered,
                outputs[1],
                extras,
                boxes,
                class_indices,
                x_scale,
                y_scale,
            )
        line_thickness = max(2, round(min(original_width, original_height) / 300))
        font_scale = max(0.5, min(original_width, original_height) / 1200)
        for detection_index, (box, score, class_index_value) in enumerate(
            zip(boxes, scores, class_indices)
        ):
            class_index = int(class_index_value)
            left = min(original_width - 1, max(0, int(round(float(box[0]) * x_scale))))
            top = min(original_height - 1, max(0, int(round(float(box[1]) * y_scale))))
            right = min(original_width - 1, max(0, int(round(float(box[2]) * x_scale))))
            bottom = min(original_height - 1, max(0, int(round(float(box[3]) * y_scale))))
            color = _color(class_index)
            if contract.model_task == "obb" and extras.shape[1]:
                center = ((left + right) / 2.0, (top + bottom) / 2.0)
                size = (max(1.0, float(right - left)), max(1.0, float(bottom - top)))
                angle_degrees = float(extras[detection_index, 0]) * 180.0 / np.pi
                points = np.rint(cv2.boxPoints((center, size, angle_degrees))).astype(np.int32)
                cv2.polylines(rendered, [points], True, color, line_thickness, cv2.LINE_AA)
            else:
                cv2.rectangle(rendered, (left, top), (right, bottom), color, line_thickness)
            class_name = (
                contract.class_names[class_index]
                if 0 <= class_index < len(contract.class_names)
                else f"class {class_index}"
            )
            label = f"{class_name} {float(score):.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                line_thickness,
            )
            label_top = max(0, top - label_height - baseline - 6)
            cv2.rectangle(
                rendered,
                (left, label_top),
                (min(original_width - 1, left + label_width + 8), top),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                rendered,
                label,
                (left + 4, max(label_height + 2, top - baseline - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
                cv2.LINE_AA,
            )
            if contract.model_task == "pose" and contract.keypoint_shape is not None:
                keypoint_count, keypoint_dimensions = contract.keypoint_shape
                needed_values = keypoint_count * keypoint_dimensions
                if extras.shape[1] < needed_values:
                    raise InferenceValidationError(
                        f"Pose output has {extras.shape[1]} extra values, expected {needed_values}."
                    )
                keypoints = extras[detection_index, :needed_values].reshape(
                    keypoint_count,
                    keypoint_dimensions,
                )
                _draw_pose(
                    cv2,
                    rendered,
                    keypoints,
                    x_scale,
                    y_scale,
                    _color(detection_index + 1),
                    line_thickness,
                )
        details = {"detections": len(boxes)}
    elif contract.model_task == "classify":
        probabilities = np.asarray(outputs[0])
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise InferenceValidationError(
                f"Classification output must be [1, classes], received {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise InferenceValidationError("Classification output contains NaN or Inf values.")
        top_indices = np.argsort(probabilities[0])[::-1][: min(5, probabilities.shape[1])]
        rendered = source_image
        top_classes = [
            {
                "index": int(index),
                "name": (
                    contract.class_names[index]
                    if index < len(contract.class_names)
                    else f"class {index}"
                ),
                "confidence": float(probabilities[0, index]),
            }
            for index in top_indices
        ]
        _draw_text(
            cv2,
            rendered,
            [f"{entry['name']} {entry['confidence']:.3f}" for entry in top_classes],
        )
        details = {"top_classes": top_classes}
    else:
        rendered, details = _render_dense(
            cv2,
            np,
            source_image,
            outputs[0],
            contract.model_task,
        )

    _write_image_atomically(cv2, output_path.resolve(), rendered)
    result_opened = config.auto_open_result and _open_result(output_path.resolve())
    report = {
        "model": str(config.model_path.resolve()),
        "task": contract.model_task,
        "family": contract.model_family,
        "providers": session.get_providers(),
        "device": device_type,
        "io_binding": {
            "bound_once": True,
            "input_device": "cpu",
            "output_device": "cpu",
            "preallocated": True,
        },
        "input": {
            "name": contract.input_name,
            "dtype": contract.input_dtype,
            "shape": list(contract.input_shape),
        },
        "outputs": [
            {
                "name": name,
                "shape": list(np.asarray(output).shape),
                "dtype": str(np.asarray(output).dtype),
            }
            for name, output in zip(contract.output_names, outputs)
        ],
        "source_image": str(image_path.resolve()),
        "rendered_image": str(output_path.resolve()),
        "result_opened": result_opened,
        **details,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_self_test() -> None:
    cv2, np, _, _ = _import_runtime_dependencies()
    config = InferenceConfig(
        model_path=Path("unused.onnx"),
        image_path=None,
        output_path=None,
        providers=("CPUExecutionProvider",),
        ort_log=False,
        ort_fp16=False,
        max_threads=0,
        device_id=0,
        model_task=None,
        model_family=None,
        resize_shape=None,
        end_to_end=None,
        keypoint_shape=None,
        class_names=None,
        confidence_threshold=0.3,
        nms_iou_threshold=0.45,
        max_detections=100,
        auto_open_result=False,
        overwrite_output=True,
    )
    regular_contract = ArtifactContract(
        model_family="ultralytics",
        model_task="detect",
        yolo_version=12,
        input_name="images",
        input_shape=(1, 3, 32, 32),
        input_dtype="uint8",
        output_names=("output0",),
        output_shapes=((1, 3, 6),),
        output_dtypes=("float32",),
        resize_shape=(32, 32),
        end_to_end=False,
        keypoint_shape=None,
        class_names=(),
    )
    output = np.array(
        [[
            [10.0, 10.0, 8.0, 8.0, 0.9, 1.0],
            [10.5, 10.5, 8.0, 8.0, 0.8, 1.0],
            [25.0, 25.0, 4.0, 4.0, 0.7, 2.0],
        ]],
        dtype=np.float32,
    )
    boxes, scores, class_indices, extras = _postprocess_detections(
        output,
        regular_contract,
        config,
    )
    assert boxes.shape == (2, 4)
    assert np.allclose(boxes[0], [6.0, 6.0, 14.0, 14.0])
    assert np.allclose(scores, [0.9, 0.7])
    assert class_indices.tolist() == [1, 2]
    assert class_indices.dtype == np.int32
    assert extras.shape == (2, 0)

    end_to_end_contract = replace(regular_contract, yolo_version=26, end_to_end=True)
    end_to_end_output = np.array(
        [[[2.0, 4.0, 20.0, 24.0, 0.9, 3.0], [3.0, 5.0, 21.0, 25.0, 0.8, 3.0]]],
        dtype=np.float32,
    )
    end_to_end_boxes, _, _, _ = _postprocess_detections(
        end_to_end_output,
        end_to_end_contract,
        config,
    )
    assert np.allclose(end_to_end_boxes, end_to_end_output[0, :, :4])

    input_array = np.empty((1, 3, 2, 2), dtype=np.uint8)
    buffers = StaticIOBuffers(
        input_array=input_array,
        output_arrays=(),
        resize_array=np.empty((2, 2, 3), dtype=np.uint8),
        input_ortvalue=None,
        output_ortvalues=(),
        binding=None,
    )
    source_bgr = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    input_contract = replace(
        regular_contract,
        input_shape=(1, 3, 2, 2),
        resize_shape=(2, 2),
    )
    _prepare_input(cv2, np, source_bgr, input_contract, buffers)
    expected_rgb_chw = source_bgr[..., ::-1].transpose(2, 0, 1)
    assert np.array_equal(input_array[0], expected_rgb_chw)
    print(
        "Self-test passed: detection postprocessing and fused static-buffer preprocessing "
        "are operational."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic postprocessing checks.")
    parser.add_argument("--model", type=Path, help="Override MODEL_PATH.")
    parser.add_argument("--task", choices=sorted(_SUPPORTED_TASKS), help="Override MODEL_TASK.")
    parser.add_argument("--image", type=Path, help="Override IMAGE_PATH.")
    parser.add_argument("--output", type=Path, help="Override OUTPUT_PATH.")
    parser.add_argument(
        "--provider",
        choices=tuple(_PROVIDER_CHAINS),
        help="Override PROVIDERS with a configured execution-provider chain.",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open the rendered result.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    config = InferenceConfig.from_user_configuration()
    if args.model is not None:
        config = replace(config, model_path=args.model)
    if args.task is not None:
        config = replace(config, model_task=args.task)
    if args.image is not None:
        config = replace(config, image_path=args.image)
    if args.output is not None:
        config = replace(config, output_path=args.output)
    if args.provider is not None:
        config = replace(config, providers=_PROVIDER_CHAINS[args.provider])
    if args.no_open:
        config = replace(config, auto_open_result=False)
    run_inference(config)


if __name__ == "__main__":
    main()

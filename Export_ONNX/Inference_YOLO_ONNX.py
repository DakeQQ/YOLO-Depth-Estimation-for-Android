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
_POSE_KEYPOINT_CONFIDENCE_THRESHOLD = 0.25
_COCO_POSE_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
_ORT_TO_NUMPY_DTYPES = {
    "tensor(uint8)": "uint8",
    "tensor(float)": "float32",
}


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
    resize_shape: tuple[int, int]
    end_to_end: bool
    keypoint_shape: tuple[int, int] | None
    class_names: tuple[str, ...]


def _import_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise InferenceConfigurationError(
            "Inference requires numpy, onnxruntime, and opencv-python in the active environment."
        ) from error
    return cv2, np, ort


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
    try:
        input_dtype = _ORT_TO_NUMPY_DTYPES[image_input.type]
    except KeyError as error:
        raise InferenceValidationError(
            f"Unsupported public input dtype {image_input.type!r}; expected uint8 or float32."
        ) from error

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
            intersection_size = np.maximum(intersection_bottom_right - intersection_top_left, 0.0)
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
    return np.asarray(selected[:max_detections], dtype=np.int64)


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
    valid &= detections[:, 4] >= np.float32(config.confidence_threshold)
    valid &= detections[:, 5] == np.rint(detections[:, 5])
    detections = detections[valid].astype(np.float32, copy=False)
    if not detections.size:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, max(0, detections.shape[-1] - 6)), dtype=np.float32),
        )

    scores = detections[:, 4]
    class_indices = detections[:, 5].astype(np.int64)
    extras = detections[:, 6:]
    if contract.end_to_end and contract.model_task != "obb":
        boxes = detections[:, :4].copy()
    else:
        centers = detections[:, :2]
        half_sizes = detections[:, 2:4] * np.float32(0.5)
        boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)

    resize_height, resize_width = contract.resize_shape
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, float(resize_width))
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, float(resize_height))
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    class_indices = class_indices[valid_boxes]
    extras = extras[valid_boxes]

    if contract.end_to_end:
        selected = np.argsort(scores)[::-1][: config.max_detections]
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
            temporary_file.write(payload.tobytes())
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
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
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
    logits = coefficients[:, :mask_channels] @ prototypes[0].reshape(mask_channels, -1)
    masks = (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).reshape(
        -1, mask_height, mask_width
    )
    for mask, box, class_index in zip(masks[::-1], boxes[::-1], class_indices[::-1]):
        resized_mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        left = max(0, int(round(float(box[0]) * x_scale)))
        top = max(0, int(round(float(box[1]) * y_scale)))
        right = min(image.shape[1], int(round(float(box[2]) * x_scale)))
        bottom = min(image.shape[0], int(round(float(box[3]) * y_scale)))
        inside_box = np.zeros(resized_mask.shape, dtype=bool)
        inside_box[top:bottom, left:right] = True
        selected = (resized_mask >= 0.5) & inside_box
        if selected.any():
            color = np.asarray(_color(int(class_index)), dtype=np.float32)
            image[selected] = (
                image[selected].astype(np.float32) * 0.55 + color * 0.45
            ).astype(np.uint8)


def _render_dense(cv2: Any, np: Any, image: Any, output: Any, task: str) -> tuple[Any, dict[str, Any]]:
    dense_output = np.asarray(output)
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
        rendered = cv2.addWeighted(image, 0.4, color_map, 0.6, 0)
        return rendered, {"classes_present": int(np.unique(labels).size)}

    depth = np.squeeze(dense_output)
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
    scale = max(float(upper - lower), 1e-6)
    normalized = np.clip((depth - lower) / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    rendered = cv2.addWeighted(image, 0.25, color_map, 0.75, 0)
    return rendered, {
        "minimum_depth": float(depth.min()),
        "maximum_depth": float(depth.max()),
    }


def _prepare_input(cv2: Any, np: Any, image: Any, contract: ArtifactContract) -> Any:
    _, _, input_height, input_width = contract.input_shape
    resized_bgr = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(resized_rgb.transpose(2, 0, 1)[None])
    return tensor if contract.input_dtype == "uint8" else tensor.astype(np.float32)


def _validate_outputs(np: Any, outputs: list[Any], contract: ArtifactContract) -> None:
    if len(outputs) != len(contract.output_names):
        raise InferenceValidationError(
            f"ONNX Runtime returned {len(outputs)} outputs, expected {len(contract.output_names)}."
        )
    for name, output in zip(contract.output_names, outputs):
        array = np.asarray(output)
        if not array.size:
            raise InferenceValidationError(f"Output {name!r} is empty.")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise InferenceValidationError(f"Output {name!r} contains NaN or Inf values.")


def run_inference(config: InferenceConfig) -> dict[str, Any]:
    cv2, np, ort = _import_runtime_dependencies()
    validate_config(config, ort.get_available_providers())
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            str(config.model_path.resolve()),
            sess_options=session_options,
            providers=list(config.providers),
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
    input_tensor = _prepare_input(cv2, np, source_image, contract)
    try:
        outputs = session.run(None, {contract.input_name: input_tensor})
    except Exception as error:
        raise InferenceValidationError(
            f"ONNX Runtime inference failed for {config.model_path}: {error}"
        ) from error
    _validate_outputs(np, outputs, contract)

    original_height, original_width = source_image.shape[:2]
    details: dict[str, Any]
    if contract.model_task in _DETECTION_LIKE_TASKS:
        boxes, scores, class_indices, extras = _postprocess_detections(
            outputs[0], contract, config
        )
        rendered = source_image.copy()
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
        top_indices = np.argsort(probabilities[0])[::-1][: min(5, probabilities.shape[1])]
        rendered = source_image.copy()
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
    _, np, _ = _import_runtime_dependencies()
    config = InferenceConfig(
        model_path=Path("unused.onnx"),
        image_path=None,
        output_path=None,
        providers=("CPUExecutionProvider",),
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
    print("Self-test passed: regular detection NMS and end-to-end box handling are operational.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic postprocessing checks.")
    parser.add_argument("--model", type=Path, help="Override MODEL_PATH.")
    parser.add_argument("--task", choices=sorted(_SUPPORTED_TASKS), help="Override MODEL_TASK.")
    parser.add_argument("--image", type=Path, help="Override IMAGE_PATH.")
    parser.add_argument("--output", type=Path, help="Override OUTPUT_PATH.")
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
    if args.no_open:
        config = replace(config, auto_open_result=False)
    run_inference(config)


if __name__ == "__main__":
    main()

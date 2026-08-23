#!/usr/bin/env python3
"""Standalone static ONNX exporter for the Android YOLO models.

This file owns the project-specific graph behavior and does not patch or import
the legacy ``modeling_modified`` directories.

Portions of the Ultralytics graph behavior are adapted from Ultralytics,
licensed under AGPL-3.0: https://github.com/ultralytics/ultralytics
Portions of the YOLO-NAS graph behavior are adapted from Deci-AI SuperGradients
3.7.1, licensed under Apache-2.0: https://github.com/Deci-AI/super-gradients
"""

from __future__ import annotations

from collections import Counter
import copy
import importlib
import importlib.metadata
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

sys.dont_write_bytecode = True


# ============================================================================
# USER EXPORT CONFIGURATION
# Edit this section to select the model and the static Android ONNX contract.
# ============================================================================
# Model selection
# Supported model/task matrix ("auto" resolves the task from the checkpoint):
#
# Series/family        detect  segment  pose  obb  classify  semantic  depth
# Ultralytics YOLOv8      ✓        ✓      ✓    ✓       ✓        -        -
# Ultralytics YOLOv9      ✓        ✓      ✓    ✓       ✓        -        -
# Ultralytics YOLOv10     ✓        ✓      ✓    ✓       ✓        -        -
# Ultralytics YOLOv11     ✓        ✓      ✓    ✓       ✓        -        -
# Ultralytics YOLOv12     ✓        ✓      ✓    ✓       ✓        -        -
# Ultralytics YOLO26      ✓        ✓      ✓    ✓       ✓        ✓        ✓
# YOLO-NAS S/M/L          ✓        -      -    -       -        -        -

MODEL_FAMILY = "ultralytics"
MODEL_PATH = "yolo26n.pt"
MODEL_TASK = "detect"
AUTO_DOWNLOAD_MODEL = True
MODEL_DOWNLOAD_RETRIES = 0
MODEL_DIRECTORY = Path(__file__).with_name("models")
YOLO_VERSION = None
YOLO_NAS_VARIANT = "m"
YOLO_NAS_NUM_CLASSES = 80

# Static Android ONNX contract
INPUT_SHAPE = (1, 3, 720, 1280)
RESIZE_SHAPE = (360, 640)
PUBLIC_INPUT_DTYPE = "uint8"
MODEL_PRECISION = "float32"
STATIC_SHAPES = True

# Export execution
OUTPUT_PATH = None
OPSET = 20
DEVICE = "cpu"
OVERWRITE_OUTPUT = True
RUN_INFERENCE_DEMO = True


ULTRALYTICS_TASK_SUFFIXES = {
    "detect": "",
    "segment": "-seg",
    "pose": "-pose",
    "obb": "-obb",
    "classify": "-cls",
    "semantic": "-sem",
    "depth": "-depth",
}
DETECTION_LIKE_TASKS = frozenset({"detect", "segment", "pose", "obb"})
ULTRALYTICS_MODEL_VERSION_PATTERN = re.compile(
    r"yolo(?:v)?(?P<version>[0-9]+)", re.IGNORECASE
)
ULTRALYTICS_OFFICIAL_MODEL_PATTERN = re.compile(
    r"^yolo(?:v)?(?:8|9|10|11|12|26)[a-z0-9]*(?:-(?:seg|pose|obb|cls|sem|depth))?\.pt$",
    re.IGNORECASE,
)
ULTRALYTICS_TASK_SUFFIX_PATTERN = re.compile(
    r"-(?:seg|pose|obb|cls|sem|depth)$",
    re.IGNORECASE,
)
LEGACY_OUTPUT_NAMES = {"ultralytics": "output0", "yolo_nas": "output"}


def _resolve_model_checkpoint_path(
    configured_model_path: str | Path,
    model_family: str,
    model_task: str,
) -> Path:
    model_path = Path(configured_model_path)
    if (
        model_family != "ultralytics"
        or model_path.parent != Path(".")
        or ULTRALYTICS_OFFICIAL_MODEL_PATTERN.fullmatch(model_path.name) is None
    ):
        return model_path

    if model_task == "auto":
        matching_tasks = [
            task
            for task, suffix in ULTRALYTICS_TASK_SUFFIXES.items()
            if suffix and model_path.stem.lower().endswith(suffix)
        ]
        resolved_task = matching_tasks[0] if matching_tasks else "detect"
    else:
        resolved_task = model_task

    base_stem = ULTRALYTICS_TASK_SUFFIX_PATTERN.sub("", model_path.stem)
    checkpoint_name = f"{base_stem}{ULTRALYTICS_TASK_SUFFIXES[resolved_task]}.pt"
    return Path(MODEL_DIRECTORY) / resolved_task / checkpoint_name


def _resolve_output_path(
    model_path: Path,
    configured_output_path: str | Path | None,
) -> Path:
    if configured_output_path is not None:
        return Path(configured_output_path)
    return model_path.with_name(f"{model_path.stem}.onnx")


def _infer_ultralytics_version(model_path: Path) -> int:
    versions = {
        int(match.group("version"))
        for match in ULTRALYTICS_MODEL_VERSION_PATTERN.finditer(model_path.name)
    }
    return next(iter(versions))


def _module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _package_version(package_name: str) -> str:
    return importlib.metadata.version(package_name)


@dataclass(frozen=True)
class ExportConfig:
    model_family: Literal["ultralytics", "yolo_nas"]
    model_path: Path
    model_task: str
    yolo_version: int
    yolo_nas_variant: str
    yolo_nas_num_classes: int
    input_shape: tuple[int, int, int, int]
    resize_shape: tuple[int, int]
    output_path: Path
    opset_version: int
    static_shapes: bool
    device: str
    model_precision: str
    public_input_dtype: str
    overwrite_output: bool

    @classmethod
    def from_user_configuration(cls) -> "ExportConfig":
        model_path = _resolve_model_checkpoint_path(MODEL_PATH, MODEL_FAMILY, MODEL_TASK)
        output_path = _resolve_output_path(model_path, OUTPUT_PATH)
        yolo_version = 0
        if MODEL_FAMILY == "ultralytics":
            yolo_version = (
                YOLO_VERSION
                if YOLO_VERSION is not None
                else _infer_ultralytics_version(model_path)
            )
        model_task = (
            "detect"
            if MODEL_FAMILY == "yolo_nas" and MODEL_TASK == "auto"
            else MODEL_TASK
        )
        return cls(
            model_family=MODEL_FAMILY,
            model_path=model_path,
            model_task=model_task,
            yolo_version=yolo_version,
            yolo_nas_variant=YOLO_NAS_VARIANT,
            yolo_nas_num_classes=YOLO_NAS_NUM_CLASSES,
            input_shape=tuple(INPUT_SHAPE),
            resize_shape=tuple(RESIZE_SHAPE),
            output_path=output_path,
            opset_version=OPSET,
            static_shapes=STATIC_SHAPES,
            device=DEVICE,
            model_precision=MODEL_PRECISION,
            public_input_dtype=PUBLIC_INPUT_DTYPE,
            overwrite_output=OVERWRITE_OUTPUT,
        )


def _static_resize_scale(config: ExportConfig) -> tuple[float, float]:
    input_height, input_width = config.input_shape[2:]
    resize_height, resize_width = config.resize_shape
    return resize_height / input_height, resize_width / input_width


def _stride_aligned_padding(
    resize_shape: tuple[int, int],
    strides: Any,
) -> tuple[int, int, int, int]:
    stride_values = tuple(int(round(float(stride))) for stride in strides)
    maximum_stride = max(stride_values, default=1)
    resize_height, resize_width = resize_shape
    padded_height = math.ceil(resize_height / maximum_stride) * maximum_stride
    padded_width = math.ceil(resize_width / maximum_stride) * maximum_stride
    return 0, padded_width - resize_width, 0, padded_height - resize_height


def _ensure_model_checkpoint(
    config: ExportConfig,
    *,
    auto_download: bool = AUTO_DOWNLOAD_MODEL,
) -> None:
    if config.model_path.is_file() or not auto_download:
        return
    if (
        config.model_family != "ultralytics"
        or ULTRALYTICS_OFFICIAL_MODEL_PATTERN.fullmatch(config.model_path.name) is None
    ):
        return

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[model-download] task={config.model_task} "
        f"checkpoint={config.model_path.name} destination={config.model_path}"
    )
    downloads = _module("ultralytics.utils.downloads")
    try:
        downloads.attempt_download_asset(
            config.model_path,
            retry=MODEL_DOWNLOAD_RETRIES,
        )
    except ConnectionError as error:
        raise ConnectionError(
            f"Could not download {config.model_path.name}. Place the checkpoint at "
            f"'{config.model_path}' or set MODEL_PATH to an existing local file."
        ) from error
    if not config.model_path.is_file():
        raise FileNotFoundError(
            f"The model download did not create '{config.model_path}'. Place the "
            "checkpoint there or set MODEL_PATH to an existing local file."
        )


def _import_torch() -> Any:
    return _module("torch")


def _extract_model_class_names(*models: Any) -> tuple[str, ...]:
    for model in models:
        for attribute in ("names", "class_names", "_class_names"):
            names = getattr(model, attribute, None)
            if isinstance(names, dict) and names:
                ordered_names = sorted((int(index), str(name)) for index, name in names.items())
                if [index for index, _ in ordered_names] == list(range(len(ordered_names))):
                    return tuple(name for _, name in ordered_names)
            if isinstance(names, (list, tuple)) and names:
                return tuple(str(name) for name in names)
    return ()


def _configure_ultralytics_head(model: Any) -> Any:
    layers = model.model
    head = layers[-1]
    head.export = True
    if hasattr(head, "dynamic"):
        head.dynamic = False
    if hasattr(head, "format"):
        head.format = "onnx"
    return layers


def _fold_input_normalization_into_first_conv(
    model: Any,
    config: ExportConfig,
    torch: Any,
) -> bool:
    if config.model_precision != "float32":
        return False
    if getattr(model, "_standalone_input_normalization_folded", False):
        return True
    with torch.no_grad():
        model.model[0].conv.weight.mul_(1.0 / 255.0)
    model._standalone_input_normalization_folded = True
    return True


def _specialize_static_channel_splits(model: Any) -> tuple[str, ...]:
    specialized: list[str] = []
    for name, module in model.named_modules():
        forward_split = getattr(module, "forward_split", None)
        if not callable(forward_split):
            continue
        split_function = getattr(type(module), "forward_split", None)
        if getattr(module.forward, "__func__", None) is split_function:
            continue
        module.forward = forward_split
        specialized.append(name)
    return tuple(specialized)


def _static_area_attention_forward(module: Any, x: Any) -> Any:
    batch, _, height, width = module._standalone_input_shape
    positions = height * width
    qkv = module.qkv(x).reshape(
        batch,
        module.all_head_dim * 3,
        positions,
    ).transpose(1, 2)
    working_batch = batch
    working_positions = positions
    if module.area > 1:
        qkv = qkv.reshape(
            batch * module.area,
            positions // module.area,
            module.all_head_dim * 3,
        )
        working_batch = batch * module.area
        working_positions = positions // module.area
    q, k, v = (
        qkv.view(
            working_batch,
            working_positions,
            module.num_heads,
            module.head_dim * 3,
        )
        .permute(0, 2, 3, 1)
        .split([module.head_dim, module.head_dim, module.head_dim], dim=2)
    )
    attention = (q * (module.head_dim**-0.5)).transpose(-2, -1) @ k
    attention = attention.softmax(dim=-1)
    x = v @ attention.transpose(-2, -1)
    if module.area > 1:
        x = (
            x.reshape(batch, module.area, module.all_head_dim, working_positions)
            .permute(0, 2, 1, 3)
            .reshape(batch, module.all_head_dim, height, width)
        )
        v = (
            v.reshape(batch, module.area, module.all_head_dim, working_positions)
            .permute(0, 2, 1, 3)
            .reshape(batch, module.all_head_dim, height, width)
        )
    else:
        x = x.reshape(batch, module.all_head_dim, height, width)
        v = v.reshape(batch, module.all_head_dim, height, width)
    return module.proj(x + module.pe(v))


def _specialize_static_attention(
    wrapper: Any,
    images: Any,
    config: ExportConfig,
    torch: Any,
) -> None:
    candidates = [
        (name, module)
        for name, module in wrapper.named_modules()
        if type(module).__module__ == "ultralytics.nn.modules.block"
        and type(module).__name__ == "AAttn"
    ]
    head = wrapper.layers[-1]
    specialize_decode = (
        type(head).__module__ == "ultralytics.nn.modules.head"
        and callable(getattr(head, "decode_bboxes", None))
        and hasattr(head, "end2end")
        and hasattr(head, "xyxy")
        and getattr(wrapper, "model_task", "detect") != "obb"
    )
    specialize_obb_postprocess = (
        type(head).__module__ == "ultralytics.nn.modules.head"
        and type(head).__name__ in {"OBB", "OBB26"}
        and bool(getattr(head, "end2end", False))
        and bool(getattr(head, "export", False))
        and getattr(head, "format", None) == "onnx"
        and not bool(getattr(head, "agnostic_nms", False))
        and config.static_shapes
        and config.input_shape[0] == 1
    )
    if not candidates and not specialize_decode and not specialize_obb_postprocess:
        wrapper.static_export_paths_prepared = True
        return

    captured_shapes: dict[str, tuple[int, ...]] = {}
    handles = [
        module.register_forward_pre_hook(
            lambda current, arguments, module_name=name: captured_shapes.__setitem__(
                module_name,
                tuple(arguments[0].shape),
            )
        )
        for name, module in candidates
    ]
    try:
        wrapper(images)
    finally:
        for handle in handles:
            handle.remove()

    for name, module in candidates:
        module._standalone_input_shape = captured_shapes[name]
        module.forward = types.MethodType(_static_area_attention_forward, module)

    if specialize_decode:
        def static_decode_bboxes(
            current: Any,
            bboxes: Any,
            anchors: Any,
            xywh: bool = True,
        ) -> Any:
            left_top, right_bottom = bboxes.split((2, 2), dim=1)
            top_left = anchors - left_top
            bottom_right = anchors + right_bottom
            if xywh and not current.end2end and not current.xyxy:
                center = (top_left + bottom_right) / 2
                size = bottom_right - top_left
                return torch.cat((center, size), dim=1)
            return torch.cat((top_left, bottom_right), dim=1)

        head.decode_bboxes = types.MethodType(static_decode_bboxes, head)

    if specialize_obb_postprocess:
        def static_end_to_end_obb_forward(
            current: Any,
            features: list[Any],
        ) -> Any:
            predictions = current.forward_head(features, **current.one2one)
            logits = predictions["scores"]

            candidate_count = min(current.max_det, logits.shape[2])
            anchor_scores = logits.max(dim=1)[0].sigmoid()
            anchor_indices = current._grouped_topk(
                anchor_scores,
                candidate_count,
                1,
            )[1]
            candidate_scores = logits.index_select(
                2,
                anchor_indices[0],
            ).sigmoid().transpose(1, 2)
            selected_scores, flat_indices = current._grouped_topk(
                candidate_scores.flatten(1),
                candidate_count,
                1,
            )
            candidate_indices = flat_indices // current.nc
            selected_anchor_indices = anchor_indices.index_select(
                1,
                candidate_indices[0],
            )[0]

            angles = predictions["angle"].index_select(
                2,
                selected_anchor_indices,
            )
            current.angle = angles
            boxes = current.decode_bboxes(
                current.dfl(
                    predictions["boxes"].index_select(
                        2,
                        selected_anchor_indices,
                    )
                ),
                current.anchors.index_select(
                    1,
                    selected_anchor_indices,
                ).unsqueeze(0),
            ) * current.strides.index_select(1, selected_anchor_indices)
            current.angle = predictions["angle"]
            class_indices = (flat_indices % current.nc).unsqueeze(-1).float()
            return torch.cat(
                (
                    boxes.transpose(1, 2),
                    selected_scores.unsqueeze(-1),
                    class_indices,
                    angles.transpose(1, 2),
                ),
                dim=-1,
            )

        head.forward = types.MethodType(static_end_to_end_obb_forward, head)

    wrapper.static_export_paths_prepared = True


def _prepare_ultralytics_model(model: Any, config: ExportConfig, torch: Any) -> Any:
    model = copy.deepcopy(model).to(config.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.float()
    fuse = getattr(model, "fuse", None)
    if callable(fuse):
        try:
            fused_model = fuse(verbose=False)
        except TypeError:
            fused_model = fuse()
        if fused_model is not None:
            model = fused_model
    if config.model_precision == "float16":
        model.half()
    _configure_ultralytics_head(model)
    _fold_input_normalization_into_first_conv(model, config, torch)
    model._standalone_static_split_modules = _specialize_static_channel_splits(model)
    return model


def _extract_end_to_end_output(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def _flatten_tensor_outputs(value: Any, torch: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(
            output
            for item in value
            for output in _flatten_tensor_outputs(item, torch)
        )
    return ()


def _compact_ultralytics_predictions(result: Any, head: Any, torch: Any) -> Any:
    if isinstance(result, tuple):
        boxes, scores = result
        extras = boxes[:, 4:] if boxes.shape[1] > 4 else boxes[:, :0]
        boxes = boxes[:, :4]
    else:
        minimum_channels = 4 + head.nc
        boxes = result[:, :4]
        scores = result[:, 4:minimum_channels]
        extras = result[:, minimum_channels:]
    max_scores, max_indices = torch.max(scores, dim=1, keepdim=True)
    return torch.cat(
        (boxes, max_scores, max_indices.to(dtype=max_scores.dtype), extras),
        dim=1,
    ).transpose(1, 2)


def _adapt_ultralytics_outputs(
    result: Any,
    head: Any,
    model_task: str,
    model_dtype: Any,
    torch: Any,
) -> Any:
    auxiliary_outputs: tuple[Any, ...] = ()
    prediction = result
    if model_task == "segment":
        prediction = result[0]
        auxiliary_outputs = tuple(
            output
            for item in result[1:]
            for output in _flatten_tensor_outputs(item, torch)
        )

    if model_task in DETECTION_LIKE_TASKS:
        if bool(getattr(head, "end2end", False)):
            primary_output = _extract_end_to_end_output(prediction)
        else:
            primary_output = _compact_ultralytics_predictions(prediction, head, torch)
        outputs = tuple(
            output.to(dtype=model_dtype)
            for output in (primary_output, *auxiliary_outputs)
        )
    else:
        outputs = _flatten_tensor_outputs(result, torch)

    return outputs[0] if len(outputs) == 1 else outputs


def build_ultralytics_wrapper(model: Any, config: ExportConfig) -> Any:
    torch = _import_torch()
    layers = _configure_ultralytics_head(model)
    first_parameter = next(model.parameters(), None)
    model_dtype = first_parameter.dtype if first_parameter is not None else torch.float32

    class StaticUltralyticsWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = layers
            self.model_task = config.model_task
            self.model_dtype = model_dtype
            self.resize_scale = _static_resize_scale(config)
            self.input_padding = _stride_aligned_padding(
                config.resize_shape,
                getattr(layers[-1], "stride", ()),
            )
            self.input_normalization_folded = bool(
                getattr(model, "_standalone_input_normalization_folded", False)
            )
            self.static_split_modules = tuple(
                getattr(model, "_standalone_static_split_modules", ())
            )

        def _preprocess(self, images: Any) -> Any:
            images = torch.nn.functional.interpolate(
                images.float(),
                scale_factor=self.resize_scale,
                mode="bilinear",
                align_corners=True,
                recompute_scale_factor=False,
            )
            if any(self.input_padding):
                images = torch.nn.functional.pad(
                    images,
                    self.input_padding,
                    value=114.0,
                )
            if not self.input_normalization_folded:
                images = images * (1.0 / 255.0)
            return images.to(dtype=self.model_dtype)

        def prepare_for_export(self, images: Any) -> None:
            if getattr(self, "static_export_paths_prepared", False):
                self(images)
                return
            _specialize_static_attention(self, images, config, torch)

        def _run_layers(self, images: Any) -> Any:
            value = self._preprocess(images)
            layer_outputs = []
            for module in self.layers:
                connection = module.f
                if connection != -1:
                    if isinstance(connection, int):
                        value = layer_outputs[connection]
                    else:
                        value = [
                            value if reference == -1 else layer_outputs[reference]
                            for reference in connection
                        ]
                value = module(value)
                layer_outputs.append(value)
            return value

        def forward(self, images: Any) -> Any:
            result = self._run_layers(images)
            return _adapt_ultralytics_outputs(
                result,
                self.layers[-1],
                self.model_task,
                self.model_dtype,
                torch,
            )

    return StaticUltralyticsWrapper().eval()


def load_ultralytics_wrapper(
    config: ExportConfig,
) -> tuple[Any, dict[str, str], tuple[str, ...], ExportConfig]:
    torch = _import_torch()
    installed_version = _package_version("ultralytics")
    ultralytics = _module("ultralytics")
    model_options = {} if config.model_task == "auto" else {"task": config.model_task}
    yolo = ultralytics.YOLO(str(config.model_path), **model_options)
    resolved_task = str(
        getattr(yolo, "task", None)
        or getattr(yolo.model, "task", None)
        or config.model_task
    )
    config = replace(config, model_task=resolved_task)
    model = _prepare_ultralytics_model(yolo.model, config, torch)
    class_names = _extract_model_class_names(yolo, model)
    return (
        build_ultralytics_wrapper(model, config),
        {"ultralytics": installed_version},
        class_names,
        config,
    )


def _legacy_yolo_nas_distance_to_box(points: Any, distance: Any, torch: Any) -> Any:
    left_top, right_bottom = torch.split(distance, 2, dim=-1)
    return torch.cat(
        [points.unsqueeze(0) + 0.5 * (right_bottom - left_top), right_bottom + left_top],
        dim=-1,
    )


def _yolo_nas_modules(model: Any) -> tuple[Any, Any, Any]:
    return model.backbone, model.neck, model.heads


def _prepare_yolo_nas_model(model: Any, config: ExportConfig) -> Any:
    model = model.to(config.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.float()
    if config.model_precision == "float16":
        model.half()
    return model


def build_yolo_nas_wrapper(model: Any, config: ExportConfig) -> Any:
    torch = _import_torch()
    backbone, neck, heads = _yolo_nas_modules(model)
    first_parameter = next(model.parameters(), None)
    model_dtype = first_parameter.dtype if first_parameter is not None else torch.float32

    class StaticYoloNasWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.neck = neck
            self.heads = heads
            self.model_dtype = model_dtype
            self.resize_scale = _static_resize_scale(config)
            self.register_buffer("anchor_points", None, persistent=False)
            self.register_buffer("stride_tensor", None, persistent=False)

        def _preprocess(self, images: Any) -> Any:
            images = torch.nn.functional.interpolate(
                images.float(),
                scale_factor=self.resize_scale,
                mode="bilinear",
                align_corners=True,
                recompute_scale_factor=False,
            )
            return (images * (1.0 / 255.0)).to(dtype=self.model_dtype)

        def prepare_for_export(self, images: Any) -> None:
            features = tuple(self.neck(self.backbone(self._preprocess(images))))
            anchor_points, stride_tensor = self.heads._generate_anchors(features)
            self.anchor_points = anchor_points.detach()
            self.stride_tensor = stride_tensor.detach()

        def forward(self, images: Any) -> Any:
            features = tuple(self.neck(self.backbone(self._preprocess(images))))
            max_class_scores = []
            max_class_indices = []
            reduced_distances = []
            for index, feature in enumerate(features, start=1):
                _, _, height, width = feature.shape
                anchors = height * width
                distances, logits = getattr(self.heads, f"head{index}")(feature)
                logits = logits.reshape([1, self.heads.num_classes, anchors])
                max_scores, max_indices = torch.max(logits.sigmoid(), dim=1, keepdim=True)
                distances = distances.reshape(
                    [-1, 4, self.heads.reg_max + 1, anchors]
                ).permute(0, 2, 3, 1)
                distances = torch.nn.functional.conv2d(
                    torch.nn.functional.softmax(distances, dim=1),
                    weight=self.heads.proj_conv,
                ).squeeze(1)
                max_class_scores.append(max_scores)
                max_class_indices.append(max_indices)
                reduced_distances.append(distances)

            if self.anchor_points is None or self.stride_tensor is None:
                anchor_points, stride_tensor = self.heads._generate_anchors(features)
            else:
                anchor_points = self.anchor_points
                stride_tensor = self.stride_tensor
            boxes = _legacy_yolo_nas_distance_to_box(
                anchor_points,
                torch.cat(reduced_distances, dim=1),
                torch,
            ) * stride_tensor
            max_scores = torch.cat(max_class_scores, dim=2).transpose(1, 2)
            max_indices = torch.cat(max_class_indices, dim=2).transpose(1, 2).to(
                dtype=self.model_dtype
            )
            return torch.cat((boxes, max_scores, max_indices), dim=2)

    return StaticYoloNasWrapper().eval()


def load_yolo_nas_wrapper(
    config: ExportConfig,
) -> tuple[Any, dict[str, str], tuple[str, ...]]:
    installed_version = _package_version("super-gradients")
    models = _module("super_gradients.training.models")
    object_names = _module("super_gradients.common.object_names")
    model_name = getattr(object_names.Models, f"YOLO_NAS_{config.yolo_nas_variant.upper()}")
    model = models.get(
        model_name,
        num_classes=config.yolo_nas_num_classes,
        download_required_code=False,
        checkpoint_path=str(config.model_path),
    )
    model = _prepare_yolo_nas_model(model, config)
    class_names = _extract_model_class_names(model)
    return (
        build_yolo_nas_wrapper(model, config),
        {"super-gradients": installed_version},
        class_names,
    )


def _legacy_output_name(config: ExportConfig) -> str:
    return LEGACY_OUTPUT_NAMES[config.model_family]


def _output_names(config: ExportConfig, output_count: int) -> tuple[str, ...]:
    if config.model_family == "yolo_nas":
        return (_legacy_output_name(config),)
    return tuple(f"output{index}" for index in range(output_count))


def _as_output_sequence(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _export_static_onnx(wrapper: Any, config: ExportConfig) -> Any:
    torch = _import_torch()
    onnx = _module("onnx")
    dummy_dtype = torch.uint8 if config.public_input_dtype == "uint8" else torch.float32
    dummy_input = torch.ones(
        config.input_shape,
        dtype=dummy_dtype,
        device=config.device,
    )
    prepare_for_export = getattr(wrapper, "prepare_for_export", None)
    if callable(prepare_for_export):
        with torch.inference_mode():
            prepare_for_export(dummy_input)
    with torch.inference_mode():
        output_names = _output_names(
            config,
            len(_as_output_sequence(wrapper(dummy_input))),
        )

    serialized_model = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            serialized_model,
            input_names=["images"],
            output_names=list(output_names),
            opset_version=config.opset_version,
            dynamo=False,
            do_constant_folding=True,
        )
    return onnx.load_model_from_string(serialized_model.getvalue())


def _add_onnx_metadata(
    onnx_model: Any,
    config: ExportConfig,
    dependency_versions: dict[str, str],
    class_names: tuple[str, ...],
    artifact_metadata: dict[str, str],
) -> None:
    metadata = {
        "standalone_exporter": Path(__file__).name,
        "model_family": config.model_family,
        "model_task": config.model_task,
        "model_source": str(config.model_path),
        "input_name": "images",
        "input_dtype": config.public_input_dtype,
        "input_shape": str(config.input_shape),
        "output_name": _legacy_output_name(config),
        "output_names": ",".join(value.name for value in onnx_model.graph.output),
        "resize_shape": str(config.resize_shape),
        "static_shapes": str(config.static_shapes),
        "opset": str(config.opset_version),
        **artifact_metadata,
    }
    if class_names:
        metadata["class_names"] = json.dumps(class_names, ensure_ascii=True)
    if config.model_family == "ultralytics":
        metadata["yolo_version"] = str(config.yolo_version)
        metadata["license"] = "AGPL-3.0 License (https://ultralytics.com/license)"
    else:
        metadata["yolo_nas_variant"] = config.yolo_nas_variant
        metadata["yolo_nas_num_classes"] = str(config.yolo_nas_num_classes)
        metadata["license"] = "Apache-2.0 (https://github.com/Deci-AI/super-gradients)"
    metadata.update(
        {f"dependency.{key}": value for key, value in dependency_versions.items()}
    )

    existing = {entry.key: entry for entry in onnx_model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = value


def _finalize_onnx(
    onnx_model: Any,
    config: ExportConfig,
    dependency_versions: dict[str, str],
    class_names: tuple[str, ...],
    artifact_metadata: dict[str, str],
) -> Any:
    if getattr(onnx_model, "ir_version", 0) > 10:
        onnx_model.ir_version = 10
    _add_onnx_metadata(
        onnx_model,
        config,
        dependency_versions,
        class_names,
        artifact_metadata,
    )
    return onnx_model


def _publish_onnx_atomically(onnx_model: Any, output_path: Path) -> None:
    onnx = _module("onnx")
    onnx.shape_inference.infer_shapes(
        onnx_model,
        strict_mode=True,
        data_prop=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".tmp.onnx",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        onnx.save_model(onnx_model, str(temporary_path), save_as_external_data=False)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _onnx_element_type_name(onnx: Any, element_type: int) -> str:
    return onnx.TensorProto.DataType.Name(element_type).lower()


def _onnx_value_shape(value_info: Any) -> tuple[int | str, ...]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return tuple(dimensions)


def _onnx_tensor_nbytes(onnx: Any, element_type: int, shape: tuple[Any, ...]) -> int | None:
    if any(not isinstance(dimension, int) for dimension in shape):
        return None
    try:
        itemsize = onnx.helper.tensor_dtype_to_np_dtype(element_type).itemsize
    except (KeyError, TypeError, ValueError):
        return None
    return math.prod(shape) * itemsize


def _onnx_graph_metrics(onnx_model: Any) -> dict[str, Any]:
    onnx = _module("onnx")
    inferred_model = onnx.shape_inference.infer_shapes(
        onnx_model,
        strict_mode=True,
        data_prop=True,
    )
    graph = inferred_model.graph
    histogram = Counter(node.op_type for node in graph.node)
    initializer_names = {initializer.name for initializer in graph.initializer}

    tensor_bytes: dict[str, int] = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        tensor_type = value.type.tensor_type
        nbytes = _onnx_tensor_nbytes(
            onnx,
            tensor_type.elem_type,
            _onnx_value_shape(value),
        )
        if nbytes is not None:
            tensor_bytes[value.name] = nbytes
    for initializer in graph.initializer:
        nbytes = _onnx_tensor_nbytes(
            onnx,
            initializer.data_type,
            tuple(initializer.dims),
        )
        if nbytes is not None:
            tensor_bytes[initializer.name] = nbytes

    known_references = 0
    total_references = 0
    estimated_bytes_moved = 0
    for node in graph.node:
        for name in (*node.input, *node.output):
            if not name:
                continue
            total_references += 1
            nbytes = tensor_bytes.get(name)
            if nbytes is not None:
                known_references += 1
                estimated_bytes_moved += nbytes

    consumers = Counter(
        name
        for node in graph.node
        for name in node.input
        if name and name not in initializer_names
    )
    graph_output_names = {value.name for value in graph.output}
    live = {
        value.name: tensor_bytes[value.name]
        for value in graph.input
        if value.name not in initializer_names and value.name in tensor_bytes
    }
    peak_live_activation_bytes = sum(live.values())
    for node in graph.node:
        for name in node.output:
            if name in tensor_bytes:
                live[name] = tensor_bytes[name]
        peak_live_activation_bytes = max(
            peak_live_activation_bytes,
            sum(live.values()),
        )
        for name in node.input:
            if name not in consumers:
                continue
            consumers[name] -= 1
            if consumers[name] == 0 and name not in graph_output_names:
                live.pop(name, None)

    cast_ops = {"Cast", "CastLike"}
    shape_ops = {"Shape", "Size"}
    layout_ops = {
        "DepthToSpace",
        "Flatten",
        "Reshape",
        "SpaceToDepth",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    }
    indexing_ops = {
        "Gather",
        "GatherElements",
        "GatherND",
        "NonZero",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Slice",
        "Split",
        "TopK",
    }
    elementwise_ops = {
        "Abs",
        "Add",
        "And",
        "Cast",
        "Ceil",
        "Clip",
        "Div",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Floor",
        "Gelu",
        "Greater",
        "GreaterOrEqual",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Log",
        "Mod",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "Pow",
        "Reciprocal",
        "Relu",
        "Round",
        "Sigmoid",
        "Sign",
        "Sin",
        "Cos",
        "Sqrt",
        "Sub",
        "Tanh",
        "Where",
        "Xor",
    }
    definite_copy_ops = {
        "Concat",
        "Expand",
        "Pad",
        "ScatterND",
        "Tile",
        "Transpose",
    }

    def count_ops(operator_types: set[str]) -> int:
        return sum(histogram[operator_type] for operator_type in operator_types)

    def interface(values: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": value.name,
                "dtype": _onnx_element_type_name(
                    onnx,
                    value.type.tensor_type.elem_type,
                ),
                "shape": list(_onnx_value_shape(value)),
            }
            for value in values
        ]

    custom_nodes = sorted(
        {
            f"{node.domain or 'ai.onnx'}::{node.op_type}"
            for node in graph.node
            if node.domain not in {"", "ai.onnx"}
            or node.op_type.startswith(("ATen", "PythonOp"))
        }
    )
    return {
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "initializer_bytes": sum(
            tensor_bytes.get(initializer.name, 0)
            for initializer in graph.initializer
        ),
        "operator_histogram": dict(sorted(histogram.items())),
        "cast_ops": count_ops(cast_ops),
        "shape_ops": count_ops(shape_ops),
        "layout_ops": count_ops(layout_ops),
        "indexing_ops": count_ops(indexing_ops),
        "elementwise_ops": count_ops(elementwise_ops),
        "definite_materializing_copy_ops": count_ops(definite_copy_ops),
        "reshape_ops_materialization_unknown": histogram["Reshape"],
        "estimated_bytes_moved": estimated_bytes_moved,
        "estimated_peak_live_activation_bytes": peak_live_activation_bytes,
        "bytes_moved_known_reference_ratio": (
            known_references / total_references if total_references else 1.0
        ),
        "inputs": interface(graph.input),
        "outputs": interface(graph.output),
        "opset_imports": [
            [entry.domain or "ai.onnx", entry.version]
            for entry in inferred_model.opset_import
        ],
        "custom_or_fallback_nodes": custom_nodes,
    }


def _artifact_report(
    onnx_model: Any,
    output_path: Path,
    config: ExportConfig,
    dependency_versions: dict[str, str],
    published: bool,
) -> dict[str, Any]:
    onnx = _module("onnx")
    return {
        "artifact": str(output_path),
        "published": published,
        "bytes": len(onnx_model.SerializeToString()),
        "family": config.model_family,
        "task": config.model_task,
        "source_weights": str(config.model_path),
        "input": {
            "name": "images",
            "dtype": config.public_input_dtype,
            "shape": list(config.input_shape),
        },
        "outputs": [
            {
                "name": value.name,
                "dtype": _onnx_element_type_name(
                    onnx,
                    value.type.tensor_type.elem_type,
                ),
                "shape": list(_onnx_value_shape(value)),
            }
            for value in onnx_model.graph.output
        ],
        "opset": [
            [entry.domain or "ai.onnx", entry.version]
            for entry in onnx_model.opset_import
        ],
        "ir_version": onnx_model.ir_version,
        "dependencies": dependency_versions,
        "graph_metrics": _onnx_graph_metrics(onnx_model),
        "inference": (
            "runs automatically after publication"
            if RUN_INFERENCE_DEMO
            else "disabled"
        ),
    }


def export_android_onnx(config: ExportConfig) -> dict[str, Any]:
    output_path = config.output_path.resolve()
    if output_path.exists() and not config.overwrite_output:
        onnx_model = _module("onnx").load(str(output_path), load_external_data=False)
        report = _artifact_report(onnx_model, output_path, config, {}, False)
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    _ensure_model_checkpoint(config)
    if config.model_family == "ultralytics":
        wrapper, dependency_versions, class_names, config = load_ultralytics_wrapper(config)
    else:
        wrapper, dependency_versions, class_names = load_yolo_nas_wrapper(config)

    onnx_model = _export_static_onnx(wrapper, config)
    head = wrapper.layers[-1] if hasattr(wrapper, "layers") else None
    artifact_metadata = {
        "end_to_end": str(bool(getattr(head, "end2end", False))),
    }
    keypoint_shape = getattr(head, "kpt_shape", None)
    if isinstance(keypoint_shape, (tuple, list)):
        artifact_metadata["keypoint_shape"] = json.dumps(tuple(keypoint_shape))
    onnx_model = _finalize_onnx(
        onnx_model,
        config,
        dependency_versions,
        class_names,
        artifact_metadata,
    )
    _publish_onnx_atomically(onnx_model, output_path)
    report = _artifact_report(
        onnx_model,
        output_path,
        config,
        dependency_versions,
        True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if RUN_INFERENCE_DEMO:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("Inference_YOLO_ONNX.py")),
                "--model",
                str(output_path),
                "--task",
                config.model_task,
            ],
            check=False,
        )
    return report


def main() -> None:
    export_android_onnx(ExportConfig.from_user_configuration())


if __name__ == "__main__":
    main()
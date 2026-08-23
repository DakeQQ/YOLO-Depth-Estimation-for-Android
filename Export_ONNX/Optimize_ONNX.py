#!/usr/bin/env python3
"""Optimize and optionally quantize standalone YOLO ONNX models.

The AFFINE_REFINE_V2 objective and pipeline structure follow the Qwen-v3
optimizer, adapted to YOLO's constant Conv weights. Q4/Q8 are represented with
standard blocked DequantizeLinear weights; DYNAMIC emits runtime activation
quantization and ConvInteger nodes. F16 converts floating-point graph values to
float16, while F32 runs the graph optimization passes without quantization.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim


# ============================================================================
# USER OPTIMIZATION CONFIGURATION
# Edit this section to select the model, quantization, and graph passes.
# ============================================================================
ORIGINAL_MODEL_PATH = Path(__file__).with_name("models") / "detect" / "yolo26n-det.onnx"
OPTIMIZED_MODEL_PATH = None            # None writes <source-stem>_<method>.onnx beside the source.
QUANT_METHOD = "DYNAMIC"               # "Q4" | "Q8" | "DYNAMIC" | "F16" | "F32".
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 32                        # Q4/Q8 blocked Conv weights: power of two in [16, 256].
QUANT_SYMMETRIC = False                # Q4/Q8: False usually improves weight reconstruction.
DYNAMIC_WEIGHT_SYMMETRIC = True        # Required for portable per-channel ConvInteger zero points.
NODES_TO_INCLUDE = None                # Optional Conv node-name allowlist.
NODES_TO_EXCLUDE = None                # Optional Conv node-name denylist.
OVERWRITE_OUTPUT = True

# Qwen-v3 AFFINE_REFINE_V2 controls. The seed protects large weights; the main
# pass minimizes plain block MSE under a magnitude-weighted Pareto bound.
AFFINE_V2_SEED_ITERATIONS = 4
AFFINE_V2_SEED_ZP_RADIUS = 2
AFFINE_V2_ITERATIONS = 6
AFFINE_V2_CLIP_RATIOS = (1.0, 0.94, 0.82, 0.70, 0.55)
AFFINE_V2_WEIGHTED_TOLERANCE = 0.15
AFFINE_V2_ASYM_ZP_SWEEP_LIMIT = 32

# Qwen-v3 optimization order: onnxslim -> Transformers optimizer -> onnxslim.
RUN_ONNXSLIM = True
SLIM_NO_SHAPE_INFER = False
SLIM_SKIP_FUSION_PATTERNS = None
SLIM_SKIP_OPTIMIZATIONS = None
SLIM_SIZE_THRESHOLD = None
RUN_TRANSFORMERS_OPTIMIZER = True
OPTIMIZER_LEVEL = 2                    # 0 | 1 | 2 | 99.
OPTIMIZER_MODEL_TYPE = "bert"          # Generic template used by the reference pipeline.
OPTIMIZER_NUM_HEADS = 0
OPTIMIZER_HIDDEN_SIZE = 0
OPTIMIZER_USE_GPU = False
OPTIMIZER_PROVIDER = "CPUExecutionProvider"
OPTIMIZER_ONLY_ONNXRUNTIME = False

VERIFY_WITH_ONNXRUNTIME = True
VERIFY_PROVIDER = "CPUExecutionProvider"
OVERWRITE_METADATA = True


class OptimizationConfigurationError(ValueError):
    """Raised when the selected optimization plan is invalid."""


class OptimizationValidationError(RuntimeError):
    """Raised when a rewritten or optimized model violates its contract."""


@dataclass
class RefineStats:
    blocks: int = 0
    improved_blocks: int = 0
    seed_error: float = 0.0
    refined_error: float = 0.0

    def add(self, other: "RefineStats") -> None:
        self.blocks += other.blocks
        self.improved_blocks += other.improved_blocks
        self.seed_error += other.seed_error
        self.refined_error += other.refined_error


@dataclass(frozen=True)
class OptimizeConfig:
    source_path: Path
    output_path: Path
    method: str
    block_size: int
    symmetric: bool
    dynamic_symmetric: bool
    nodes_to_include: tuple[str, ...] | None
    nodes_to_exclude: tuple[str, ...] | None
    overwrite_output: bool
    verify_with_onnxruntime: bool

    @classmethod
    def from_user_configuration(cls) -> "OptimizeConfig":
        source_path = Path(ORIGINAL_MODEL_PATH)
        method = QUANT_METHOD.upper()
        output_path = (
            Path(OPTIMIZED_MODEL_PATH)
            if OPTIMIZED_MODEL_PATH is not None
            else source_path.with_name(f"{source_path.stem}_{method.lower()}.onnx")
        )
        return cls(
            source_path=source_path,
            output_path=output_path,
            method=method,
            block_size=BLOCK_SIZE,
            symmetric=QUANT_SYMMETRIC,
            dynamic_symmetric=DYNAMIC_WEIGHT_SYMMETRIC,
            nodes_to_include=(
                tuple(NODES_TO_INCLUDE) if NODES_TO_INCLUDE is not None else None
            ),
            nodes_to_exclude=(
                tuple(NODES_TO_EXCLUDE) if NODES_TO_EXCLUDE is not None else None
            ),
            overwrite_output=OVERWRITE_OUTPUT,
            verify_with_onnxruntime=VERIFY_WITH_ONNXRUNTIME,
        )


def validate_config(config: OptimizeConfig) -> None:
    if config.method not in {"Q4", "Q8", "DYNAMIC", "F16", "F32"}:
        raise OptimizationConfigurationError(
            "QUANT_METHOD must be 'Q4', 'Q8', 'DYNAMIC', 'F16', or 'F32'; "
            f"received {config.method!r}."
        )
    uses_affine_quantization = config.method in {"Q4", "Q8", "DYNAMIC"}
    if uses_affine_quantization and WEIGHT_ONLY_ALGORITHM != "AFFINE_REFINE_V2":
        raise OptimizationConfigurationError(
            "This YOLO optimizer supports WEIGHT_ONLY_ALGORITHM='AFFINE_REFINE_V2' only."
        )
    if not config.source_path.is_file():
        raise OptimizationConfigurationError(
            f"ORIGINAL_MODEL_PATH does not exist or is not a file: {config.source_path}."
        )
    if config.source_path.suffix.lower() != ".onnx":
        raise OptimizationConfigurationError("ORIGINAL_MODEL_PATH must end in .onnx.")
    if config.output_path.suffix.lower() != ".onnx":
        raise OptimizationConfigurationError("OPTIMIZED_MODEL_PATH must end in .onnx.")
    if config.source_path.resolve() == config.output_path.resolve():
        raise OptimizationConfigurationError(
            "OPTIMIZED_MODEL_PATH must differ from ORIGINAL_MODEL_PATH so a failed run cannot "
            "destroy the source artifact."
        )
    if config.output_path.exists() and not config.overwrite_output:
        raise OptimizationConfigurationError(
            f"OPTIMIZED_MODEL_PATH already exists: {config.output_path}. Set OVERWRITE_OUTPUT=True "
            "or choose another path."
        )
    if uses_affine_quantization and (
        config.block_size < 16
        or config.block_size > 256
        or config.block_size & (config.block_size - 1)
    ):
        raise OptimizationConfigurationError(
            f"BLOCK_SIZE must be a power of two in [16, 256]; received {config.block_size}."
        )
    if config.method == "DYNAMIC" and not config.dynamic_symmetric:
        raise OptimizationConfigurationError(
            "DYNAMIC_WEIGHT_SYMMETRIC must remain True: ORT ConvInteger requires a scalar "
            "weight zero point, while per-channel asymmetric zero points fail at execution time."
        )
    if uses_affine_quantization and (
        AFFINE_V2_SEED_ITERATIONS < 1 or AFFINE_V2_ITERATIONS < 1
    ):
        raise OptimizationConfigurationError("AFFINE_REFINE_V2 iteration counts must be positive.")
    if uses_affine_quantization and AFFINE_V2_SEED_ZP_RADIUS < 0:
        raise OptimizationConfigurationError("AFFINE_V2_SEED_ZP_RADIUS must be nonnegative.")
    if uses_affine_quantization and AFFINE_V2_WEIGHTED_TOLERANCE < 0.0:
        raise OptimizationConfigurationError("AFFINE_V2_WEIGHTED_TOLERANCE must be nonnegative.")
    if uses_affine_quantization and AFFINE_V2_ASYM_ZP_SWEEP_LIMIT < 16:
        raise OptimizationConfigurationError(
            "AFFINE_V2_ASYM_ZP_SWEEP_LIMIT must be at least 16 so Q4 sweeps every zero point."
        )
    if uses_affine_quantization:
        ratios = np.asarray(AFFINE_V2_CLIP_RATIOS, dtype=np.float32)
        if ratios.ndim != 1 or not ratios.size or np.any((ratios <= 0.0) | (ratios > 1.0)):
            raise OptimizationConfigurationError(
                "AFFINE_V2_CLIP_RATIOS must be a non-empty sequence in (0, 1]."
            )
    if OPTIMIZER_LEVEL not in {0, 1, 2, 99}:
        raise OptimizationConfigurationError("OPTIMIZER_LEVEL must be 0, 1, 2, or 99.")


def _initial_affine_seed(
    weight: np.ndarray,
    bits: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the deterministic magnitude-aware seed used by AFFINE_REFINE_V2."""
    maxq = np.float32((1 << bits) - 1)
    midpoint = np.float32(1 << (bits - 1))
    tiny = np.finfo(np.float32).tiny
    minimum = weight.min(axis=1, keepdims=True)
    maximum = weight.max(axis=1, keepdims=True)

    if symmetric:
        positive_max = np.maximum(maximum, np.float32(0.0))
        negative_max = np.maximum(-minimum, np.float32(0.0))
        scale = np.maximum(
            positive_max / (maxq - midpoint),
            negative_max / midpoint,
        )
        scale = np.where(scale > tiny, scale, np.float32(1.0)).astype(np.float32)
        zero_point = np.full((weight.shape[0], 1), midpoint, dtype=np.float32)
    else:
        span = maximum - minimum
        scale = np.where(span > tiny, span / maxq, np.float32(1.0)).astype(np.float32)
        zero_point = np.clip(np.rint(-minimum / scale), 0.0, maxq).astype(np.float32)

    quantized = np.clip(np.rint(weight / scale + zero_point), 0.0, maxq)
    rms = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True, dtype=np.float32))
    importance = rms + np.abs(weight)
    importance = np.where(
        importance.sum(axis=1, keepdims=True) > 0.0,
        importance,
        np.float32(1.0),
    )
    best_error = np.sum(
        importance * (weight - scale * (quantized - zero_point)) ** 2,
        axis=1,
    )
    best_q = quantized.copy()
    best_scale = scale.copy()
    best_zp = zero_point.copy()

    deltas = (0,) if symmetric else range(-AFFINE_V2_SEED_ZP_RADIUS, AFFINE_V2_SEED_ZP_RADIUS + 1)
    for delta in deltas:
        candidate_zp = np.clip(zero_point + np.float32(delta), 0.0, maxq)
        candidate_scale = scale.copy()
        for _ in range(AFFINE_V2_SEED_ITERATIONS):
            candidate_q = np.clip(
                np.rint(weight / candidate_scale + candidate_zp),
                0.0,
                maxq,
            )
            centered = candidate_q - candidate_zp
            denominator = np.sum(importance * centered * centered, axis=1, keepdims=True)
            numerator = np.sum(importance * centered * weight, axis=1, keepdims=True)
            fitted = np.divide(
                numerator,
                denominator,
                out=candidate_scale.copy(),
                where=denominator > tiny,
            )
            candidate_scale = np.where(
                np.isfinite(fitted) & (fitted > tiny),
                fitted,
                candidate_scale,
            )
        candidate_q = np.clip(
            np.rint(weight / candidate_scale + candidate_zp),
            0.0,
            maxq,
        )
        candidate_error = np.sum(
            importance * (weight - candidate_scale * (candidate_q - candidate_zp)) ** 2,
            axis=1,
        )
        take = candidate_error < best_error
        best_q[take] = candidate_q[take]
        best_scale[take] = candidate_scale[take]
        best_zp[take] = candidate_zp[take]
        best_error[take] = candidate_error[take]

    return best_q, best_scale, best_zp


def _affine_refine_v2_blocks(
    values: np.ndarray,
    bits: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RefineStats]:
    """Minimize plain block MSE under the Qwen-v3 weighted Pareto guard."""
    weight = np.ascontiguousarray(values, dtype=np.float32)
    if weight.ndim != 2 or weight.shape[1] == 0:
        raise OptimizationValidationError(
            f"AFFINE_REFINE_V2 expects non-empty 2-D blocks, received {weight.shape}."
        )
    if not np.isfinite(weight).all():
        raise OptimizationValidationError("AFFINE_REFINE_V2 refuses NaN or Inf weights.")

    maxq_int = (1 << bits) - 1
    maxq = np.float32(maxq_int)
    midpoint = 1 << (bits - 1)
    tiny = np.finfo(np.float32).tiny
    clip_ratios = np.asarray(AFFINE_V2_CLIP_RATIOS, dtype=np.float32)
    best_q, best_scale, best_zp = _initial_affine_seed(weight, bits, symmetric)

    rms = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True, dtype=np.float32))
    importance = rms + np.abs(weight)
    importance = np.where(
        importance.sum(axis=1, keepdims=True) > 0.0,
        importance,
        np.float32(1.0),
    )
    seed_residual = weight - best_scale * (best_q - best_zp)
    baseline_plain = np.sum(seed_residual * seed_residual, axis=1)
    baseline_weighted = np.sum(importance * seed_residual * seed_residual, axis=1)
    weighted_bound = np.float32(1.0 + AFFINE_V2_WEIGHTED_TOLERANCE) * baseline_weighted
    local_plain = baseline_plain.copy()
    positive_max = np.maximum(weight.max(axis=1, keepdims=True), np.float32(0.0))
    negative_max = np.maximum(-weight.min(axis=1, keepdims=True), np.float32(0.0))

    if symmetric:
        zero_point_candidates: list[np.ndarray] = [
            np.full((weight.shape[0], 1), midpoint, dtype=np.float32)
        ]
    elif maxq_int + 1 <= AFFINE_V2_ASYM_ZP_SWEEP_LIMIT:
        zero_point_candidates = [
            np.full((weight.shape[0], 1), value, dtype=np.float32)
            for value in range(maxq_int + 1)
        ]
    else:
        seed_zp = best_zp.astype(np.int64)
        half = AFFINE_V2_ASYM_ZP_SWEEP_LIMIT // 2
        window_low = np.clip(seed_zp - half, 0, maxq_int)
        overflow = np.maximum(
            window_low + AFFINE_V2_ASYM_ZP_SWEEP_LIMIT - 1 - maxq_int,
            0,
        )
        window_low = np.clip(window_low - overflow, 0, maxq_int)
        zero_point_candidates = [
            np.clip(window_low + offset, 0, maxq_int).astype(np.float32)
            for offset in range(AFFINE_V2_ASYM_ZP_SWEEP_LIMIT)
        ]

    for zero_point in zero_point_candidates:
        positive_denominator = maxq - zero_point
        positive_scale = np.divide(
            positive_max,
            positive_denominator,
            out=np.zeros_like(positive_max),
            where=positive_denominator > 0.0,
        )
        negative_scale = np.divide(
            negative_max,
            zero_point,
            out=np.zeros_like(negative_max),
            where=zero_point > 0.0,
        )
        coverage_scale = np.maximum(positive_scale, negative_scale)
        coverage_scale = np.where(coverage_scale > tiny, coverage_scale, np.float32(1.0))
        initial_scales = [best_scale]
        initial_scales.extend(coverage_scale * ratio for ratio in clip_ratios)

        for initial_scale in initial_scales:
            candidate_scale = initial_scale.copy()
            for _ in range(AFFINE_V2_ITERATIONS):
                candidate_q = np.clip(
                    np.rint(weight / candidate_scale + zero_point),
                    0.0,
                    maxq,
                )
                centered = candidate_q - zero_point
                denominator = np.sum(centered * centered, axis=1, keepdims=True)
                numerator = np.sum(centered * weight, axis=1, keepdims=True)
                fitted = np.divide(
                    numerator,
                    denominator,
                    out=candidate_scale.copy(),
                    where=denominator > tiny,
                )
                candidate_scale = np.where(
                    np.isfinite(fitted) & (fitted > tiny),
                    fitted,
                    candidate_scale,
                )

            candidate_q = np.clip(
                np.rint(weight / candidate_scale + zero_point),
                0.0,
                maxq,
            )
            residual = weight - candidate_scale * (candidate_q - zero_point)
            candidate_plain = np.sum(residual * residual, axis=1)
            candidate_weighted = np.sum(importance * residual * residual, axis=1)
            take = (candidate_plain < local_plain) & (candidate_weighted <= weighted_bound)
            best_q[take] = candidate_q[take]
            best_scale[take] = candidate_scale[take]
            best_zp[take] = zero_point[take]
            local_plain[take] = candidate_plain[take]

    stats = RefineStats(
        blocks=weight.shape[0],
        improved_blocks=int(np.count_nonzero(local_plain < baseline_plain)),
        seed_error=float(baseline_plain.sum(dtype=np.float64)),
        refined_error=float(local_plain.sum(dtype=np.float64)),
    )
    return (
        best_q.astype(np.uint8),
        best_scale[:, 0].astype(np.float32),
        best_zp[:, 0].astype(np.uint8),
        stats,
    )


def _quantize_rows(
    rows: np.ndarray,
    block_size: int,
    bits: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RefineStats]:
    values = np.asarray(rows)
    if values.ndim != 2 or values.dtype.kind != "f":
        raise OptimizationValidationError(
            f"Conv weights must be a floating-point matrix, received {values.shape} {values.dtype}."
        )
    row_count, width = values.shape
    block_count = (width + block_size - 1) // block_size
    quantized = np.empty((row_count, width), dtype=np.uint8)
    scales = np.empty((row_count, block_count), dtype=np.float32)
    zero_points = np.empty((row_count, block_count), dtype=np.uint8)
    total = RefineStats()
    for block_index in range(block_count):
        start = block_index * block_size
        end = min(start + block_size, width)
        local_q, local_scale, local_zp, stats = _affine_refine_v2_blocks(
            values[:, start:end],
            bits,
            symmetric,
        )
        quantized[:, start:end] = local_q
        scales[:, block_index] = local_scale
        zero_points[:, block_index] = local_zp
        total.add(stats)
    return quantized, scales, zero_points, total


def _make_uint4_initializer(name: str, values: np.ndarray) -> TensorProto:
    logical = np.ascontiguousarray(values, dtype=np.uint8)
    flat = logical.reshape(-1)
    if flat.size & 1:
        flat = np.pad(flat, (0, 1))
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return helper.make_tensor(
        name,
        TensorProto.UINT4,
        logical.shape,
        packed.tobytes(),
        raw=True,
    )


def _make_quant_initializer(name: str, values: np.ndarray, bits: int) -> TensorProto:
    if bits == 4:
        return _make_uint4_initializer(name, values)
    if bits == 8:
        return numpy_helper.from_array(np.ascontiguousarray(values, dtype=np.uint8), name=name)
    raise OptimizationValidationError(f"Unsupported quantized initializer width: {bits}.")


def _graph_used_names(graph: onnx.GraphProto) -> set[str]:
    used = {value.name for value in graph.input}
    used.update(value.name for value in graph.output)
    used.update(value.name for value in graph.value_info)
    used.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        if node.name:
            used.add(node.name)
        used.update(name for name in node.input if name)
        used.update(name for name in node.output if name)
    return used


def _make_name_factory(graph: onnx.GraphProto, prefix: str):
    used = _graph_used_names(graph)

    def make(suffix: str) -> str:
        base = f"{prefix}{suffix}"
        if base not in used:
            used.add(base)
            return base
        index = 1
        while f"{base}_{index}" in used:
            index += 1
        name = f"{base}_{index}"
        used.add(name)
        return name

    return make


def _node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _drop_unused_initializers(graph: onnx.GraphProto) -> int:
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    keep = [initializer for initializer in graph.initializer if initializer.name in used]
    removed = len(graph.initializer) - len(keep)
    if removed:
        graph.ClearField("initializer")
        graph.initializer.extend(keep)
    return removed


def _ensure_default_opset(model: onnx.ModelProto, minimum: int) -> None:
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            opset.version = max(opset.version, minimum)
            return
    model.opset_import.append(helper.make_opsetid("", minimum))


def _selected(node: onnx.NodeProto, config: OptimizeConfig) -> bool:
    if node.op_type != "Conv":
        return False
    if config.nodes_to_include is not None and node.name not in config.nodes_to_include:
        return False
    if config.nodes_to_exclude is not None and node.name in config.nodes_to_exclude:
        return False
    return True


def _rewrite_qdq_conv(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    weight: TensorProto,
    config: OptimizeConfig,
    bits: int,
    make_name: Any,
) -> tuple[list[onnx.NodeProto], RefineStats] | None:
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim < 3 or weight_array.dtype.kind != "f":
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; "
            "Conv weight must be a floating-point tensor with rank >= 3."
        )
        return None
    output_channels = weight_array.shape[0]
    flattened_width = int(np.prod(weight_array.shape[1:], dtype=np.int64))
    quantized, scales, zero_points, stats = _quantize_rows(
        weight_array.reshape(output_channels, flattened_width),
        config.block_size,
        bits,
        config.symmetric,
    )

    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    shape_name = make_name("shape")
    dequantized_name = make_name("dequantized")
    reshaped_name = make_name("reshaped")
    scale_dtype = weight_array.dtype
    graph.initializer.extend(
        [
            _make_quant_initializer(weight_name, quantized, bits),
            numpy_helper.from_array(scales.astype(scale_dtype, copy=False), name=scale_name),
            _make_quant_initializer(zero_point_name, zero_points, bits),
            numpy_helper.from_array(np.asarray(weight_array.shape, dtype=np.int64), name=shape_name),
        ]
    )
    prefix = f"AFFINE_REFINE_V2_Q{bits}"
    dequantize = helper.make_node(
        "DequantizeLinear",
        [weight_name, scale_name, zero_point_name],
        [dequantized_name],
        name=make_name(f"{prefix}_dequantize"),
        axis=1,
        block_size=config.block_size,
    )
    reshape = helper.make_node(
        "Reshape",
        [dequantized_name, shape_name],
        [reshaped_name],
        name=make_name(f"{prefix}_reshape"),
    )
    conv = onnx.NodeProto()
    conv.CopyFrom(node)
    conv.input[1] = reshaped_name
    conv.name = make_name(f"{node.name or 'conv'}_{prefix}")
    return [dequantize, reshape, conv], stats


def _rewrite_dynamic_conv(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    weight: TensorProto,
    initializer_map: dict[str, TensorProto],
    make_name: Any,
) -> tuple[list[onnx.NodeProto], RefineStats] | None:
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim < 3 or weight_array.dtype != np.float32:
        print(
            f"  AFFINE_REFINE_V2 dynamic: skipping {node.name or weight.name!r}; "
            "portable DynamicQuantizeLinear + ConvInteger requires float32 Conv weights."
        )
        return None
    output_channels = weight_array.shape[0]
    flattened_width = int(np.prod(weight_array.shape[1:], dtype=np.int64))
    quantized, scales, zero_points, stats = _quantize_rows(
        weight_array.reshape(output_channels, flattened_width),
        flattened_width,
        8,
        True,
    )
    midpoint = np.uint8(128)
    if not np.all(zero_points == midpoint):
        raise OptimizationValidationError(
            f"Symmetric dynamic Conv {node.name!r} emitted non-midpoint zero points."
        )

    weight_name = make_name("weight")
    weight_zp_name = make_name("weight_zero_point")
    weight_scale_name = make_name("weight_scale")
    broadcast_shape = (1, output_channels) + (1,) * (weight_array.ndim - 2)
    graph.initializer.extend(
        [
            numpy_helper.from_array(quantized.reshape(weight_array.shape), name=weight_name),
            numpy_helper.from_array(np.asarray(midpoint, dtype=np.uint8), name=weight_zp_name),
            numpy_helper.from_array(scales[:, 0].reshape(broadcast_shape), name=weight_scale_name),
        ]
    )

    activation_q = make_name("activation_quantized")
    activation_scale = make_name("activation_scale")
    activation_zp = make_name("activation_zero_point")
    integer_output = make_name("integer_output")
    float_output = make_name("float_output")
    combined_scale = make_name("combined_scale")
    has_bias = len(node.input) >= 3 and bool(node.input[2])
    scaled_output = make_name("scaled_output") if has_bias else node.output[0]
    prefix = "AFFINE_REFINE_V2_DYNAMIC"
    replacement = [
        helper.make_node(
            "DynamicQuantizeLinear",
            [node.input[0]],
            [activation_q, activation_scale, activation_zp],
            name=make_name(f"{prefix}_quantize_activation"),
        ),
        helper.make_node(
            "ConvInteger",
            [activation_q, weight_name, activation_zp, weight_zp_name],
            [integer_output],
            name=make_name(f"{node.name or 'conv'}_{prefix}_integer"),
            **_node_attributes(node),
        ),
        helper.make_node(
            "Cast",
            [integer_output],
            [float_output],
            name=make_name(f"{prefix}_cast"),
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            [activation_scale, weight_scale_name],
            [combined_scale],
            name=make_name(f"{prefix}_combine_scales"),
        ),
        helper.make_node(
            "Mul",
            [float_output, combined_scale],
            [scaled_output],
            name=make_name(f"{prefix}_dequantize_output"),
        ),
    ]
    if has_bias:
        bias = initializer_map.get(node.input[2])
        if bias is None:
            print(
                f"  AFFINE_REFINE_V2 dynamic: skipping {node.name or weight.name!r}; "
                "Conv bias is not a constant initializer."
            )
            return None
        bias_array = numpy_helper.to_array(bias)
        if bias_array.shape != (output_channels,) or bias_array.dtype != np.float32:
            print(
                f"  AFFINE_REFINE_V2 dynamic: skipping {node.name or weight.name!r}; "
                f"unsupported bias shape/dtype {bias_array.shape}/{bias_array.dtype}."
            )
            return None
        bias_name = make_name("bias")
        graph.initializer.append(
            numpy_helper.from_array(bias_array.reshape(broadcast_shape), name=bias_name)
        )
        replacement.append(
            helper.make_node(
                "Add",
                [scaled_output, bias_name],
                list(node.output),
                name=make_name(f"{prefix}_bias"),
            )
        )
    return replacement, stats


def quantize_affine_v2_model(
    model: onnx.ModelProto,
    config: OptimizeConfig,
) -> tuple[int, RefineStats]:
    if config.method not in {"Q4", "Q8", "DYNAMIC"}:
        raise OptimizationConfigurationError(
            f"AFFINE_REFINE_V2 does not support method {config.method!r}."
        )
    bits = 4 if config.method == "Q4" else 8 if config.method == "Q8" else None
    total = RefineStats()
    quantized_convs = 0

    def rewrite_graph(graph: onnx.GraphProto) -> None:
        nonlocal quantized_convs
        initializer_map = {initializer.name: initializer for initializer in graph.initializer}
        make_name = _make_name_factory(graph, f"affine_refine_v2_{config.method.lower()}_")
        new_nodes: list[onnx.NodeProto] = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)

            replacement = None
            if _selected(node, config) and len(node.input) >= 2:
                weight = initializer_map.get(node.input[1])
                if weight is not None:
                    replacement = (
                        _rewrite_qdq_conv(graph, node, weight, config, bits, make_name)
                        if bits is not None
                        else _rewrite_dynamic_conv(
                            graph,
                            node,
                            weight,
                            initializer_map,
                            make_name,
                        )
                    )
            if replacement is None:
                new_nodes.append(node)
            else:
                replacement_nodes, stats = replacement
                new_nodes.extend(replacement_nodes)
                total.add(stats)
                quantized_convs += 1

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)

    rewrite_graph(model.graph)
    if quantized_convs == 0:
        raise OptimizationValidationError(
            f"{config.method} selected no eligible constant Conv weights; refusing to publish a "
            "model whose requested quantization would be a no-op."
        )
    if bits is not None:
        _ensure_default_opset(model, 21)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2: quantized {quantized_convs} Conv node(s); improved "
        f"{total.improved_blocks}/{total.blocks} blocks over the weighted seed; "
        f"plain MSE ratio={ratio:.6f}."
    )
    return quantized_convs, total


def _value_shape(value: onnx.ValueInfoProto) -> tuple[int | str | None, ...]:
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _interface_signature(model: onnx.ModelProto) -> dict[str, tuple[tuple[Any, ...], ...]]:
    def signature(values: Any) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                value.name,
                value.type.tensor_type.elem_type,
                _value_shape(value),
            )
            for value in values
        )

    return {
        "inputs": signature(model.graph.input),
        "outputs": signature(model.graph.output),
    }


def _update_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    existing = {entry.key: entry for entry in model.metadata_props}
    for key, value in values.items():
        if key in existing:
            if OVERWRITE_METADATA:
                existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)


def _run_onnxslim(model_path: Path) -> None:
    slim(
        model=str(model_path),
        output_model=str(model_path),
        no_shape_infer=SLIM_NO_SHAPE_INFER,
        skip_fusion_patterns=SLIM_SKIP_FUSION_PATTERNS,
        skip_optimizations=SLIM_SKIP_OPTIMIZATIONS,
        size_threshold=SLIM_SIZE_THRESHOLD,
        save_as_external_data=False,
        verbose=False,
    )


def _run_transformers_optimizer(model_path: Path) -> None:
    optimized = optimize_model(
        str(model_path),
        model_type=OPTIMIZER_MODEL_TYPE,
        num_heads=OPTIMIZER_NUM_HEADS,
        hidden_size=OPTIMIZER_HIDDEN_SIZE,
        opt_level=OPTIMIZER_LEVEL,
        use_gpu=OPTIMIZER_USE_GPU,
        only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
        verbose=False,
        provider=OPTIMIZER_PROVIDER,
    )
    optimized.save_model_to_file(str(model_path), use_external_data_format=False)
    del optimized
    gc.collect()


def _convert_model_to_float16(model: onnx.ModelProto) -> onnx.ModelProto:
    converted = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        force_fp16_initializers=True,
    )
    if not any(
        initializer.data_type == TensorProto.FLOAT16
        for initializer in converted.graph.initializer
    ):
        raise OptimizationValidationError(
            "F16 conversion produced no float16 initializers; refusing to publish a no-op model."
        )
    producers: dict[str, onnx.NodeProto] = {}
    unique_nodes: list[onnx.NodeProto] = []
    for node in converted.graph.node:
        outputs = [name for name in node.output if name]
        conflicts = [producers[name] for name in outputs if name in producers]
        if conflicts:
            serialized = node.SerializeToString()
            if all(existing.SerializeToString() == serialized for existing in conflicts):
                continue
            raise OptimizationValidationError(
                f"F16 conversion produced conflicting nodes for output {outputs!r}."
            )
        unique_nodes.append(node)
        producers.update((name, node) for name in outputs)
    if len(unique_nodes) != len(converted.graph.node):
        converted.graph.ClearField("node")
        converted.graph.node.extend(unique_nodes)
    from onnxruntime.transformers.onnx_model import OnnxModel

    sorted_model = OnnxModel(converted)
    sorted_model.topological_sort()
    return sorted_model.model


def _coverage(model: onnx.ModelProto, method: str) -> int:
    method_token = f"AFFINE_REFINE_V2_{method}"
    if method in {"Q4", "Q8"}:
        return sum(
            node.op_type == "DequantizeLinear" and method_token in node.name
            for node in model.graph.node
        )
    return sum(
        node.op_type == "ConvInteger" and method_token in node.name
        for node in model.graph.node
    )


def _runtime_input(session: ort.InferenceSession) -> tuple[str, np.ndarray]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise OptimizationValidationError(
            f"Runtime verification requires one public input, found {[value.name for value in inputs]!r}."
        )
    value = inputs[0]
    if any(not isinstance(dimension, int) or dimension <= 0 for dimension in value.shape):
        raise OptimizationValidationError(
            f"Runtime verification requires a static input shape, found {value.shape!r}."
        )
    shape = tuple(value.shape)
    element_count = int(np.prod(shape, dtype=np.int64))
    if value.type == "tensor(uint8)":
        data = (np.arange(element_count, dtype=np.uint32) % 251).astype(np.uint8).reshape(shape)
    elif value.type == "tensor(float)":
        data = ((np.arange(element_count, dtype=np.uint32) % 251).astype(np.float32)).reshape(shape)
    else:
        raise OptimizationValidationError(
            f"Runtime verification does not support public input dtype {value.type!r}."
        )
    return value.name, data


def _runtime_verify(source_path: Path, optimized_path: Path) -> list[dict[str, Any]]:
    if VERIFY_PROVIDER not in ort.get_available_providers():
        raise OptimizationConfigurationError(
            f"VERIFY_PROVIDER={VERIFY_PROVIDER!r} is unavailable; "
            f"available={ort.get_available_providers()!r}."
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    source_session = ort.InferenceSession(
        str(source_path),
        sess_options=options,
        providers=[VERIFY_PROVIDER],
    )
    optimized_session = ort.InferenceSession(
        str(optimized_path),
        sess_options=options,
        providers=[VERIFY_PROVIDER],
    )
    input_name, input_data = _runtime_input(source_session)
    optimized_input = optimized_session.get_inputs()[0]
    if optimized_input.name != input_name or tuple(optimized_input.shape) != input_data.shape:
        raise OptimizationValidationError("Optimized runtime input contract differs from the source model.")
    expected_outputs = source_session.run(None, {input_name: input_data})
    actual_outputs = optimized_session.run(None, {input_name: input_data})
    if len(expected_outputs) != len(actual_outputs):
        raise OptimizationValidationError(
            f"Runtime output count changed from {len(expected_outputs)} to {len(actual_outputs)}."
        )

    report: list[dict[str, Any]] = []
    output_names = [value.name for value in optimized_session.get_outputs()]
    for name, expected, actual in zip(output_names, expected_outputs, actual_outputs):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise OptimizationValidationError(
                f"Output {name!r} contract changed: source={expected.shape}/{expected.dtype}, "
                f"optimized={actual.shape}/{actual.dtype}."
            )
        if np.issubdtype(actual.dtype, np.floating):
            if not np.isfinite(actual).all():
                raise OptimizationValidationError(f"Output {name!r} contains NaN or Inf values.")
            expected_float = expected.astype(np.float32, copy=False)
            actual_float = actual.astype(np.float32, copy=False)
            residual = actual_float - expected_float
            rmse = float(np.sqrt(np.mean(residual * residual, dtype=np.float64)))
            reference_rms = float(
                np.sqrt(np.mean(expected_float * expected_float, dtype=np.float64))
            )
            nrmse = rmse / max(reference_rms, 1e-12)
            dot = float(np.sum(expected_float * actual_float, dtype=np.float64))
            expected_norm = float(np.sqrt(np.sum(expected_float * expected_float, dtype=np.float64)))
            actual_norm = float(np.sqrt(np.sum(actual_float * actual_float, dtype=np.float64)))
            cosine = dot / max(expected_norm * actual_norm, 1e-12)
            report.append(
                {
                    "name": name,
                    "shape": list(actual.shape),
                    "dtype": str(actual.dtype),
                    "nrmse": nrmse,
                    "cosine": cosine,
                    "max_abs_error": float(np.max(np.abs(residual), initial=0.0)),
                }
            )
        else:
            mismatch = float(np.mean(expected != actual))
            report.append(
                {
                    "name": name,
                    "shape": list(actual.shape),
                    "dtype": str(actual.dtype),
                    "mismatch_fraction": mismatch,
                }
            )
    return report


def optimize_onnx(config: OptimizeConfig) -> dict[str, Any]:
    validate_config(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    source_model = onnx.load(str(config.source_path), load_external_data=True)
    onnx.checker.check_model(source_model)
    original_interface = _interface_signature(source_model)
    original_size = config.source_path.stat().st_size
    quantized_convs = 0
    stats = RefineStats()
    method_metadata = {"optimization.method": config.method}
    if config.method in {"Q4", "Q8", "DYNAMIC"}:
        quantized_convs, stats = quantize_affine_v2_model(source_model, config)
        method_metadata.update({
            "quantization.method": config.method,
            "quantization.algorithm": WEIGHT_ONLY_ALGORITHM,
            "quantization.block_size": str(
                config.block_size if config.method in {"Q4", "Q8"} else "per-output-channel"
            ),
            "quantization.symmetric": str(
                config.symmetric if config.method in {"Q4", "Q8"} else config.dynamic_symmetric
            ),
            "quantization.conv_count": str(quantized_convs),
        })
    elif config.method == "F16":
        method_metadata.update({
            "precision.method": "F16",
            "precision.keep_io_types": "True",
        })
    else:
        method_metadata["precision.method"] = "F32"
    _update_metadata(source_model, method_metadata)

    with tempfile.TemporaryDirectory(
        prefix=f".{config.output_path.stem}.",
        dir=config.output_path.parent,
    ) as temporary_directory:
        work_path = Path(temporary_directory) / config.output_path.name
        onnx.save(source_model, str(work_path), save_as_external_data=False)
        del source_model
        gc.collect()

        if RUN_ONNXSLIM:
            print("  onnxslim pass 1...")
            _run_onnxslim(work_path)
        if RUN_TRANSFORMERS_OPTIMIZER:
            print("  Transformers optimizer...")
            _run_transformers_optimizer(work_path)
        if RUN_ONNXSLIM:
            print("  onnxslim pass 2...")
            _run_onnxslim(work_path)

        final_model = onnx.load(str(work_path), load_external_data=False)
        if config.method == "F16":
            final_model = _convert_model_to_float16(final_model)
        onnx.checker.check_model(final_model)
        final_interface = _interface_signature(final_model)
        if final_interface != original_interface:
            raise OptimizationValidationError(
                f"Graph interface changed during optimization: source={original_interface!r}, "
                f"optimized={final_interface!r}."
            )
        if config.method in {"Q4", "Q8", "DYNAMIC"}:
            coverage = _coverage(final_model, config.method)
            if coverage != quantized_convs:
                raise OptimizationValidationError(
                    f"Optimization retained {coverage}/{quantized_convs} "
                    f"{config.method} Conv rewrites."
                )
        elif config.method == "F16" and not any(
            initializer.data_type == TensorProto.FLOAT16
            for initializer in final_model.graph.initializer
        ):
            raise OptimizationValidationError(
                "Optimization removed all float16 initializers from the F16 model."
            )
        _update_metadata(
            final_model,
            {
                "optimization.pipeline": "onnxslim -> transformers optimizer -> onnxslim",
                "optimization.optimizer_level": str(OPTIMIZER_LEVEL),
            },
        )
        onnx.save(final_model, str(work_path), save_as_external_data=False)
        del final_model
        gc.collect()

        runtime_report = (
            _runtime_verify(config.source_path, work_path)
            if config.verify_with_onnxruntime
            else None
        )
        optimized_size = work_path.stat().st_size
        os.replace(work_path, config.output_path)

    ratio = stats.refined_error / stats.seed_error if stats.seed_error else 1.0
    report = {
        "source": str(config.source_path.resolve()),
        "output": str(config.output_path.resolve()),
        "method": config.method,
        "algorithm": (
            WEIGHT_ONLY_ALGORITHM
            if config.method in {"Q4", "Q8", "DYNAMIC"}
            else "none"
        ),
        "quantized_convs": quantized_convs,
        "blocks": stats.blocks,
        "improved_blocks": stats.improved_blocks,
        "plain_mse_ratio": ratio,
        "source_bytes": original_size,
        "optimized_bytes": optimized_size,
        "size_ratio": optimized_size / original_size,
        "runtime_validation": runtime_report or "not-requested",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _synthetic_conv_model() -> onnx.ModelProto:
    generator = np.random.default_rng(20260822)
    weight = generator.normal(0.0, 0.2, size=(4, 3, 3, 3)).astype(np.float32)
    bias = generator.normal(0.0, 0.05, size=(4,)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["images", "weight", "bias"],
                ["output0"],
                name="synthetic_conv",
            )
        ],
        "synthetic_yolo_conv",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, 6, 6])],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    model.metadata_props.add(key="model_family", value="ultralytics")
    model.metadata_props.add(key="model_task", value="detect")
    return model


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="yolo-optimize-self-test-") as directory:
        root = Path(directory)
        source_path = root / "source.onnx"
        onnx.save(_synthetic_conv_model(), str(source_path))
        for method in ("Q4", "Q8", "DYNAMIC", "F16", "F32"):
            config = OptimizeConfig(
                source_path=source_path,
                output_path=root / f"optimized_{method.lower()}.onnx",
                method=method,
                block_size=16,
                symmetric=False,
                dynamic_symmetric=True,
                nodes_to_include=None,
                nodes_to_exclude=None,
                overwrite_output=True,
                verify_with_onnxruntime=True,
            )
            report = optimize_onnx(config)
            expected_quantized_convs = 1 if method in {"Q4", "Q8", "DYNAMIC"} else 0
            assert report["quantized_convs"] == expected_quantized_convs
            assert config.output_path.is_file()
            optimized = onnx.load(str(config.output_path), load_external_data=False)
            if expected_quantized_convs:
                assert _coverage(optimized, method) == 1
            elif method == "F16":
                assert any(
                    initializer.data_type == TensorProto.FLOAT16
                    for initializer in optimized.graph.initializer
                )
            else:
                assert not any(
                    initializer.data_type == TensorProto.FLOAT16
                    for initializer in optimized.graph.initializer
                )
    print(
        "Self-test passed: Q4, Q8, and DYNAMIC AFFINE_REFINE_V2 rewrites; F16 conversion; "
        "F32 pass-through; onnxslim; Transformers optimization; ONNX checker; and ORT "
        "execution are operational."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic optimizer checks.")
    parser.add_argument("--input", type=Path, help="Override ORIGINAL_MODEL_PATH.")
    parser.add_argument("--output", type=Path, help="Override OPTIMIZED_MODEL_PATH.")
    parser.add_argument(
        "--method",
        choices=("Q4", "Q8", "DYNAMIC", "F16", "F32"),
        help="Override QUANT_METHOD.",
    )
    parser.add_argument(
        "--no-runtime-check",
        action="store_true",
        help="Skip ONNX Runtime execution after optimization.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    config = OptimizeConfig.from_user_configuration()
    if args.input is not None:
        config = replace(config, source_path=args.input)
        if args.output is None and OPTIMIZED_MODEL_PATH is None:
            config = replace(
                config,
                output_path=args.input.with_name(
                    f"{args.input.stem}_{config.method.lower()}.onnx"
                ),
            )
    if args.method is not None:
        config = replace(config, method=args.method)
        if args.output is None and OPTIMIZED_MODEL_PATH is None:
            config = replace(
                config,
                output_path=config.source_path.with_name(
                    f"{config.source_path.stem}_{args.method.lower()}.onnx"
                ),
            )
    if args.output is not None:
        config = replace(config, output_path=args.output)
    if args.no_runtime_check:
        config = replace(config, verify_with_onnxruntime=False)
    optimize_onnx(config)


if __name__ == "__main__":
    main()

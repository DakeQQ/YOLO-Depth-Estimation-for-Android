from typing import Optional

import torch
from torch import Tensor

__all__ = ["batch_distance2bbox"]


def batch_distance2bbox(points: Tensor, distance: Tensor, max_shapes: Optional[Tensor] = None) -> Tensor:
    """Decode distance prediction to bounding box for batch.

    :param points: [B, ..., 2], "xy" format
    :param distance: [B, ..., 4], "ltrb" format
    :param max_shapes: [B, 2], "h,w" format, Shape of the image.
    :return: Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, dim=-1)
    return torch.cat([points.unsqueeze(0) + 0.5 * (rb-lt), rb+lt], dim=-1)

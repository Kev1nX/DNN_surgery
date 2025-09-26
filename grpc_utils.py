"""Shared gRPC utility helpers for tensor serialization.

This module centralizes PyTorch tensor <-> protobuf conversions so that both the
client and server reuse the same, well-tested logic. Consolidating the helpers
here reduces duplication and keeps cross-cutting concerns (like device moves or
buffer handling) in a single place.
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Union

import torch

from gRPC.protobuf import dnn_inference_pb2

__all__ = ["tensor_to_proto", "proto_to_tensor"]

logger = logging.getLogger(__name__)


def tensor_to_proto(
    tensor: torch.Tensor,
    *,
    ensure_cpu: bool = False,
    detach: bool = True,
) -> dnn_inference_pb2.Tensor:
    """Convert a ``torch.Tensor`` into a ``dnn_inference_pb2.Tensor`` message.

    Args:
        tensor: The tensor to serialize.
        ensure_cpu: If True, move the tensor to CPU memory before serialization.
        detach: If True, detach the tensor from the autograd graph prior to
            serialization.

    Returns:
        A protobuf message containing the serialized tensor bytes and metadata.
    """
    if detach:
        tensor = tensor.detach()

    if ensure_cpu and tensor.device.type != "cpu":
        tensor = tensor.cpu()

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    tensor_bytes = buffer.getvalue()

    shape = dnn_inference_pb2.TensorShape(dimensions=list(tensor.shape))

    logger.debug(
        "Serialized tensor - shape=%s, dtype=%s, bytes=%d",
        tuple(tensor.shape),
        tensor.dtype,
        len(tensor_bytes),
    )

    return dnn_inference_pb2.Tensor(
        data=tensor_bytes,
        shape=shape,
        dtype=str(tensor.dtype),
        requires_grad=tensor.requires_grad,
    )


def proto_to_tensor(
    proto: dnn_inference_pb2.Tensor,
    *,
    device: Optional[Union[torch.device, str]] = None,
    map_location: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Convert a ``dnn_inference_pb2.Tensor`` back into a ``torch.Tensor``.

    Args:
        proto: The protobuf tensor message to deserialize.
        device: Optional target device to move the tensor to after deserialization.
        map_location: Optional ``map_location`` argument forwarded to ``torch.load``.

    Returns:
        The deserialized PyTorch tensor.
    """
    buffer = io.BytesIO(proto.data)
    tensor = torch.load(buffer, map_location=map_location)

    if device is not None:
        tensor = tensor.to(device)

    logger.debug(
        "Deserialized tensor - shape=%s, dtype=%s, device=%s",
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
    )

    return tensor

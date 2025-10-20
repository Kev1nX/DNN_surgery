"""Shared gRPC utility helpers for tensor serialization.

This module centralizes PyTorch tensor <-> protobuf conversions so that both the
client and server reuse the same, well-tested logic. Consolidating the helpers
here reduces duplication and keeps cross-cutting concerns (like device moves or
buffer handling) in a single place.

Supports optional quantization of intermediate tensors during transfer to reduce
network bandwidth by ~4x (FP32 → INT8).
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
    quantize: bool = False,
) -> dnn_inference_pb2.Tensor:
    """Convert a ``torch.Tensor`` into a ``dnn_inference_pb2.Tensor`` message.

    Args:
        tensor: The tensor to serialize.
        ensure_cpu: If True, move the tensor to CPU memory before serialization.
        detach: If True, detach the tensor from the autograd graph prior to
            serialization.
        quantize: If True, quantize the tensor to INT8 before serialization to
            reduce transfer size by ~4x (only for float tensors).

    Returns:
        A protobuf message containing the serialized tensor bytes and metadata.
    """
    if detach:
        tensor = tensor.detach()

    if ensure_cpu and tensor.device.type != "cpu":
        tensor = tensor.cpu()

    # Apply quantization if requested and tensor is float
    original_dtype = tensor.dtype
    was_quantized = False
    if quantize and tensor.is_floating_point():
        try:
            from quantization import ModelQuantizer
            tensor = ModelQuantizer.quantize_tensor(tensor)
            was_quantized = True
            logger.debug(
                "Quantized tensor for transfer - original dtype=%s, quantized dtype=%s",
                original_dtype,
                tensor.dtype,
            )
        except Exception as e:
            logger.warning("Failed to quantize tensor for transfer: %s. Sending as FP32.", e)

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    tensor_bytes = buffer.getvalue()

    shape = dnn_inference_pb2.TensorShape(dimensions=list(tensor.shape))

    logger.debug(
        "Serialized tensor - shape=%s, dtype=%s, bytes=%d, quantized=%s",
        tuple(tensor.shape),
        tensor.dtype,
        len(tensor_bytes),
        was_quantized,
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
    dequantize: bool = True,
) -> torch.Tensor:
    """Convert a ``dnn_inference_pb2.Tensor`` back into a ``torch.Tensor``.

    Args:
        proto: The protobuf tensor message to deserialize.
        device: Optional target device to move the tensor to after deserialization.
        map_location: Optional ``map_location`` argument forwarded to ``torch.load``.
        dequantize: If True, automatically dequantize INT8 tensors back to FP32.
            This restores the original precision for inference.

    Returns:
        The deserialized PyTorch tensor (dequantized if applicable).
    """
    buffer = io.BytesIO(proto.data)
    tensor = torch.load(buffer, map_location=map_location)

    # Automatically dequantize if tensor is quantized
    was_quantized = tensor.is_quantized if hasattr(tensor, 'is_quantized') else False
    if dequantize and was_quantized:
        original_dtype = tensor.dtype
        tensor = tensor.dequantize()
        logger.debug(
            "Dequantized tensor - original dtype=%s, dequantized dtype=%s",
            original_dtype,
            tensor.dtype,
        )

    if device is not None:
        tensor = tensor.to(device)

    logger.debug(
        "Deserialized tensor - shape=%s, dtype=%s, device=%s, was_quantized=%s",
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        was_quantized,
    )

    return tensor

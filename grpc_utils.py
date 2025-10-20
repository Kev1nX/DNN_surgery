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

    # Store original shape BEFORE quantization (quantized tensors can have shape issues)
    original_shape = list(tensor.shape)
    original_dtype = str(tensor.dtype)

    # Apply quantization if requested and tensor is float
    was_quantized = False
    if quantize and tensor.is_floating_point():
        try:
            from quantization import ModelQuantizer
            # Quantize the tensor - this preserves shape but changes dtype
            tensor = ModelQuantizer.quantize_tensor(tensor)
            was_quantized = True
            logger.debug(
                "Quantized tensor for transfer - shape=%s, original dtype=%s, quantized dtype=%s",
                tuple(original_shape),
                original_dtype,
                tensor.dtype,
            )
        except Exception as e:
            logger.warning("Failed to quantize tensor for transfer: %s. Sending as FP32.", e)
            was_quantized = False

    # Serialize the tensor (quantized or FP32)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    tensor_bytes = buffer.getvalue()

    # Use original shape (before quantization) to avoid shape corruption
    shape = dnn_inference_pb2.TensorShape(dimensions=original_shape)

    logger.debug(
        "Serialized tensor - shape=%s, dtype=%s, bytes=%d, quantized=%s",
        tuple(original_shape),
        tensor.dtype,
        len(tensor_bytes),
        was_quantized,
    )

    return dnn_inference_pb2.Tensor(
        data=tensor_bytes,
        shape=shape,
        dtype=original_dtype if was_quantized else str(tensor.dtype),
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

    # Check if tensor is quantized and needs dequantization
    was_quantized = hasattr(tensor, 'is_quantized') and tensor.is_quantized
    
    if dequantize and was_quantized:
        original_dtype = tensor.dtype
        tensor = tensor.dequantize()
        logger.debug(
            "Dequantized tensor - original dtype=%s, dequantized dtype=%s, shape=%s",
            original_dtype,
            tensor.dtype,
            tuple(tensor.shape),
        )
        
        # Verify shape matches proto metadata
        expected_shape = tuple(proto.shape.dimensions)
        if tuple(tensor.shape) != expected_shape:
            logger.warning(
                "Shape mismatch after dequantization! Expected %s but got %s. "
                "This may indicate quantization corruption.",
                expected_shape,
                tuple(tensor.shape),
            )
            # Try to reshape if dimensions match
            if tensor.numel() == torch.prod(torch.tensor(expected_shape)).item():
                logger.info("Reshaping tensor from %s to %s", tuple(tensor.shape), expected_shape)
                tensor = tensor.reshape(expected_shape)
            else:
                logger.error(
                    "Cannot reshape - element count mismatch: %d vs %d",
                    tensor.numel(),
                    torch.prod(torch.tensor(expected_shape)).item(),
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

"""Central configuration values shared across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class GrpcSettings:
    """Encapsulate gRPC channel/server sizing defaults.

    Attributes:
        max_message_mb: Maximum message size in MiB allowed for both sending and
            receiving. Defaults to 50MB which safely covers ImageNet tensors and
            intermediate activations.
    """

    max_message_mb: int = 10

    @property
    def max_message_bytes(self) -> int:
        return self.max_message_mb * 1024 * 1024

    @property
    def channel_options(self) -> List[Tuple[str, int]]:
        """Return options suitable when creating gRPC channels or servers."""
        return [
            ("grpc.max_receive_message_length", self.max_message_bytes),
            ("grpc.max_send_message_length", self.max_message_bytes),
        ]


GRPC_SETTINGS = GrpcSettings()

__all__ = ["GRPC_SETTINGS", "GrpcSettings"]

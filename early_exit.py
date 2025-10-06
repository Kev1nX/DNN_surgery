"""Optional early-exit inference utilities for DNN_surgery.

This module introduces an alternative inference client that can terminate
processing early on the edge device when intermediate classifiers are
confident enough. The original `DNNInferenceClient` remains untouched so the
baseline behaviour is preserved.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import grpc
import torch
import torch.nn as nn
import torch.nn.functional as F

import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
from dnn_inference_client import DNNInferenceClient
from dnn_surgery import DNNSurgery
from grpc_utils import proto_to_tensor, tensor_to_proto

logger = logging.getLogger(__name__)


@dataclass
class EarlyExitConfig:
    """Configuration for optional early exits.

    Attributes:
        enabled: Master switch for early exits. When False, behaviour falls
            back to the standard pipeline.
        confidence_threshold: Global softmax confidence threshold that must be
            exceeded to trigger an exit.
        exit_points: Optional explicit list of edge-layer indices where exit
            heads should be attached. If omitted, residual blocks are detected
            automatically.
        per_layer_thresholds: Optional custom thresholds per exit layer index.
        max_exits: Optional cap on the number of exit heads to attach (starting
            from shallower layers).
        head_hidden_dim: Optional hidden dimension for exit heads; when None a
            single linear layer is used on top of global average pooled
            features.
        head_state_dicts: Optional mapping from exit layer index to a PyTorch
            state_dict file that contains pre-trained head weights.
    """

    enabled: bool = True
    confidence_threshold: float = 0.9
    exit_points: Optional[List[int]] = None
    per_layer_thresholds: Dict[int, float] = field(default_factory=dict)
    max_exits: Optional[int] = None
    head_hidden_dim: Optional[int] = None
    head_state_dicts: Dict[int, str] = field(default_factory=dict)


@dataclass
class EarlyExitDecision:
    """Summary of an early-exit decision for a single sample."""

    triggered: bool
    layer_index: Optional[int] = None
    confidence: float = 0.0
    prediction: Optional[int] = None
    logits: Optional[torch.Tensor] = None


def _is_residual_block(module: nn.Module) -> bool:
    """Heuristically determine whether a module represents a residual block."""

    if not isinstance(module, nn.Sequential):
        return False

    child_names = {child.__class__.__name__ for child in module.children()}
    return any("BasicBlock" in name or "Bottleneck" in name for name in child_names)


class EarlyExitHead(nn.Module):
    """A lightweight classifier head attached to an intermediate activation."""

    def __init__(self, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self._initialized = False
        self.pool: Optional[nn.AdaptiveAvgPool2d] = None
        self.classifier: Optional[nn.Sequential] = None

    def _initialize(self, x: torch.Tensor) -> None:
        if x.dim() > 2:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = x.shape[1]
        else:
            self.pool = None
            in_features = x.shape[1]

        layers: List[nn.Module] = []
        if self.hidden_dim:
            layers.extend(
                [
                    nn.Linear(in_features, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(self.hidden_dim, self.num_classes),
                ]
            )
        else:
            layers.append(nn.Linear(in_features, self.num_classes))

        self.classifier = nn.Sequential(*layers)
        self._initialized = True
        self.register_module("classifier", self.classifier)
        if self.pool is not None:
            self.register_module("pool", self.pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._initialize(x)

        if self.pool is not None:
            x = self.pool(x)

        if x.dim() > 2:
            x = torch.flatten(x, 1)

        return self.classifier(x)


class EarlyExitInferenceClient(DNNInferenceClient):
    """Inference client with optional early-exit support."""

    def __init__(
        self,
        server_address: str,
        dnn_surgery: DNNSurgery,
        edge_model: Optional[nn.Module] = None,
        exit_config: Optional[EarlyExitConfig] = None,
        max_inflight_requests: Optional[int] = None,
    ):
        edge_model = edge_model if edge_model is not None else dnn_surgery.splitter.get_edge_model()
        super().__init__(server_address, edge_model=edge_model, max_inflight_requests=max_inflight_requests)
        self.exit_config = exit_config or EarlyExitConfig()
        self.base_model = dnn_surgery.model
        self.exit_history: List[EarlyExitDecision] = []

        if not self.exit_config.enabled:
            self.exit_points: List[int] = []
            self.exit_heads: Dict[int, EarlyExitHead] = {}
            self.num_classes = self._infer_num_classes()
            return

        self.num_classes = self._infer_num_classes()
        self.exit_points = self._determine_exit_points()
        self.exit_heads: Dict[int, EarlyExitHead] = {
            idx: EarlyExitHead(self.num_classes, hidden_dim=self.exit_config.head_hidden_dim).eval()
            for idx in self.exit_points
        }
        self.exit_thresholds: Dict[int, float] = {
            idx: self.exit_config.per_layer_thresholds.get(idx, self.exit_config.confidence_threshold)
            for idx in self.exit_points
        }
        self._maybe_load_head_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_num_classes(self) -> int:
        if hasattr(self.base_model, "fc") and isinstance(self.base_model.fc, nn.Linear):
            return self.base_model.fc.out_features

        classifier = getattr(self.base_model, "classifier", None)
        if isinstance(classifier, nn.Linear):
            return classifier.out_features
        if isinstance(classifier, nn.Sequential):
            for module in reversed(classifier):
                if isinstance(module, nn.Linear):
                    return module.out_features

        raise ValueError("Unable to infer number of classes from model")

    def _determine_exit_points(self) -> List[int]:
        if self.exit_config.exit_points is not None:
            explicit = [idx for idx in self.exit_config.exit_points if self.edge_model and idx < len(self.edge_model.layers)]
            if self.exit_config.max_exits is not None:
                return explicit[: self.exit_config.max_exits]
            return explicit

        if self.edge_model is None or not hasattr(self.edge_model, "layers"):
            return []

        candidate_indices = [
            idx for idx, layer in enumerate(self.edge_model.layers) if _is_residual_block(layer)
        ]

        if self.exit_config.max_exits is not None:
            candidate_indices = candidate_indices[: self.exit_config.max_exits]

        return candidate_indices

    def _maybe_load_head_weights(self) -> None:
        if not self.exit_config.head_state_dicts:
            return

        for idx, path in self.exit_config.head_state_dicts.items():
            if idx not in self.exit_heads:
                logger.warning("Skipping state dict for undefined exit head at index %s", idx)
                continue
            if not os.path.exists(path):
                logger.warning("State dict path does not exist for head %s: %s", idx, path)
                continue
            try:
                state_dict = torch.load(path, map_location="cpu")
                self.exit_heads[idx].load_state_dict(state_dict)
                logger.info("Loaded pre-trained head weights for exit %s from %s", idx, path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to load head weights for exit %s: %s", idx, exc)

    def _forward_edge_without_exits(self, sample: torch.Tensor) -> Tuple[torch.Tensor, EarlyExitDecision, float]:
        if self.edge_model is None:
            return sample, EarlyExitDecision(triggered=False), 0.0

        with torch.no_grad():
            edge_start = time.perf_counter()
            output = self.edge_model(sample)
            edge_time = (time.perf_counter() - edge_start) * 1000

        decision = EarlyExitDecision(triggered=False)
        return output, decision, edge_time

    def _forward_edge_with_exits(self, sample: torch.Tensor) -> Tuple[torch.Tensor, EarlyExitDecision, float]:
        if self.edge_model is None or not self.exit_heads:
            return self._forward_edge_without_exits(sample)

        activation = sample
        with torch.no_grad():
            edge_start = time.perf_counter()
            for idx, layer in enumerate(self.edge_model.layers):
                if self.edge_model._needs_flattening(layer, activation):  # type: ignore[attr-defined]
                    activation = torch.flatten(activation, 1)
                activation = layer(activation)

                if idx not in self.exit_heads:
                    continue

                head = self.exit_heads[idx].to(activation.device)
                logits = head(activation)
                probabilities = F.softmax(logits, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
                confidence_value = confidence.item()
                threshold = self.exit_thresholds.get(idx, self.exit_config.confidence_threshold)

                logger.debug(
                    "Exit candidate at layer %s: confidence=%.4f threshold=%.4f", idx, confidence_value, threshold
                )

                if confidence_value >= threshold:
                    edge_time = (time.perf_counter() - edge_start) * 1000
                    decision = EarlyExitDecision(
                        triggered=True,
                        layer_index=idx,
                        confidence=confidence_value,
                        prediction=prediction.item(),
                        logits=logits.detach(),
                    )
                    self.exit_history.append(decision)
                    logger.info(
                        "Early exit triggered at layer %s with confidence %.3f (prediction=%s)",
                        idx,
                        confidence_value,
                        prediction.item(),
                    )
                    return logits.detach(), decision, edge_time

            edge_time = (time.perf_counter() - edge_start) * 1000

        decision = EarlyExitDecision(triggered=False)
        return activation, decision, edge_time

    def _finalize_remote_request(
        self,
        idx: int,
        future: grpc.Future,
        send_time: float,
        sample_metrics: Dict[str, float],
        aggregated_timings: Dict[str, float],
        results: List[Optional[torch.Tensor]],
        batch_size: int,
        model_id: str,
    ) -> None:
        try:
            response = future.result()
        except grpc.RpcError as rpc_error:  # pylint: disable=no-member
            logger.error("gRPC error: %s: %s", rpc_error.code(), rpc_error.details())
            raise

        recv_time = time.perf_counter()
        total_time = (recv_time - send_time) * 1000

        if not response.success:
            logger.error("Server processing failed: %s", response.error_message)
            raise RuntimeError(f"Server error: {response.error_message}")

        result_tensor = proto_to_tensor(response.tensor)
        if torch.isnan(result_tensor).any():
            raise ValueError("Received tensor contains NaN values")
        if torch.isinf(result_tensor).any():
            raise ValueError("Received tensor contains infinite values")

        transfer_time = total_time * 0.2
        cloud_time = total_time - transfer_time

        sample_metrics['transfer_time'] = transfer_time
        sample_metrics['cloud_time'] = cloud_time

        aggregated_timings['transfer_time'] += transfer_time
        aggregated_timings['cloud_time'] += cloud_time

        self.transfer_times.append(transfer_time)
        self.cloud_times.append(cloud_time)

        logger.info("Cloud processing time: %.2fms", cloud_time)
        logger.info("Transfer time: %.2fms", transfer_time)
        logger.info("Total time: %.2fms", total_time)
        logger.info("Output tensor stats - Shape: %s", result_tensor.shape)

        if result_tensor.dim() == 2:
            probs = torch.softmax(result_tensor, dim=1)
            max_prob, pred = probs.max(1)
            logger.info(
                "Prediction confidence: %.3f, Predicted class: %s",
                max_prob.item(),
                pred.item(),
            )

        logger.info("Successfully processed tensor %s/%s with model %s", idx + 1, batch_size, model_id)
        results[idx] = result_tensor

    def process_tensor(
        self,
        tensor: torch.Tensor,
        model_id: str,
        requires_cloud_processing: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.exit_config.enabled or not self.exit_heads:
            return super().process_tensor(tensor, model_id, requires_cloud_processing)

        batch_size = tensor.shape[0]
        is_batch = batch_size > 1

        aggregated_timings = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}
        results: List[Optional[torch.Tensor]] = [None] * batch_size
        pending_requests: Deque[Tuple[int, grpc.Future, float, Dict[str, float]]] = deque()
        max_concurrency = max(1, min(batch_size, self.max_inflight_requests))
        current_exit_count = 0

        if is_batch:
            logger.info(
                "Processing batched tensor of shape %s (batch_size=%s) with model %s using early exits",
                tensor.shape,
                batch_size,
                model_id,
            )

        try:
            for idx in range(batch_size):
                sample = tensor if not is_batch else tensor[idx:idx + 1]
                sample_metrics = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}

                logger.info("Processing sample %s/%s with shape %s", idx + 1, batch_size, sample.shape)
                logger.info(
                    "Input tensor stats - Min: %.3f, Max: %.3f, Mean: %.3f",
                    sample.min().item(),
                    sample.max().item(),
                    sample.mean().item(),
                )

                processed_sample, decision, edge_time = self._forward_edge_with_exits(sample)
                sample_metrics['edge_time'] = edge_time
                aggregated_timings['edge_time'] += edge_time
                if edge_time:
                    self.edge_times.append(edge_time)

                if decision.triggered:
                    current_exit_count += 1
                    results[idx] = decision.logits
                    # No transfer/cloud times when exiting on the edge
                    continue

                if not requires_cloud_processing:
                    results[idx] = processed_sample
                    continue

                request = dnn_inference_pb2.InferenceRequest(
                    tensor=tensor_to_proto(processed_sample, ensure_cpu=True),
                    model_id=model_id,
                )

                send_time = time.perf_counter()
                future = self.stub.ProcessTensor.future(request)
                pending_requests.append((idx, future, send_time, sample_metrics))

                if len(pending_requests) >= max_concurrency:
                    idx_r, future_r, send_time_r, metrics_r = pending_requests.popleft()
                    self._finalize_remote_request(
                        idx_r,
                        future_r,
                        send_time_r,
                        metrics_r,
                        aggregated_timings,
                        results,
                        batch_size,
                        model_id,
                    )

            while pending_requests:
                idx_r, future_r, send_time_r, metrics_r = pending_requests.popleft()
                self._finalize_remote_request(
                    idx_r,
                    future_r,
                    send_time_r,
                    metrics_r,
                    aggregated_timings,
                    results,
                    batch_size,
                    model_id,
                )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error processing tensor(s) with early exits: %s", exc)
            raise

        if not is_batch:
            assert results[0] is not None
            summary = aggregated_timings.copy()
            summary['early_exit_triggered'] = current_exit_count
            return results[0], summary

        batched_result = torch.cat(results, dim=0)  # type: ignore[arg-type]
        avg_timings = {key: (value / batch_size) for key, value in aggregated_timings.items()}
        avg_timings['total_batch_processing_time'] = sum(aggregated_timings.values())
        avg_timings['batch_size'] = batch_size
        avg_timings['early_exit_count'] = float(current_exit_count)
        avg_timings['early_exit_rate'] = current_exit_count / batch_size if batch_size else 0.0

        logger.info(
            "Batch processing with early exits completed. Total time: %.2fms, Average per sample: %.2fms",
            avg_timings['total_batch_processing_time'],
            avg_timings['total_batch_processing_time'] / batch_size if batch_size else 0.0,
        )

        return batched_result, avg_timings

    def get_exit_statistics(self) -> Dict[str, float]:
        triggered = [decision for decision in self.exit_history if decision.triggered]
        if not triggered:
            return {'total_exits': 0}

        per_layer = Counter(decision.layer_index for decision in triggered)
        total = len(triggered)
        most_common_layer, most_common_count = per_layer.most_common(1)[0]
        return {
            'total_exits': float(total),
            'most_frequent_exit_layer': float(most_common_layer if most_common_layer is not None else -1),
            'most_frequent_exit_fraction': most_common_count / total,
        }


def run_distributed_inference_with_early_exit(
    model_id: str,
    input_tensor: torch.Tensor,
    dnn_surgery: DNNSurgery,
    exit_config: Optional[EarlyExitConfig] = None,
    split_point: Optional[int] = None,
    server_address: str = 'localhost:50051',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run distributed inference with optional early exits on the edge."""

    exit_config = exit_config or EarlyExitConfig()

    if split_point is None:
        optimal_split, analysis = dnn_surgery.find_optimal_split(input_tensor, server_address)
        split_point = optimal_split
        logger.info("NeuroSurgeon optimal split point: %s", split_point)
        logger.info("Predicted total time: %.2fms", analysis['min_total_time'])
    else:
        logger.info("Using manual split point: %s", split_point)
        success = dnn_surgery._send_split_decision_to_server(split_point, server_address)  # pylint: disable=protected-access
        if not success:
            logger.error("Failed to send manual split decision to server")

    dnn_surgery.splitter.set_split_point(split_point)
    edge_model = dnn_surgery.splitter.get_edge_model()
    cloud_model = dnn_surgery.splitter.get_cloud_model()
    requires_cloud_processing = cloud_model is not None

    if not requires_cloud_processing:
        logger.info("Split point %s means all inference on client side - server communication optional", split_point)

    client = EarlyExitInferenceClient(
        server_address=server_address,
        dnn_surgery=dnn_surgery,
        edge_model=edge_model,
        exit_config=exit_config,
    )

    result, timings = client.process_tensor(input_tensor, model_id, requires_cloud_processing)
    timing_summary = client.get_timing_summary()
    timings.update(timing_summary)
    timings['split_point'] = split_point

    if exit_config.enabled:
        timings.update(client.get_exit_statistics())

    return result, timings

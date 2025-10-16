
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
    confidence_threshold: float = 0.7  # Lower threshold - 0% confidence to exit
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
    """Early exit head that reuses the model's classifier.
    
    Simple architecture:
    1. Global Average Pooling - spatial reduction (if needed)
    2. Flatten
    3. Optional adapter layer - project features to match classifier input dimension
    4. Reused pretrained classifier head
    
    This is lightweight and leverages the pretrained classifier directly.
    """

    def __init__(
        self, 
        num_classes: int,
        pretrained_classifier: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_classifier = pretrained_classifier
        self._initialized = False
        
        # Architecture components (minimal)
        self.pool: Optional[nn.AdaptiveAvgPool2d] = None
        self.adapter: Optional[nn.Linear] = None
        self.fc_classifier: Optional[nn.Linear] = None

    def _initialize(self, x: torch.Tensor) -> None:
        device = x.device
        
        # Add pooling if features are spatial (4D)
        if x.dim() > 2:
            self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            in_features = x.shape[1]
        else:
            self.pool = None
            in_features = x.shape[1]
        
        # If we have pretrained classifier, use it directly or with adapter
        if self.pretrained_classifier is not None:
            target_dim = self.pretrained_classifier.in_features
            
            # Add adapter if intermediate features don't match classifier input
            if in_features != target_dim:
                self.adapter = nn.Linear(in_features, target_dim).to(device)
                logger.info(f"Added adapter layer: {in_features} -> {target_dim} (to match pretrained classifier)")
            else:
                self.adapter = None
                logger.info(f"Using pretrained classifier directly (dimensions match: {in_features})")
            
            # Use the pretrained classifier
            self.fc_classifier = self.pretrained_classifier
        else:
            # No pretrained classifier - create new one
            self.adapter = None
            self.fc_classifier = nn.Linear(in_features, self.num_classes).to(device)
            logger.warning(f"No pretrained classifier available - created new classifier: {in_features} -> {self.num_classes}")
        
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        if not self._initialized:
            self._initialize(x)

        # Global average pooling (if spatial features)
        if self.pool is not None:
            x = self.pool(x)

        # Flatten to 2D
        if x.dim() > 2:
            x = torch.flatten(x, 1)

        # Apply adapter if present
        if self.adapter is not None:
            x = self.adapter(x)

        # Return classifier logits
        x = self.fc_classifier(x)

        return x


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
        
        # Get pretrained classifier to reuse in exit heads
        pretrained_classifier = None
        if hasattr(self.base_model, 'fc') and isinstance(self.base_model.fc, nn.Linear):
            pretrained_classifier = self.base_model.fc
            logger.info(f"Using pretrained classifier from base model: {pretrained_classifier.in_features} -> {pretrained_classifier.out_features}")
        
        # Create exit heads that reuse the pretrained classifier
        self.exit_heads: Dict[int, EarlyExitHead] = {
            idx: EarlyExitHead(self.num_classes, pretrained_classifier=pretrained_classifier).eval()
            for idx in self.exit_points
        }
        self.exit_thresholds: Dict[int, float] = {
            idx: self.exit_config.per_layer_thresholds.get(idx, self.exit_config.confidence_threshold)
            for idx in self.exit_points
        }
        logger.info(f"Exit thresholds: {self.exit_thresholds}")
        self._load_head_weights()

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
            logger.info(f"Using explicit exit points: {explicit}")
            return explicit

        if self.edge_model is None or not hasattr(self.edge_model, "layers"):
            return []

        num_layers = len(self.edge_model.layers)
        logger.info(f"Edge model has {num_layers} layers, determining exit points...")

        # Try to find residual blocks first
        candidate_indices = [
            idx for idx, layer in enumerate(self.edge_model.layers) if _is_residual_block(layer)
        ]
        
        # If no residual blocks found, use evenly spaced layers as fallback
        if not candidate_indices:
            max_exits = self.exit_config.max_exits or 3
            # Place exits at 25%, 50%, 75% of the network
            if num_layers == 0:
                logger.warning(f"Edge model has 0 layers - cannot place exit points!")
                return []
            
            spacing = max(1, num_layers // (max_exits + 1))
            candidate_indices = [spacing * (i + 1) for i in range(max_exits) if spacing * (i + 1) < num_layers]

        if self.exit_config.max_exits is not None:
            candidate_indices = candidate_indices[: self.exit_config.max_exits]
            logger.info(f"Limited to {self.exit_config.max_exits} exits: {candidate_indices}")

        return candidate_indices

    def _load_head_weights(self) -> None:
        """Load trained weights from checkpoints if available.
        
        If no checkpoints exist, exit heads already use the pretrained classifier by default
        (passed during construction), so no further initialization is needed.
        """
        for idx in self.exit_heads:
            exit_head = self.exit_heads[idx]
            checkpoint_path = self.exit_config.head_state_dicts.get(idx)
            
            # Try to load from checkpoint if path is provided and exists
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    # Use strict=False to handle dimension mismatches gracefully
                    # This can happen when the same exit index has different feature dimensions
                    # at different split points (e.g., exit at layer 5 might be 128ch or 256ch)
                    missing_keys, unexpected_keys = exit_head.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys or unexpected_keys:
                        logger.warning("Checkpoint for exit %s has mismatched dimensions - will use pretrained classifier instead", idx)
                        logger.debug("Missing keys: %s", missing_keys)
                        logger.debug("Unexpected keys: %s", unexpected_keys)
                    else:
                        logger.info("✓ Loaded trained weights for exit %s from %s", idx, checkpoint_path)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Failed to load checkpoint for exit %s: %s - will use pretrained classifier", idx, exc)
            else:
                # No checkpoint - exit head already uses pretrained classifier by default
                logger.info("✓ Exit %s uses pretrained model classifier (no checkpoint provided)", idx)

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

                # Log every exit check with INFO level to see what's happening
                logger.info(
                    "Exit candidate at layer %s: confidence=%.4f threshold=%.4f prediction=%s", 
                    idx, confidence_value, threshold, prediction.item()
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

        # Get actual server timing from response (matches standard client behavior)
        server_exec_time = getattr(response, 'server_execution_time_ms', 0.0)
        server_total_time = getattr(response, 'server_total_time_ms', 0.0)

        # Debug logging for timing breakdown
        logger.debug(f"[TIMING DEBUG] Sample {idx + 1}/{batch_size}:")
        logger.debug(f"  Total round-trip time: {total_time:.2f}ms")
        logger.debug(f"  Server exec time (from response): {server_exec_time:.2f}ms")
        logger.debug(f"  Server total time (from response): {server_total_time:.2f}ms")

        # Calculate transfer time as the difference between total round-trip and server-reported time
        transfer_time = max(total_time - server_total_time, 0.0) if server_total_time > 0 else max(total_time - server_exec_time, 0.0)
        cloud_time = max(server_exec_time, 0.0)
        
        logger.debug(f"  Calculated transfer time: {transfer_time:.2f}ms")
        logger.debug(f"  Calculated cloud time: {cloud_time:.2f}ms")

        sample_metrics['transfer_time'] = transfer_time
        sample_metrics['cloud_time'] = cloud_time

        aggregated_timings['transfer_time'] += transfer_time
        aggregated_timings['cloud_time'] += cloud_time

        self.transfer_times.append(transfer_time)
        self.cloud_times.append(cloud_time)

        logger.info("Cloud processing time (server reported): %.2fms", cloud_time)
        logger.info("Transfer time (measured): %.2fms", transfer_time)
        if server_total_time > 0:
            logger.info("Server total handling time: %.2fms", server_total_time)
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
        if not self.exit_config.enabled:
            return super().process_tensor(tensor, model_id, requires_cloud_processing)
        
        if not self.exit_heads:
            logger.warning("Early exit enabled but no exit heads available - falling back to standard processing")
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
                    logger.info(
                        "✓ Sample %d/%d exited early at layer %d with %.1f%% confidence (prediction: %d)",
                        idx + 1,
                        batch_size,
                        decision.layer_index if decision.layer_index is not None else -1,
                        decision.confidence * 100,
                        decision.prediction if decision.prediction is not None else -1,
                    )
                    # No transfer/cloud times when exiting on the edge
                    continue

                if not requires_cloud_processing:
                    results[idx] = processed_sample
                    logger.info("✗ Sample %d/%d did NOT exit early - all processing on edge (no cloud model)", idx + 1, batch_size)
                    continue
                
                logger.info("✗ Sample %d/%d did NOT exit early - sending to cloud for processing", idx + 1, batch_size)

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
            summary['total_batch_processing_time'] = sum(aggregated_timings.values())  # Add total time for plotting
            summary['early_exit_count'] = float(current_exit_count)
            summary['early_exit_rate'] = float(current_exit_count)  # For single sample, rate is 1.0 if exited, 0.0 if not
            logger.info(
                "Single sample processing with early exits completed. Total time: %.2fms, Early exit: %s",
                summary['total_batch_processing_time'],
                "Yes" if current_exit_count > 0 else "No"
            )
            return results[0], summary

        batched_result = torch.cat(results, dim=0)  # type: ignore[arg-type]
        
        # Calculate averages correctly accounting for early exits
        # For transfer/cloud: only divide by samples that actually used cloud
        non_exit_count = batch_size - current_exit_count
        
        avg_timings = {}
        avg_timings['edge_time'] = aggregated_timings['edge_time'] / batch_size if batch_size > 0 else 0.0
        
        # Only average transfer/cloud over samples that actually went to cloud
        if non_exit_count > 0:
            avg_timings['transfer_time'] = aggregated_timings['transfer_time'] / non_exit_count
            avg_timings['cloud_time'] = aggregated_timings['cloud_time'] / non_exit_count
        else:
            # All samples exited early
            avg_timings['transfer_time'] = 0.0
            avg_timings['cloud_time'] = 0.0
        
        # Total batch time is the sum of all actual times
        avg_timings['total_batch_processing_time'] = sum(aggregated_timings.values())
        avg_timings['batch_size'] = batch_size
        avg_timings['early_exit_count'] = float(current_exit_count)
        avg_timings['early_exit_rate'] = current_exit_count / batch_size if batch_size else 0.0
        avg_timings['non_exit_count'] = non_exit_count

        logger.info(
            "Batch processing with early exits completed. Total time: %.2fms, Average per sample: %.2fms, "
            "Early exits: %d/%d (%.1f%%)",
            avg_timings['total_batch_processing_time'],
            avg_timings['total_batch_processing_time'] / batch_size if batch_size else 0.0,
            current_exit_count,
            batch_size,
            avg_timings['early_exit_rate'] * 100,
        )
        
        logger.info(
            "Timing breakdown (for non-exited samples): Edge=%.2fms, Transfer=%.2fms, Cloud=%.2fms",
            avg_timings['edge_time'],
            avg_timings['transfer_time'] if non_exit_count > 0 else 0.0,
            avg_timings['cloud_time'] if non_exit_count > 0 else 0.0,
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


def find_optimal_split_with_early_exit(
    dnn_surgery: DNNSurgery,
    input_tensor: torch.Tensor,
    server_address: str,
    exit_config: EarlyExitConfig,
) -> Tuple[int, Dict]:
    """Find optimal split point for baseline inference (without early exits).
    
    Early exits are used opportunistically during inference, but the split point
    is optimized for the case when confidence is NOT high enough to exit early.
    This ensures good performance regardless of whether early exits trigger.
    
    Important: Split point must be at least 1 so there are edge layers for exit heads.

    Args:
        dnn_surgery: DNNSurgery instance with the model to split.
        input_tensor: Input tensor for profiling.
        server_address: Server address for distributed inference.
        exit_config: Early exit configuration (not used for split selection).

    Returns:
        Tuple of (optimal_split_point, timing_analysis).
    """
    from dnn_inference_client import DNNInferenceClient
    
    num_splits = len(dnn_surgery.splitter.layers) + 1
    logger.info("=== Finding Optimal Split Point (Baseline without Early Exits) ===")
    logger.info("Early exits will be used opportunistically at runtime, but split point optimized for baseline case")
    
    # CRITICAL: Skip split point 0 - early exits require edge layers
    min_split = 1
    logger.info(f"Testing {num_splits - min_split} split points ({min_split} to {num_splits - 1}) - skipping split point 0 (no edge layers for exit heads)")
    
    split_analysis = {}
    
    # Profile each valid split point WITHOUT early exits
    for split_point in range(min_split, num_splits):
        logger.info(f"Testing split point {split_point}/{num_splits - 1} (baseline)...")
        
        dnn_surgery.splitter.set_split_point(split_point)
        edge_model = dnn_surgery.splitter.get_edge_model(
            quantize=dnn_surgery.enable_quantization,
            quantizer=dnn_surgery.quantizer,
        )
        
        # Use standard client WITHOUT early exits
        client = DNNInferenceClient(server_address, edge_model)
        requires_cloud = dnn_surgery.splitter.get_cloud_model() is not None
        dnn_surgery._send_split_decision_to_server(split_point, server_address)  # pylint: disable=protected-access
        
        try:
            _, timings = client.process_tensor(input_tensor, dnn_surgery.model_name, requires_cloud)
            
            edge_time = timings.get('edge_time', 0.0)
            transfer_time = timings.get('transfer_time', 0.0)
            cloud_time = timings.get('cloud_time', 0.0)
            total_time = edge_time + transfer_time + cloud_time
            
            split_analysis[split_point] = {
                'edge_time': edge_time,
                'transfer_time': transfer_time,
                'cloud_time': cloud_time,
                'total_time': total_time,
            }
            
            logger.info(f"  Split {split_point}: Total={total_time:.1f}ms (Edge={edge_time:.1f}ms, "
                       f"Transfer={transfer_time:.1f}ms, Cloud={cloud_time:.1f}ms)")
            
        except Exception as exc:
            logger.error(f"Failed to test split point {split_point}: {exc}")
            split_analysis[split_point] = {
                'edge_time': 0.0,
                'transfer_time': 0.0,
                'cloud_time': 0.0,
                'total_time': float('inf'),
            }
    
    # Find optimal split among valid candidates
    optimal_split = min(split_analysis.keys(), key=lambda k: split_analysis[k]['total_time'])
    min_time = split_analysis[optimal_split]['total_time']
    
    logger.info("=== Baseline Optimal Split Point Found ===")
    logger.info(f"Optimal split: {optimal_split} with baseline time: {min_time:.1f}ms")
    logger.info("Early exits will be attempted at runtime as an optimization")
    
    # Configure server with optimal split
    dnn_surgery._send_split_decision_to_server(optimal_split, server_address)  # pylint: disable=protected-access
    
    from visualization import build_split_timing_summary, format_split_summary
    
    split_summary = build_split_timing_summary(split_analysis, dnn_surgery.get_split_layer_names())
    
    return optimal_split, {
        'optimal_split': optimal_split,
        'min_total_time': min_time,
        'all_splits': split_analysis,
        'split_summary': split_summary,
        'split_summary_table': format_split_summary(split_summary, sort_by_total_time=False),
        'layer_names': dnn_surgery.get_split_layer_names(),
        'split_config_success': True,
    }


def run_distributed_inference_with_early_exit(
    model_id: str,
    input_tensor: torch.Tensor,
    dnn_surgery: DNNSurgery,
    exit_config: Optional[EarlyExitConfig] = None,
    split_point: Optional[int] = None,
    server_address: str = 'localhost:50051',
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run distributed inference with optional early exits on the edge.
    
    Args:
        model_id: Model identifier
        input_tensor: Input tensor
        dnn_surgery: DNNSurgery instance for model splitting
        exit_config: Early exit configuration
        split_point: Optional manual split point (if None, will use NeuroSurgeon optimization)
        server_address: Server address
        auto_plot: Whether to automatically generate plots (default: True)
        plot_show: Whether to display plots interactively (default: True)
        plot_path: Optional path to save plots (default: None, uses "plots/" directory)
        
    Returns:
        Tuple of (result tensor, timing dictionary)
    """

    exit_config = exit_config or EarlyExitConfig()

    if split_point is None:
        if exit_config.enabled:
            # Use early exit profiling when early exit is enabled (regardless of explicit exit points)
            logger.info("Early exit enabled - using early exit profiling to find optimal split")
            optimal_split, analysis = find_optimal_split_with_early_exit(
                dnn_surgery, input_tensor, server_address, exit_config
            )
            split_point = optimal_split
            logger.info("NeuroSurgeon optimal split point (with early exits): %s", split_point)
            logger.info("Predicted total time: %.2fms", analysis['min_total_time'])
        else:
            # Standard profiling without early exits
            logger.info("Early exit disabled - using standard profiling")
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

    # Generate plot if auto_plot is enabled
    if auto_plot:
        try:
            from dnn_inference_client import resolve_plot_paths
            from visualization import plot_actual_inference_breakdown
            
            _, actual_plot_path = resolve_plot_paths(model_id, split_point, plot_path)
            # Add earlyexit suffix to distinguish from normal inference
            actual_plot_path = actual_plot_path.with_name(actual_plot_path.stem + "_earlyexit" + actual_plot_path.suffix)
            actual_plot_path.parent.mkdir(parents=True, exist_ok=True)
            
            total_metrics = {
                'edge_time': timings.get('edge_time', 0),
                'transfer_time': timings.get('transfer_time', 0),
                'cloud_time': timings.get('cloud_time', 0),
                'total_batch_processing_time': timings.get('total_batch_processing_time', 0),
            }
            
            plot_actual_inference_breakdown(
                total_metrics,
                show=plot_show,
                save_path=str(actual_plot_path),
                title=f"Measured Inference Breakdown - Early Exit ({model_id})"
            )
            timings['actual_split_plot_path'] = str(actual_plot_path.resolve())
            logger.info(f"Early exit plot saved to {actual_plot_path}")
        except ImportError as plt_error:
            logger.warning(f"Auto-plot requested but dependencies unavailable: {plt_error}")
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

    return result, timings

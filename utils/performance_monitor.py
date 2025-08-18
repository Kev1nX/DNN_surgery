import time
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Container for inference performance metrics.
    
    Attributes:
        total_time_ms: Total time for complete inference pipeline
        preprocessing_time_ms: Time taken for data preprocessing
        inference_time_ms: Time for model inference only
        postprocessing_time_ms: Time for output postprocessing
        data_transfer_size_bytes: Size of data transferred between edge and cloud
    """
    total_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    data_transfer_size_bytes: int
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary format with derived calculations."""
        return {
            'total_time_ms': self.total_time_ms,
            'preprocessing_time_ms': self.preprocessing_time_ms,
            'inference_time_ms': self.inference_time_ms,
            'postprocessing_time_ms': self.postprocessing_time_ms,
            'data_transfer_size_bytes': self.data_transfer_size_bytes,
            'data_transfer_size_mb': self.data_transfer_size_bytes / (1024 * 1024),
            'throughput_samples_per_sec': 1000 / self.total_time_ms if self.total_time_ms > 0 else 0,
            'data_transfer_rate_mbps': (self.data_transfer_size_bytes * 8) / (self.total_time_ms * 1000)
        }

class PerformanceMonitor:
    """Monitor and log inference performance metrics.
    
    Tracks timing and data transfer metrics for model inference operations.
    Provides functionality to record metrics, calculate statistics, and generate reports.
    """
    
    def __init__(self, log_dir: str = 'performance_logs'):
        """Initialize the performance monitor.
        
        Args:
            log_dir: Directory path for saving performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics: List[InferenceMetrics] = []
        
    
    def record_inference(self, metrics: InferenceMetrics):
        """Record metrics from an inference"""
        self.metrics.append(metrics)
        
    def get_summary_stats(self) -> Dict[str, dict]:
        """Calculate summary statistics of recorded metrics.
        
        Computes statistics including:
        - Mean, median, std deviation
        - Min and max values
        - Throughput metrics
        - Data transfer rates
        
        Returns:
            Dictionary containing statistics for each metric type
        """
        if not self.metrics:
            logger.warning("No metrics available for summary statistics")
            return {}
            
        # Basic metrics to track
        metrics_dict = {
            'total_time_ms': [],
            'preprocessing_time_ms': [],
            'inference_time_ms': [],
            'postprocessing_time_ms': [],
            'data_transfer_size_bytes': [],
        }
        
        # Derived metrics
        derived_metrics = {
            'throughput_samples_per_sec': [],
            'data_transfer_rate_mbps': []
        }
        
        # Collect metrics
        for m in self.metrics:
            # Basic metrics
            for key in metrics_dict.keys():
                metrics_dict[key].append(getattr(m, key))
            
            # Calculate derived metrics
            if m.total_time_ms > 0:
                derived_metrics['throughput_samples_per_sec'].append(1000 / m.total_time_ms)
                derived_metrics['data_transfer_rate_mbps'].append(
                    (m.data_transfer_size_bytes * 8) / (m.total_time_ms * 1000)
                )
        
        # Combine all metrics
        metrics_dict.update(derived_metrics)
        
        # Calculate statistics
        summary = {}
        for key, values in metrics_dict.items():
            if values:  # Only calculate if we have values
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
            
        return summary
    
    def save_metrics(self, prefix: str = ''):
        """Save metrics to JSON file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = self.log_dir / f'{prefix}metrics_{timestamp}.json'
        
        data = {
            'individual_metrics': [m.to_dict() for m in self.metrics],
            'summary_stats': self.get_summary_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            
        return filename
    
    def print_summary(self):
        """Print a formatted performance summary to console."""
        summary = self.get_summary_stats()
        if not summary:
            logger.warning("No metrics available for display")
            return
            
        logger.info("\nPerformance Summary Report")
        logger.info("=" * 50)
        
        # Group metrics by category
        timing_metrics = [
            'total_time_ms', 
            'preprocessing_time_ms', 
            'inference_time_ms', 
            'postprocessing_time_ms'
        ]
        
        throughput_metrics = [
            'throughput_samples_per_sec',
            'data_transfer_rate_mbps'
        ]
        
        data_metrics = [
            'data_transfer_size_bytes'
        ]
        
        # Log concise timing summary
        stats = summary.get('total_time_ms', {})
        if stats:
            logger.info("\nTiming Summary:")
            logger.info(f"Average latency: {stats['mean']:.2f} ± {stats['std']:.2f} ms")
            logger.info(f"Min/Max latency: {stats['min']:.2f}/{stats['max']:.2f} ms")
        
        # Log detailed timing metrics at debug level
        logger.debug("\nDetailed Timing Metrics (milliseconds):")
        logger.debug("-" * 30)
        for metric in timing_metrics:
            if metric in summary:
                stats = summary[metric]
                logger.debug(f"\n{metric.replace('_', ' ').title()}:")
                logger.debug(f"  Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
                logger.debug(f"  Range: [{stats['min']:.2f} - {stats['max']:.2f}]")
        
        # Log throughput summary
        if 'throughput_samples_per_sec' in summary:
            stats = summary['throughput_samples_per_sec']
            logger.info(f"\nThroughput: {stats['mean']:.2f} ± {stats['std']:.2f} inferences/second")
            logger.info(f"Peak throughput: {stats['max']:.2f} inferences/second")
        
        # Log data transfer summary
        if 'data_transfer_rate_mbps' in summary:
            stats = summary['data_transfer_rate_mbps']
            logger.info(f"\nData Transfer Rate: {stats['mean']:.2f} ± {stats['std']:.2f} Mbps")
            
        if 'data_transfer_size_bytes' in summary:
            stats = summary['data_transfer_size_bytes']
            mean_mb = stats['mean'] / (1024 * 1024)
            logger.info(f"Average Transfer Size: {mean_mb:.2f} MB per inference")
            
        # Log total statistics
        sample_count = summary[list(summary.keys())[0]]['count']
        logger.info(f"\nTotal Samples: {sample_count}")
        logger.info(f"Detailed metrics saved to: {self.log_dir}")
        
        # Log detailed metrics at debug level
        logger.debug("\nDetailed Throughput Metrics:")
        logger.debug("-" * 30)
        for metric in throughput_metrics:
            if metric in summary:
                stats = summary[metric]
                logger.debug(f"\n{metric.replace('_', ' ').title()}:")
                logger.debug(f"  Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
                logger.debug(f"  Range: [{stats['min']:.2f} - {stats['max']:.2f}]")
        
        logger.debug("\nDetailed Data Transfer Metrics:")
        logger.debug("-" * 30)
        for metric in data_metrics:
            if metric in summary:
                stats = summary[metric]
                mean_mb = stats['mean'] / (1024 * 1024)
                std_mb = stats['std'] / (1024 * 1024)
                min_mb = stats['min'] / (1024 * 1024)
                max_mb = stats['max'] / (1024 * 1024)
                logger.debug(f"\n{metric.replace('_', ' ').title()}:")
                logger.debug(f"  Mean ± Std: {mean_mb:.2f}MB ± {std_mb:.2f}MB")
                logger.debug(f"  Range: [{min_mb:.2f}MB - {max_mb:.2f}MB]")

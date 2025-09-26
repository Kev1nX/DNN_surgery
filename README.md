# DNN Surgery - Comprehensive Deep Neural Network Splitting System

DNN Surgery is an intelligent system that optimizes Deep Neural Network (DNN) inference across edge devices and cloud endpoints by dynamically determining optimal split points based on network conditions and hardware capabilities.

## Key Features

- **Intelligent Model Splitting**: Automatically finds optimal split points based on network bandwidth and hardware performance
- **Multi-Architecture Support**: Compatible with ResNet18, AlexNet, CNN, and custom models
- **Distributed Inference**: Client-server architecture for edge-cloud computing scenarios
- **Performance Profiling**: Comprehensive layer-wise execution and transfer metrics
- **Network Condition Simulation**: Realistic network condition modeling with bandwidth, latency, and packet loss
- **Flexible Deployment**: Supports standalone analysis, distributed server, and edge client modes

## Architecture

The system consists of three main operational modes:

1. **Analyze Mode**: Standalone analysis of models to find optimal split points
2. **Server Mode**: GPU/cloud server that handles the compute-intensive layers
3. **Client Mode**: Edge device (e.g., Raspberry Pi) that processes initial layers

## Quick Start

### Standalone Analysis
Analyze a model to find optimal split points for different network conditions:

```bash
python dnn_surgery.py --mode analyze --model resnet18 --bandwidth-range 4 8 12 16
```

### Distributed Setup

1. **Start the server** (on GPU/cloud machine):
```bash
python dnn_surgery.py --mode server --model resnet18 --port 50051 --bandwidth 10
```

2. **Run the client** (on edge device):
```bash
python dnn_surgery.py --mode client --server-address <server-ip>:50051 --test-samples 10
```

## Command Line Options

### Common Options
- `--mode {analyze,server,client}`: Operation mode
- `--model {resnet18,alexnet,cnn}`: Model architecture to use
- `--device {auto,cpu,cuda}`: Compute device (auto-detects by default)
- `--use-pretrained/--no-pretrained`: Use pretrained weights

### Analysis Mode
- `--bandwidth-range`: List of bandwidths to test (e.g., `--bandwidth-range 4 8 12 16`)
- `--save-results/--no-save-results`: Save detailed results to files

### Server Mode
- `--port`: Server port (default: 50051)
- `--bandwidth`: Expected client bandwidth in Mbps

### Client Mode
- `--server-address`: Server address (e.g., `192.168.1.100:50051`)
- `--test-samples`: Number of inference samples to test
- `--config-file`: Custom configuration file path

## Project Structure

```
DNN_surgery/
├── dnn_surgery.py           # Main consolidated system
├── server.py               # gRPC server implementation
├── dnn_inference_client.py # gRPC client implementation
├── config.py               # Shared configuration (e.g., gRPC sizing)
├── grpc_utils.py           # Tensor serialization helpers for gRPC
├── rpi_profiler.py         # Raspberry Pi hardware profiling
├── networks/               # Model architectures
│   ├── resnet18.py
│   ├── alexnet.py
│   └── cnn.py
├── gRPC/                   # Communication protocols
│   ├── protobuf/          # Generated protobuf code
│   └── protos/            # Protocol definitions
├── utils/                  # Utility modules
├── dataset/               # Data handling
└── performance_logs/      # Analysis results
```

## Example Output

### Analysis Mode Results
```
================================================================================
DNN SURGERY ANALYSIS RESULTS
================================================================================
Model: resnet18
Device: cuda
Total Layers: 11

LAYER-WISE EXECUTION METRICS:
--------------------------------------------------------------------------------
Layer  Name                 Time(ms)   Memory(KB)   Transfer(KB)
--------------------------------------------------------------------------------
0      Conv2d_0             0.45       256.0        3136.0
1      BatchNorm2d_1        0.12       0.0          3136.0
...

OPTIMAL SPLIT POINTS:
--------------------------------------------------------------------------------
BW(Mbps)   Split Layer  Total(ms)    Client(ms)   Server(ms)   Transfer(ms)
--------------------------------------------------------------------------------
4.0        2            156.78       0.67         12.45        143.66
8.0        4            87.23        2.34         12.45        72.44
12.0       6            65.89        5.67         12.45        47.77
16.0       8            58.12        8.94         12.45        36.73
```

## Advanced Features

### Client-Side Profiling
Run hardware-specific profiling for Raspberry Pi devices:

```bash
# Profile hardware capabilities
python rpi_profiler.py --model resnet18 --samples 10

# Use profiling data in distributed setup
python dnn_surgery.py --mode client --profile-first --client-id rpi_01
```

### Custom Network Conditions
The system can simulate various network conditions including:
- Bandwidth variations (1-100+ Mbps)
- Network latency (10-500+ ms)
- Packet loss (0-10%)
- Jitter effects

### Results Storage
Analysis results are automatically saved to `performance_logs/` with timestamps:
- `{model}_layer_metrics_{timestamp}.json`: Layer-wise performance data
- `{model}_optimal_splits_{timestamp}.json`: Optimal split configurations
- `{model}_report_{timestamp}.txt`: Summary report

## Model Support

### Supported Architectures
- **ResNet18**: 18-layer residual network
- **AlexNet**: Classic CNN architecture
- **CNN**: Custom lightweight CNN

### Adding Custom Models
1. Create model class in `networks/` directory
2. Implement `gen_network()` method to return layer list
3. Add model configuration to `ModelManager.MODELS`

## Performance Considerations

### Hardware Requirements
- **Server**: CUDA-compatible GPU recommended for optimal performance
- **Client**: ARM-based devices (Raspberry Pi) or x86 systems
- **Network**: Stable connection with measurable bandwidth

### Optimization Tips
- Use CUDA when available for server operations
- Profile client hardware for accurate split decisions
- Monitor network conditions for dynamic adaptation
- Cache split configurations for consistent setups

## Contributing

The system is designed for extensibility:
- Add new model architectures in `networks/`
- Extend profiling capabilities in `utils/`
- Enhance communication protocols in `gRPC/`
- Improve analysis algorithms in core `dnn_surgery.py`

## License

This project is part of ongoing research in edge-cloud computing optimization for deep learning inference.
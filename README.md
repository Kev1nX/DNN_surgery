# DNN surgery
With the physical hardware limitations of edge devices. Performing DNN inference on the network edge can be expensive. Meanwhile, the high network requirement for DNN inference using a cloud endpoint can induce large latency into the DNN inference process. DNN surgery aim to compute and determine how much of the inference can be computed by edge devices to reduce the amount of data to send to a more powerful server or cloud endpoint.



## Project Overview

The project is structured in several components:

1. **Model Management**
   - Pretrained model loading and configuration
   - Model surgery and splitting algorithms
   - Graph-based analysis for optimal split points

2. **Communication Framework**
   - gRPC-based client-server architecture
   - Efficient tensor serialization
   - Robust error handling

3. **Performance Monitoring**
   - Comprehensive timing metrics
   - Data transfer tracking
   - Accuracy evaluation

4. **Testing Framework**
   - Support for multiple model architectures
   - ImageNet validation
   - Performance profiling

## Project Structure

```
DNN_surgery/
├── gRPC/                 # gRPC communication components
│   ├── protobuf/        # Generated protobuf code
│   └── protos/          # Protocol definitions
├── utils/               # Utility functions
│   └── performance_monitor.py
├── networks/            # Model definitions
├── dataset/            # Data handling
├── dnn_surgery.py      # Main surgery logic
├── server.py           # Server implementation
└── test_local_deployment.py  # Testing framework
```

# Testing Models

The project includes a testing framework to evaluate different pretrained models using ImageNet validation data. You can test various models to measure their inference performance, accuracy, and data transfer characteristics.

## Model

Currently testing with:
- ResNet18 (pretrained on ImageNet)

## Running Tests

1. Start the server:

```bash
python test_local_deployment.py --mode server --model resnet18
```

2. In another terminal, run the client:

```bash
python test_local_deployment.py --mode client --model resnet18 --num-inferences 50
```

## Test Parameters

- `--mode`: Choose between 'server' or 'client'
- `--model`: Currently only supports 'resnet18'
- `--port`: Specify port number (default: 50051)
- `--num-inferences`: Number of test inferences to run (default: 50)
- `--split-point`: Layer index to split model for edge inference (-1 for no split)

## Test Metrics

The tests measure and report:
1. Inference timing (preprocessing, inference, postprocessing)
2. Data transfer sizes (input and output tensors)
3. Model accuracy on ImageNet classes
4. Throughput metrics

Results are saved in:
- `performance_logs/cloud/`: Server-side metrics
- `performance_logs/edge/`: Client-side metrics

## Dataset

The test framework uses ImageNet validation data. If not available, it will automatically create sample test images. For best results:
1. Download ImageNet validation set
2. Place it in `data/imagenet/val/`
3. Ensure images are organized by class in subdirectories
## System configuration

- Windows 10
- AMD Ryzen 5 2600 Six-Core 3.40 GHz
- vCPU = 4, vRAM = 8 gb

## Task description

Simple implementation of MNIST images classification.

## Model Repository Tree

```
└───model_repository
    └───cnn_onnx
        └───1
```

## Throughput and Latency

### Before optimization

- Concurrency: 1, throughput: 1335.71 infer/sec, latency 747 usec
- Concurrency: 2, throughput: 2019.1 infer/sec, latency 989 usec
- Concurrency: 3, throughput: 2008.35 infer/sec, latency 1492 usec
- Concurrency: 4, throughput: 2076.59 infer/sec, latency 1925 usec
- Concurrency: 5, throughput: 2051.42 infer/sec, latency 2436 usec

### After optimization

- Concurrency: 1, throughput: 1036.41 infer/sec, latency 964 usec
- Concurrency: 2, throughput: 1897.74 infer/sec, latency 1053 usec
- Concurrency: 3, throughput: 2576.94 infer/sec, latency 1163 usec
- Concurrency: 4, throughput: 3072.45 infer/sec, latency 1301 usec
- Concurrency: 5, throughput: 3566.17 infer/sec, latency 1401 usec

## Optimization choice explanation

All parameters were selected in such a way as to increase throughput and decrease latency.<br>
Adding multiple instances of the model does not improve performance, dynamic batching is enough.<br>
"max_queue_delay_microseconds" does not increase productivity much (1-5%).

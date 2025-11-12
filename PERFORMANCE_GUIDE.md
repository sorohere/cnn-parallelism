# Performance Analysis Guide

## Overview

This project now includes comprehensive performance metrics beyond simple execution time, allowing you to gain deep insights into the behavior and efficiency of different parallelization strategies.

## Performance Dimensions Tracked

### 1. **Execution Time Breakdown**
- **Total Time**: Complete program execution including all initialization
- **Inference Time**: Pure computation time for CNN inference
- **Model Load Time**: Time to load weights from disk
- **Data Load Time**: Time to load MNIST dataset
- **Communication Time**: MPI overhead (send/recv/wait operations)

### 2. **Throughput & Latency**
- **Throughput**: Images processed per second
- **Average Latency**: Mean time per image (milliseconds)
- **Min/Max Latency**: Best and worst case per-image processing time
- **Latency Variance**: Spread between min and max

### 3. **Layer-wise Timing**
- **Conv1 Time**: First convolutional layer computation
- **Conv2 Time**: Second convolutional layer computation  
- **FC1 Time**: First fully-connected layer computation
- **FC2 Time**: Second fully-connected layer computation
- **Output Time**: Output layer with softmax

### 4. **Parallelization Metrics**
- **Speedup**: Performance improvement vs serial (Serial Time / Parallel Time)
- **Parallel Efficiency**: Speedup normalized by process count (Speedup / N × 100%)
- **Scaling Factor**: How speedup changes with more processes

### 5. **Load Balancing (Data Parallel)**
- **Load Imbalance Factor**: Variance in work distribution across processes
- **Max/Min Process Time**: Longest and shortest process execution
- **Time Variance**: Difference between slowest and fastest process

### 6. **Communication Overhead (MPI)**
- **Communication Percentage**: Time in MPI vs computation
- **Bytes Sent/Received**: Data transfer volume
- **Send/Recv/Wait Time**: Breakdown of MPI operations

### 7. **Memory Efficiency**
- **Peak Memory Usage**: Maximum RSS (Resident Set Size)
- **Memory per Process**: For parallel implementations
- **Memory Overhead**: Increase vs serial baseline

## Running Enhanced Benchmarks

### Quick Start

```bash
# Run detailed benchmark with all metrics
make benchmark_detailed

# Analyze results with insights and recommendations
make analyze
```

### Step-by-Step

```bash
# 1. Setup and train (one-time)
make setup
make train

# 2. Compile all implementations
make compile_all

# 3. Run enhanced benchmark
make benchmark_detailed

# 4. Analyze with Python script
make analyze
```

## Understanding the Output

### Serial Baseline Example

```
========================================================================
  SERIAL INFERENCE - PERFORMANCE SUMMARY
========================================================================
  Execution Metrics:
    Total Time:              5.500 seconds
    Inference Time:          5.277 seconds
    Model Load Time:         0.150 seconds
    Data Load Time:          0.050 seconds

  Layer-wise Timing:
    Conv1 Layer:             1.850 seconds (35.1%)
    Conv2 Layer:             1.650 seconds (31.3%)
    FC1 Layer:               0.890 seconds (16.9%)
    FC2 Layer:               0.687 seconds (13.0%)
    Output Layer:            0.200 seconds (3.8%)

  Throughput & Latency:
    Throughput:              1894.96 images/second
    Avg Latency per Image:   0.528 ms
    Min Latency:             0.485 ms
    Max Latency:             0.612 ms

  Memory Usage:
    Peak Memory:             45.2 MB

  Accuracy:
    Correct Predictions:     9732 / 10000
    Accuracy:                97.32%
========================================================================
```

### Data Parallel Example (4 Processes)

```
========================================================================
  DATA PARALLEL INFERENCE - PERFORMANCE SUMMARY
========================================================================
  Execution Metrics:
    Total Time:              1.500 seconds
    Inference Time:          1.453 seconds
    Communication Time:      0.025 seconds (1.7%)

  Throughput & Latency:
    Throughput:              6881.65 images/second
    Avg Latency per Image:   0.145 ms

  Memory Usage:
    Peak Memory:             52.8 MB

  Parallelization Metrics:
    Number of Processes:     4
    Speedup:                 3.63x
    Parallel Efficiency:     90.8%

Load Balancing Analysis:
  Max Process Time:        1.453 seconds
  Min Process Time:        1.448 seconds
  Time Variance:           0.005 seconds
  Load Imbalance Factor:   0.34%

  ✓ Excellent load balance (< 5% imbalance)
========================================================================
```

## Interpreting Metrics

### Speedup & Efficiency

| Efficiency | Rating | Interpretation |
|------------|--------|----------------|
| > 90% | Excellent | Near-perfect scaling |
| 75-90% | Good | Solid parallelization |
| 60-75% | Fair | Moderate overhead |
| < 60% | Poor | Significant bottlenecks |

### Load Imbalance

| Imbalance | Rating | Action |
|-----------|--------|--------|
| < 5% | Excellent | No action needed |
| 5-15% | Good | Monitor for larger datasets |
| > 15% | Poor | Rebalance workload distribution |

### Communication Overhead

| Overhead | Rating | Interpretation |
|----------|--------|----------------|
| < 5% | Excellent | Minimal MPI impact |
| 5-15% | Good | Acceptable for parallel benefit |
| 15-30% | Fair | Consider optimization |
| > 30% | Poor | Communication dominates |

## Performance Comparison Table

The analyzer generates a comprehensive comparison:

```
====================================================================================================
                            PERFORMANCE METRICS SUMMARY
====================================================================================================
Implementation            Processes  Time(s)    Throughput      Speedup    Efficiency   Memory(MB)
----------------------------------------------------------------------------------------------------
Serial (Baseline)         1          5.277      1894.96 img/s   1.00x      100.0%       45.2
Data Parallel (2P)        2          2.783      3593.49 img/s   1.90x      94.8%        48.5
Data Parallel (4P)        4          1.453      6881.65 img/s   3.63x      90.8%        52.8
Data Parallel (8P)        8          0.979      10211.09 img/s  5.39x      67.4%        65.1
Pipeline (5P)             5          3.340      2994.01 img/s   1.58x      31.6%        47.8
====================================================================================================
```

## Key Insights

### Data Parallel Performance

**Strengths:**
- ✓ Excellent scaling up to 4 processes (90%+ efficiency)
- ✓ Minimal communication overhead (< 2%)
- ✓ Perfect load balance (< 1% imbalance)
- ✓ Best for inference workloads

**Limitations:**
- ✗ Efficiency drops at 8+ processes (memory bandwidth saturation)
- ✗ Requires full model replication (higher memory)
- ✗ Limited by Amdahl's Law (serial portions)

### Pipeline Parallel Performance

**Strengths:**
- ✓ Memory efficient (single model copy)
- ✓ Can handle very large models
- ✓ Good for training workflows

**Limitations:**
- ✗ High communication overhead (layer-to-layer transfers)
- ✗ Sequential dependencies limit parallelism
- ✗ Low efficiency for inference (31.6%)
- ✗ Better suited for training than inference

## Optimization Recommendations

### For Data Parallel

1. **Process Count**: Use 4 processes for optimal efficiency/throughput balance
2. **Memory**: Monitor per-process memory for large models
3. **CPU Binding**: Pin processes to specific cores with `mpirun --bind-to core`
4. **Network**: Use shared memory for single-node setups

### For Pipeline Parallel

1. **Use Case**: Reserve for training or very large models that don't fit in memory
2. **Communication**: Optimize with non-blocking MPI operations
3. **Batching**: Process multiple images per pipeline stage
4. **Hybrid**: Combine with data parallel for best results

## Advanced Analysis

### Layer-wise Bottleneck Analysis

The breakdown shows where computation time is spent:

```
Conv layers (Conv1 + Conv2): 66.4% of time
FC layers (FC1 + FC2 + Output): 33.6% of time
```

**Insight**: Convolutional layers dominate. Optimizations should focus here:
- SIMD vectorization for convolutions
- Cache-friendly memory access patterns
- Specialized conv kernels (Winograd, FFT-based)

### Latency Distribution

Min/Max latency reveals variance:
- **Consistent latency**: Good for real-time systems
- **High variance**: Investigate outliers (cache misses, context switches)

### Memory Scaling

Track memory growth with process count:
- **Linear growth**: Expected for data parallel
- **Sub-linear growth**: Good memory sharing
- **Super-linear growth**: Memory fragmentation issues

## Generating Custom Reports

Modify the analyzer for your needs:

```python
# Edit scripts/analyze_performance.py

def custom_metric(self, metrics):
    # Add your custom analysis
    compute_ratio = metrics['conv1_time'] / metrics['inference_time']
    print(f"Convolution ratio: {compute_ratio * 100:.1f}%")
```

## Exporting Results

Results are saved to:
- `results/benchmark_results_detailed.txt` - Full detailed output
- Can be parsed by any tool (Python, R, Excel)

Example parsing in Python:

```python
import re

with open('results/benchmark_results_detailed.txt') as f:
    content = f.read()
    
speedup = re.findall(r'Speedup:\s+([\d.]+)x', content)
print(f"Speedups: {speedup}")
```

## Troubleshooting

### High Communication Overhead

**Symptoms**: Communication time > 15% of inference time

**Solutions**:
- Reduce MPI process count
- Use shared memory (`mpirun --mca btl_sm_use_knem 1`)
- Increase workload per process (larger batches)

### Poor Load Balance

**Symptoms**: Load imbalance > 15%

**Solutions**:
- Verify even data distribution in code
- Check for non-uniform image processing times
- Monitor CPU governor settings

### Low Parallel Efficiency

**Symptoms**: Efficiency < 70%

**Solutions**:
- Profile for serial bottlenecks (Amdahl's Law)
- Reduce synchronization points
- Consider different parallelization strategy

## Research & Publication

### Metrics for Papers

Key metrics to report:
1. **Speedup vs Process Count** (with plot)
2. **Parallel Efficiency** (table)
3. **Strong/Weak Scaling** analysis
4. **Communication Overhead** breakdown
5. **Load Balance Factor**

### Generating Plots

Use the CSV export feature:

```bash
# Export to CSV
python3 scripts/export_csv.py results/benchmark_results_detailed.txt > results.csv

# Plot with your preferred tool (matplotlib, gnuplot, etc.)
```

## Summary

This enhanced benchmarking system provides:

✓ **10+ performance dimensions** tracked automatically
✓ **Detailed breakdowns** of time, memory, communication
✓ **Comparative analysis** across all implementations
✓ **Actionable recommendations** for optimization
✓ **Research-ready metrics** for papers and presentations

For questions or issues, refer to the detailed output in `results/benchmark_results_detailed.txt`.


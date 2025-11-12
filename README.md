# CNN Parallelism: Performance Analysis with MPI

Professional implementation of CNN on MNIST with comprehensive performance analysis comparing Serial, Data Parallel, and Pipeline Parallel execution using MPI.

## Quick Start

```bash
make setup              # Download MNIST dataset
make train              # Train CNN (5-10 min) → 97% accuracy
make benchmark_detailed # Run enhanced benchmarks with detailed metrics
make analyze            # Analyze results with insights
```

## Overview

This project implements a Convolutional Neural Network from scratch in C and provides three execution strategies:

1. **Serial** - Single CPU baseline
2. **Data Parallel (MPI)** - Splits images across N processes
3. **Pipeline Parallel (MPI)** - Splits network layers across 5 processes

### Key Features

✅ **15+ Performance Metrics** tracked automatically  
✅ **Multi-dimensional analysis** beyond just execution time  
✅ **Detailed bottleneck identification**  
✅ **Actionable optimization recommendations**  
✅ **Publication-ready results**

## CNN Architecture

```
Input Layer:    1×28×28 (grayscale MNIST image)
     ↓
Conv1 Layer:    16×14×14 (16 filters, 3×3 kernel, stride=2, padding=1, ReLU)
     ↓
Conv2 Layer:    32×7×7 (32 filters, 3×3 kernel, stride=2, padding=1, ReLU)
     ↓
FC1 Layer:      200 neurons (fully connected, Tanh)
     ↓
FC2 Layer:      200 neurons (fully connected, Tanh)
     ↓
Output Layer:   10 neurons (softmax for digit classification)
```

**Model Size**: ~300K parameters  
**Accuracy**: 97.32% on MNIST test set

## Performance Results

### Benchmark Summary

| Method | Processes | Accuracy | Time | Speedup | Efficiency | Throughput |
|--------|-----------|----------|------|---------|------------|------------|
| Serial | 1 | 97.32% | 5.35s | 1.0× | 100% | 1,871 img/s |
| Data Parallel | 2 | 97.32% | 2.77s | 1.93× | 96.4% | 3,605 img/s |
| Data Parallel | 4 | 97.32% | 1.45s | 3.69× | 92.2% | 6,896 img/s |
| Data Parallel | 8 | 97.32% | 0.98s | 5.44× | 68.0% | 10,181 img/s |
| Pipeline | 5 | 97.32% | 3.42s | 1.56× | 31.2% | 2,994 img/s |

### Key Insights

**Best Configuration**: Data Parallel with 4 processes
- **Speedup**: 3.69× faster than serial
- **Efficiency**: 92.2% (excellent)
- **Communication overhead**: < 2% (minimal)
- **Load balance**: < 1% imbalance (excellent distribution)

**Scaling Analysis**:
- ✓ Excellent scaling up to 4 processes (>90% efficiency)
- ⚠ Efficiency drops to 68% at 8 processes due to memory bandwidth saturation
- ✗ Pipeline parallel has 31.2% efficiency due to high communication overhead

## Performance Dimensions Tracked

### 1. Execution Time Breakdown
- Total time, inference time, model load, data load, communication time

### 2. Throughput & Latency
- Images/second, average latency per image, min/max latency, variance

### 3. Layer-wise Timing
- Time spent in Conv1, Conv2, FC1, FC2, Output layers
- Identifies computational bottlenecks

### 4. Parallelization Metrics
- Speedup (vs serial baseline)
- Parallel efficiency (speedup / processes)
- Scaling factors between configurations

### 5. Load Balancing (Data Parallel)
- Imbalance factor across processes
- Max/min process times
- Work distribution quality

### 6. Communication Overhead (MPI)
- Time in MPI operations vs computation
- Data volume transferred (bytes sent/received)
- MPI send/recv/wait breakdown

### 7. Memory Efficiency
- Peak memory usage (RSS)
- Per-process memory footprint
- Memory overhead vs serial

## Installation

### Prerequisites

**macOS:**
```bash
brew install open-mpi
```

**Linux:**
```bash
sudo apt-get install build-essential libopenmpi-dev openmpi-bin
```

**Verify installation:**
```bash
mpicc --version
mpirun --version
```

### Setup

```bash
git clone <repository-url>
cd cnn-parallelism
make setup    # Downloads MNIST dataset (~50MB)
```

## Usage

### Complete Workflow

```bash
# 1. Train the model (one-time, ~5-10 minutes)
make train
# Output: models/cnn_model.bin with 97.32% accuracy

# 2. Run enhanced benchmarks
make benchmark_detailed
# Output: results/benchmark_results_detailed.txt

# 3. Analyze with insights
make analyze
# Displays: Summary tables, detailed analysis, recommendations
```

### Individual Commands

**Train Model:**
```bash
make train
# Creates models/cnn_model.bin
```

**Run Serial Inference:**
```bash
make serial
./serial_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte
```

**Run Data Parallel (4 processes):**
```bash
make data_parallel
mpirun -np 4 ./data_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte
```

**Run Pipeline Parallel (5 processes):**
```bash
make pipeline_parallel
mpirun -np 5 ./pipeline_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte
```

**Run Standard Benchmark:**
```bash
make benchmark
# Simpler output in results/benchmark_results.txt
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Download MNIST dataset |
| `make train` | Train CNN model |
| `make compile_all` | Compile all inference programs |
| `make serial` | Compile serial inference only |
| `make data_parallel` | Compile data parallel only |
| `make pipeline_parallel` | Compile pipeline parallel only |
| `make benchmark` | Run standard benchmark |
| `make benchmark_detailed` | Run enhanced benchmark with detailed metrics |
| `make analyze` | Analyze benchmark results |
| `make clean` | Remove compiled binaries |
| `make clean_all` | Remove everything (data, models, results) |

## Example Output

### Serial Baseline

```
========================================================================
  SERIAL INFERENCE - PERFORMANCE SUMMARY
========================================================================
  Execution Metrics:
    Total Time:              5.373 seconds
    Inference Time:          5.346 seconds
    Model Load Time:         0.002 seconds
    Data Load Time:          0.005 seconds

  Throughput & Latency:
    Throughput:              1870.61 images/second
    Avg Latency per Image:   0.535 ms
    Min Latency:             0.520 ms
    Max Latency:             7.730 ms

  Memory Usage:
    Peak Memory:             11.77 MB

  Accuracy:
    Correct Predictions:     9732 / 10000
    Accuracy:                97.32%
========================================================================
```

### Data Parallel (4 Processes)

```
========================================================================
  DATA PARALLEL INFERENCE - PERFORMANCE SUMMARY
========================================================================
  Execution Metrics:
    Total Time:              1.462 seconds
    Inference Time:          1.450 seconds
    Communication Time:      0.025 seconds (1.7%)

  Throughput & Latency:
    Throughput:              6895.7 images/second
    Avg Latency per Image:   0.145 ms

  Memory Usage:
    Peak Memory:             22.7 MB

  Parallelization Metrics:
    Number of Processes:     4
    Speedup:                 3.69x
    Parallel Efficiency:     92.2%

Load Balancing Analysis:
  Max Process Time:        1.450 seconds
  Min Process Time:        1.447 seconds
  Time Variance:           0.003 seconds
  Load Imbalance Factor:   0.21%

  ✓ Excellent load balance (< 5% imbalance)
========================================================================
```

### Comparative Analysis

```
====================================================================================================
                              PERFORMANCE METRICS SUMMARY
====================================================================================================
Implementation            Processes  Time(s)    Throughput      Speedup    Efficiency   Memory(MB)
----------------------------------------------------------------------------------------------------
Serial                    1          5.346      1870.6 img/s    1.00x      100.0%       11.8
Data Parallel (2P)        2          2.774      3605.0 img/s    1.93x      96.4%        22.4
Data Parallel (4P)        4          1.450      6895.7 img/s    3.69x      92.2%        22.7
Data Parallel (8P)        8          0.982      10181.0 img/s   5.44x      68.0%        22.7
Pipeline (5P)             5          3.422      2994.0 img/s    1.56x      31.2%        11.9
====================================================================================================

KEY INSIGHTS:
  • Best Data Parallel: 4P with 3.69x speedup (92.2% efficiency)
  • Efficiency drops by 24.2% when scaling from 4P to 8P
  • Pipeline has high communication overhead (not ideal for inference)
```

## File Structure

```
cnn-parallelism/
├── src/                              # Source code
│   ├── cnn.c/h                       # CNN implementation (layers, forward/backward pass)
│   ├── mnist_loader.c/h              # MNIST dataset reader (IDX format)
│   ├── model_io.c/h                  # Binary model serialization
│   ├── performance_metrics.c/h       # Performance tracking library
│   ├── train.c                       # Training program
│   ├── inference_serial.c            # Serial baseline implementation
│   ├── inference_data_parallel.c     # Data parallel with MPI
│   └── inference_pipeline_parallel.c # Pipeline parallel with MPI
├── scripts/
│   ├── run_benchmarks.sh             # Standard benchmark script
│   ├── run_benchmarks_detailed.sh    # Enhanced benchmark with metrics
│   └── analyze_performance.py        # Python analyzer with insights
├── data/                             # MNIST dataset (auto-downloaded)
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── models/                           # Trained models
│   └── cnn_model.bin                 # Binary model file
├── results/                          # Benchmark outputs
│   ├── benchmark_results.txt
│   └── benchmark_results_detailed.txt
├── Makefile                          # Build system
└── README.md                         # This file
```

## Technical Details

### Binary Model Format

- **Magic Number**: `0x434E4E4D` for validation
- **Version**: 1
- **Contents**: Weights and biases for all 6 layers
- **Size**: ~1.2 MB
- **Format**: Binary (fast loading, portable)

### Data Parallel Strategy

**How it works:**
1. Each process loads the full model
2. Dataset split evenly across processes
3. Each process runs inference on its subset
4. Results aggregated with `MPI_Reduce()`

**Advantages:**
- ✓ Minimal communication (only final aggregation)
- ✓ Excellent scaling (near-linear up to 4 processes)
- ✓ Simple implementation

**Limitations:**
- ✗ Each process needs full model copy (higher memory)
- ✗ Scalability limited by memory bandwidth

### Pipeline Parallel Strategy

**How it works:**
1. Model layers distributed across 5 processes
2. Process 0: Input → Conv1
3. Process 1: Conv2
4. Process 2: FC1
5. Process 3: FC2
6. Process 4: Output
7. Data flows through pipeline via MPI send/recv

**Advantages:**
- ✓ Memory efficient (single model copy distributed)
- ✓ Good for very large models

**Limitations:**
- ✗ High communication overhead (layer-to-layer transfers)
- ✗ Sequential dependencies limit parallelism
- ✗ Lower efficiency for inference workloads

## Understanding Efficiency Metrics

### Speedup
**Formula**: `Serial Time / Parallel Time`

**Interpretation**:
- 2.0× = twice as fast
- 4.0× = four times as fast
- Linear speedup: N processes → N× speedup

### Parallel Efficiency
**Formula**: `(Speedup / Number of Processes) × 100%`

**Rating Scale**:
| Efficiency | Rating | Interpretation |
|------------|--------|----------------|
| > 90% | Excellent | Near-perfect scaling |
| 75-90% | Good | Solid parallelization |
| 60-75% | Fair | Moderate overhead |
| < 60% | Poor | Significant bottlenecks |

### Load Imbalance
**Formula**: `(Max Process Time - Min Process Time) / Max Process Time`

**Rating Scale**:
| Imbalance | Rating | Action Required |
|-----------|--------|-----------------|
| < 5% | Excellent | None |
| 5-15% | Good | Monitor |
| > 15% | Poor | Rebalance workload |

### Communication Overhead
**Formula**: `Communication Time / Total Time × 100%`

**Rating Scale**:
| Overhead | Rating | Interpretation |
|----------|--------|----------------|
| < 5% | Excellent | Minimal impact |
| 5-15% | Good | Acceptable |
| 15-30% | Fair | Consider optimization |
| > 30% | Poor | Communication dominates |

## Optimization Recommendations

### For Data Parallel

**Current Best**: 4 processes (3.69× speedup, 92.2% efficiency)

**Recommendations**:
1. Use 4 processes for optimal balance
2. Pin processes to cores: `mpirun --bind-to core -np 4 ...`
3. Monitor memory usage for larger models
4. Consider batch processing for better cache usage

### For Pipeline Parallel

**Current Performance**: 1.56× speedup, 31.2% efficiency

**Why low efficiency?**
- High communication overhead (18.3%)
- Layer-to-layer data transfers are expensive
- Sequential dependencies limit parallelism

**When to use**:
- Very large models that don't fit in single process memory
- Training workloads (better for backpropagation)
- Combined with data parallel (hybrid approach)

**Optimizations**:
- Use non-blocking MPI operations
- Batch multiple images per pipeline stage
- Optimize layer boundaries to minimize transfers

## For Research & Publications

### Metrics for Papers

Report these key metrics:
1. **Speedup vs Process Count** (with plot)
2. **Parallel Efficiency** (table + plot)
3. **Strong Scaling Analysis**
4. **Communication Overhead Breakdown**
5. **Load Balance Factor**
6. **Layer-wise Computation Analysis**

### Example Results Section

```
We evaluated three parallelization strategies on MNIST CNN inference:

1. Data Parallel: Achieved 3.69× speedup with 4 processes (92.2% efficiency)
   - Communication overhead: 1.7%
   - Load imbalance: 0.21%
   
2. Pipeline Parallel: Achieved 1.56× speedup with 5 processes (31.2% efficiency)
   - Communication overhead: 18.3%
   - Limited by sequential dependencies

The data parallel approach demonstrates excellent scaling efficiency up to 4 processes,
with minimal communication overhead (<2%). Beyond 4 processes, efficiency drops to 68%
due to memory bandwidth saturation (Amdahl's Law).
```

### Creating Plots

Export data from results:
```bash
grep "Speedup:" results/benchmark_results_detailed.txt
grep "Efficiency:" results/benchmark_results_detailed.txt
```

Plot with your preferred tool (matplotlib, gnuplot, R, Excel).

## Troubleshooting

### Issue: "Model not found"
**Solution**: Run `make train` first to create the model

### Issue: "Failed to load test images"
**Solution**: Run `make setup` to download MNIST dataset

### Issue: MPI errors
**Solution**: Verify MPI installation with `mpirun --version`

### Issue: Low efficiency at high process counts
**Explanation**: Normal due to Amdahl's Law and memory bandwidth limits

### Issue: Pipeline parallel has low accuracy
**Explanation**: Check that you're using exactly 5 processes (one per stage)

## Advanced Usage

### Custom Process Counts

Data Parallel with custom count:
```bash
mpirun -np 16 ./data_parallel_inference <images> <labels>
```

### CPU Binding for Better Performance

```bash
mpirun --bind-to core -np 4 ./data_parallel_inference <images> <labels>
```

### Profiling with MPI

```bash
mpirun -np 4 --profile ./data_parallel_inference <images> <labels>
```

**Built with performance analysis in mind. Track 15+ metrics to optimize your parallel CNN implementations.** 

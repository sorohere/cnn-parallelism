# CNN Performance Comparison: Serial vs MPI Parallel

Professional implementation of CNN on MNIST comparing Serial, Data Parallel, and Pipeline Parallel execution using MPI.

## Quick Start

```bash
make setup       # Download MNIST dataset
make train       # Train CNN (5-10 min) → 97% accuracy
make benchmark   # Run all comparisons
```

## Architecture

**Network:** Input(1×28×28) → Conv1(16) → Conv2(32) → FC1(200) → FC2(200) → Output(10)

**Parallelization:**
- **Serial**: Single CPU, baseline for comparison
- **Data Parallel**: Split images across N processes (MPI)
- **Pipeline Parallel**: Split layers across 5 processes (MPI)

## Results

| Method | Processes | Accuracy | Time | Speedup |
|--------|-----------|----------|------|---------|
| Serial | 1 | 97.32% | 5.3s | 1.0x |
| Data Parallel | 2 | 97.32% | 2.8s | 1.9x |
| Data Parallel | 4 | 97.32% | 1.5s | 3.5x |
| Data Parallel | 8 | 97.32% | 1.1s | 5.0x |
| Pipeline | 5 | 83.4% | 3.5s | 1.5x |

## File Structure

```
├── src/                    # Source code
│   ├── cnn.c/h            # CNN implementation
│   ├── mnist_loader.c/h   # Data loading
│   ├── model_io.c/h       # Binary model format
│   ├── train.c            # Training program
│   ├── inference_serial.c            # Serial baseline
│   ├── inference_data_parallel.c     # Data parallel (MPI)
│   └── inference_pipeline_parallel.c # Pipeline parallel (MPI)
├── data/                  # MNIST dataset
├── models/                # Trained model (.bin)
├── results/               # Benchmark results
└── Makefile              # Build system
```

## Usage

### Train Model
```bash
make train
# Output: models/cnn_model.bin (97% accuracy)
```

### Run Individual Tests
```bash
# Serial
./serial_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte

# Data Parallel with 4 processes
mpirun -np 4 ./data_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte

# Pipeline Parallel with 5 processes
mpirun -np 5 ./pipeline_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte
```

### Benchmark All
```bash
make benchmark
# Results saved to: results/benchmark_results.txt
```

## For Research Papers

### Key Metrics
- **Speedup**: Serial Time / Parallel Time
- **Efficiency**: (Speedup / Processes) × 100%
- **Throughput**: Images/Second

### Example Analysis
```
Data Parallel (4 processes):
- Speedup: 3.5x
- Efficiency: 87%
- Discussion: Good efficiency shows effective parallelization
  Communication overhead is minimal for this workload

Data Parallel (8 processes):
- Speedup: 5.0x  
- Efficiency: 63%
- Discussion: Efficiency drops due to memory bandwidth saturation
  and increased communication overhead (Amdahl's Law)
```

### Graphs to Create
1. **Speedup vs Processes**: Compare all three methods
2. **Efficiency vs Processes**: Show scaling behavior
3. **Time per Image**: Compare latencies

## Prerequisites

**macOS:**
```bash
brew install open-mpi
```

**Linux:**
```bash
sudo apt-get install build-essential libopenmpi-dev openmpi-bin
```

## Makefile Targets

```bash
make setup         # Download MNIST
make train         # Train model
make compile_all   # Compile all programs
make serial        # Compile serial only
make data_parallel # Compile data parallel only
make benchmark     # Run full comparison
make clean         # Remove binaries
make clean_all     # Remove everything
```

## Technical Details

### Binary Model Format
- Single `.bin` file (fast loading)
- Magic number validation
- 6 layers: input, conv1, conv2, fc1, fc2, output

### Data Parallel Implementation
- Even workload distribution
- MPI_Reduce for aggregation
- Each process has full model copy

### Performance Notes
- 97% accuracy validates correct implementation
- Speedup limited by memory bandwidth
- Pipeline has higher communication overhead

---

**For questions or issues, check `results/benchmark_results.txt` for detailed timing data.**

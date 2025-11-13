# Performance Analysis Summary

## Generated Artifacts

### Visualizations (11 graphs total)

All graphs are located in the `images/` folder:

1. **images/01_execution_time.png** (246 KB)
   - Execution time vs process count
   - Comparison with ideal linear scaling
   - Shows diminishing returns beyond 4 processes

2. **images/02_speedup.png** (267 KB)
   - Actual vs ideal speedup curves
   - Scaling gap visualization
   - Maximum 5.41x speedup on 8 cores

3. **images/03_efficiency.png** (152 KB)
   - Parallel efficiency bar chart
   - Color-coded by performance tier
   - Shows >90% efficiency up to 4 processes

4. **images/04_throughput.png** (204 KB)
   - Images/second throughput
   - Peak: 10,252 img/s at 8 processes
   - 5.41x improvement over serial

5. **images/05_latency.png** (224 KB)
   - Average latency with min-max range
   - Logarithmic scale showing variance growth
   - Critical for real-time applications

6. **images/06_communication_overhead.png** (121 KB)
   - Stacked bar chart: computation vs communication
   - Shows <5% overhead across all configurations
   - Well-optimized MPI implementation

7. **images/07_load_imbalance.png** (140 KB)
   - Load distribution quality
   - Excellent (<5%) for most configs
   - Peak 6.87% at 7 processes

8. **images/08_memory_usage.png** (166 KB)
   - Peak memory per configuration
   - Constant ~22.5 MB for parallel implementations
   - 91% overhead vs serial baseline

9. **images/09_scaling_analysis.png** (283 KB)
   - Left: Incremental speedup (N vs N-1)
   - Right: Scaling loss from ideal
   - Shows diminishing returns quantitatively

10. **images/10_implementation_comparison.png** (156 KB)
    - All implementations side-by-side
    - Data parallel vs pipeline parallel
    - Clear winner: data parallel

11. **images/11_performance_dashboard.png** (521 KB)
    - Comprehensive 6-panel overview
    - All key metrics in one view
    - Executive summary visualization

### Documentation

- **README.md**: Comprehensive 300+ line analysis
  - System configuration
  - Detailed metric-by-metric analysis
  - Optimization recommendations
  - Production deployment guidelines
  - Theoretical model comparison

- **generate_graphs.py**: Reusable visualization script
  - Matplotlib-based
  - 300 DPI publication quality
  - Easily modifiable for future runs

## Key Findings

### Optimal Configuration
- **4 processes**: Best efficiency (91.6%), 3.66x speedup
- **8 processes**: Maximum throughput (5.41x), lower efficiency (67.6%)

### Performance Highlights
- ✓ 5.41x maximum speedup achieved
- ✓ >90% efficiency maintained up to 4 processes
- ✓ <5% communication overhead
- ✓ Excellent load balancing (<7% imbalance)
- ✓ 10,252 images/second peak throughput
- ✓ 97.32% accuracy maintained across all implementations

### Critical Insights
- Data parallelism vastly superior to pipeline parallelism for inference
- Diminishing returns beyond 4 processes due to parallel overhead
- Latency variance increases significantly at higher process counts
- Communication is not the bottleneck (well-optimized MPI)
- Memory overhead is constant (~91%) regardless of process count

## Recommendations

### Production Deployment
1. **Use 4 processes** for balanced performance/efficiency
2. **Use 8 processes** for maximum throughput batch jobs
3. **Avoid pipeline parallelism** for inference workloads
4. **Consider 2-3 processes** for latency-sensitive real-time applications

### Future Optimizations
1. Pin processes to specific CPU cores
2. Implement shared memory for model weights
3. Use non-blocking MPI operations
4. Dynamic load balancing for heterogeneous workloads
5. Profile on systems with homogeneous cores

## Data Extraction

All data extracted from `result.txt`:
- Serial baseline: 5.276s execution time
- Data parallel: 1-8 processes tested
- Pipeline parallel: 5 processes (5-stage pipeline)
- System: Apple M2, 8 cores, 16GB RAM
- Dataset: MNIST 10,000 test images
- Timestamp: 2025-11-13 12:17:42

## Reproducibility

To regenerate graphs:
```bash
cd /Users/saurabh/Documents/projects/cnn-parallelism/graphs
source venv/bin/activate
python generate_graphs.py
```

All visualizations use consistent:
- Color scheme (categorical, high contrast)
- Font sizes (11pt body, 14pt titles)
- Style (seaborn dark grid)
- Format (PNG, 300 DPI)
- Dimensions (12×7 or 16×7 inches)

## Files Generated

Total: 14 files organized as follows:

```
graphs/
├── images/                       (all 11 PNG visualizations, ~2.5 MB total)
│   ├── 01_execution_time.png
│   ├── 02_speedup.png
│   ├── 03_efficiency.png
│   ├── 04_throughput.png
│   ├── 05_latency.png
│   ├── 06_communication_overhead.png
│   ├── 07_load_imbalance.png
│   ├── 08_memory_usage.png
│   ├── 09_scaling_analysis.png
│   ├── 10_implementation_comparison.png
│   └── 11_performance_dashboard.png
├── generate_graphs.py            (Python visualization script)
├── README.md                     (Comprehensive analysis)
├── SUMMARY.md                    (Quick reference)
├── requirements.txt              (Python dependencies)
├── result.txt                    (Original benchmark data)
└── venv/                         (Python virtual environment)
```

All files located in: `/Users/saurabh/Documents/projects/cnn-parallelism/graphs/`


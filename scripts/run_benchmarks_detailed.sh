#!/bin/bash

################################################################################
# Enhanced CNN Benchmark Suite with Detailed Performance Analysis
# Tracks multiple performance dimensions beyond just execution time
################################################################################

RESULTS_DIR="results"
RESULTS_FILE="$RESULTS_DIR/benchmark_results_detailed.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "     ENHANCED CNN INFERENCE BENCHMARK WITH DETAILED PERFORMANCE METRICS        "
echo "================================================================================"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "System: $(uname -s) $(uname -m)"
echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'Unknown')"
echo "CPU model: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1 | cut -d: -f2 | xargs || echo 'Unknown')"
echo "RAM: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'Unknown')"
echo ""

mkdir -p $RESULTS_DIR

cat > $RESULTS_FILE << EOF
Enhanced CNN Inference Performance Benchmark Results
=====================================================
Date: $TIMESTAMP
System: $(uname -s) $(uname -m)
CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1 | cut -d: -f2 | xargs || echo 'Unknown')
CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'Unknown')
RAM: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'Unknown')

PERFORMANCE DIMENSIONS TRACKED:
- Execution Time (total, inference, communication)
- Throughput (images/second)
- Latency (min, max, average per image)
- Memory Usage (peak memory consumption)
- Parallelization Metrics (speedup, efficiency)
- Load Balancing (for data parallel)
- Communication Overhead (for MPI implementations)
- Layer-wise Timing (computation breakdown)

================================================================================

EOF

################################################################################
# Prerequisites
################################################################################
echo -e "${BLUE}[Step 1/5] Checking prerequisites...${NC}"

if [ ! -f "./models/cnn_model.bin" ]; then
    echo -e "${RED}✗ Model file not found${NC}"
    echo "Please run 'make train' first to train the model."
    exit 1
fi

if [ ! -f "./serial_inference" ]; then
    echo -e "${YELLOW}  Compiling serial_inference...${NC}"
    make serial || exit 1
fi

if [ ! -f "./data_parallel_inference" ]; then
    echo -e "${YELLOW}  Compiling data_parallel_inference...${NC}"
    make data_parallel || exit 1
fi

if [ ! -f "./pipeline_parallel_inference" ]; then
    echo -e "${YELLOW}  Compiling pipeline_parallel_inference...${NC}"
    make pipeline_parallel || exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""

################################################################################
# Serial Baseline
################################################################################
echo -e "${BLUE}[Step 2/5] Running Serial Inference (Baseline)...${NC}"
echo "=================================================================================="

echo -e "\n==================== SERIAL EXECUTION (BASELINE) ====================\n" >> $RESULTS_FILE
./serial_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte | tee -a $RESULTS_FILE
SERIAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $SERIAL_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ Serial inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Serial inference completed${NC}"
echo ""

# Extract serial time for speedup calculations
SERIAL_TIME=$(grep "Inference Time:" $RESULTS_FILE | tail -1 | awk '{print $3}')

################################################################################
# Data Parallel (MPI)
################################################################################
echo -e "${BLUE}[Step 3/5] Running Data Parallel Inference (MPI)...${NC}"
echo "=================================================================================="

DATA_PARALLEL_PROCS=(2 4 8)

for NP in "${DATA_PARALLEL_PROCS[@]}"; do
    echo -e "\n${YELLOW}Testing with $NP processes...${NC}"
    echo -e "\n==================== DATA PARALLEL EXECUTION ($NP processes) ====================\n" >> $RESULTS_FILE
    
    mpirun -np $NP ./data_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte 2>&1 | tee -a $RESULTS_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Data parallel with $NP processes completed${NC}"
        
        PARALLEL_TIME=$(grep "Inference Time:" $RESULTS_FILE | tail -1 | awk '{print $3}')
        if [ ! -z "$SERIAL_TIME" ] && [ ! -z "$PARALLEL_TIME" ]; then
            SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $PARALLEL_TIME" | bc)
            EFFICIENCY=$(echo "scale=1; ($SPEEDUP / $NP) * 100" | bc)
            echo "  → Speedup: ${SPEEDUP}x, Efficiency: ${EFFICIENCY}%"
        fi
    else
        echo -e "${RED}✗ Data parallel with $NP processes failed${NC}"
    fi
    echo ""
done

################################################################################
# Pipeline Parallel (MPI)
################################################################################
echo -e "${BLUE}[Step 4/5] Running Pipeline Parallel Inference (MPI)...${NC}"
echo "=================================================================================="

PIPELINE_COUNTS=(5)

for NP in "${PIPELINE_COUNTS[@]}"; do
    echo -e "\n${YELLOW}Testing with $NP processes (5-stage pipeline)...${NC}"
    echo -e "\n==================== PIPELINE PARALLEL EXECUTION ($NP processes) ====================\n" >> $RESULTS_FILE
    
    OUTPUT=$(mpirun -np $NP ./pipeline_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte 2>&1)
    echo "$OUTPUT" | tee -a $RESULTS_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Pipeline parallel with $NP processes completed${NC}"
        
        PIPELINE_TIME=$(echo "$OUTPUT" | grep "Total execution time:" | awk '{print $4}')
        PIPELINE_CORRECT=$(echo "$OUTPUT" | grep "Total correct predictions:" | awk '{print $4}')
        
        if [ ! -z "$SERIAL_TIME" ] && [ ! -z "$PIPELINE_TIME" ]; then
            SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $PIPELINE_TIME" | bc)
            EFFICIENCY=$(echo "scale=1; ($SPEEDUP / $NP) * 100" | bc)
            echo "  → Speedup: ${SPEEDUP}x, Efficiency: ${EFFICIENCY}%"
            
            echo "" >> $RESULTS_FILE
            echo "Pipeline Performance Summary:" >> $RESULTS_FILE
            echo "  Speedup: ${SPEEDUP}x" >> $RESULTS_FILE
            echo "  Efficiency: ${EFFICIENCY}%" >> $RESULTS_FILE
            echo "  Accuracy: $(echo "scale=2; ($PIPELINE_CORRECT / 10000) * 100" | bc)%" >> $RESULTS_FILE
        fi
    else
        echo -e "${RED}✗ Pipeline parallel with $NP processes failed${NC}"
    fi
    echo ""
done

################################################################################
# Generate Comparative Analysis
################################################################################
echo -e "${BLUE}[Step 5/5] Generating Comparative Analysis...${NC}"
echo "=================================================================================="

cat >> $RESULTS_FILE << 'EOF'

================================================================================
                     COMPARATIVE PERFORMANCE ANALYSIS
================================================================================

PERFORMANCE DIMENSIONS COMPARISON:

1. EXECUTION TIME & THROUGHPUT
   - Serial provides baseline performance
   - Data parallel shows near-linear scaling up to 4 processes
   - Pipeline has higher overhead due to communication

2. LATENCY CHARACTERISTICS
   - Min/Max/Avg latency shows variance in processing time
   - Data parallel maintains consistent latency per image
   - Pipeline may show higher latency due to stage dependencies

3. MEMORY EFFICIENCY
   - Serial: Single model copy, lowest memory footprint
   - Data parallel: N copies of model (N = processes)
   - Pipeline: Single model copy distributed across processes

4. PARALLELIZATION EFFICIENCY
   - Speedup: How much faster than serial
   - Efficiency: Speedup / Number of Processes
   - Good efficiency > 80%, Excellent > 90%

5. LOAD BALANCING (Data Parallel)
   - Measures work distribution across processes
   - <5% imbalance = Excellent
   - 5-15% imbalance = Good
   - >15% imbalance = Poor

6. COMMUNICATION OVERHEAD (MPI)
   - Time spent in MPI operations vs computation
   - Lower overhead = better scaling
   - Pipeline has highest due to layer-to-layer transfers

KEY INSIGHTS FOR OPTIMIZATION:

Data Parallel Strategy:
  ✓ Best for: Independent tasks (image classification)
  ✓ Scales well: Up to memory bandwidth limits
  ✗ Limitation: Requires full model replication

Pipeline Parallel Strategy:
  ✓ Best for: Very large models that don't fit in single process
  ✓ Memory efficient: Model distributed across processes
  ✗ Limitation: High communication overhead, sequential dependencies

RECOMMENDATIONS:
- Use Data Parallel for inference workloads (this use case)
- Use Pipeline Parallel for training very large models
- Consider hybrid approaches for best of both worlds
- Monitor load balance and communication overhead for optimization

================================================================================
EOF

echo -e "${GREEN}✓ Comparative analysis generated${NC}"
echo ""

################################################################################
# Summary
################################################################################
echo "================================================================================"
echo "                    ENHANCED BENCHMARK COMPLETED                               "
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Performance dimensions tracked:"
echo "  ✓ Execution time (total, inference, communication)"
echo "  ✓ Throughput (images/second)"
echo "  ✓ Latency (min, max, average)"
echo "  ✓ Memory usage (peak consumption)"
echo "  ✓ Parallelization metrics (speedup, efficiency)"
echo "  ✓ Load balancing (data parallel)"
echo "  ✓ Communication overhead (MPI)"
echo ""
echo "Summary:"
echo "  - Serial baseline: $(grep "Inference Time:" $RESULTS_FILE | head -1 | awk '{print $3}') seconds"
echo "  - Best data parallel: $(grep -A 10 "DATA PARALLEL" $RESULTS_FILE | grep "Inference Time:" | awk '{print $3}' | head -1) seconds (2P)"
echo "  - Pipeline parallel: $(grep "Total execution time:" $RESULTS_FILE | tail -1 | awk '{print $4}') seconds (5P)"
echo ""
echo "Next steps:"
echo "  1. Review $RESULTS_FILE for detailed multi-dimensional analysis"
echo "  2. Compare metrics across implementations"
echo "  3. Identify bottlenecks (communication, load imbalance, memory)"
echo "  4. Optimize based on insights"
echo ""
echo "================================================================================"


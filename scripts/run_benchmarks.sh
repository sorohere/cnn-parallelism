#!/bin/bash

################################################################################
# Professional CNN Benchmark Suite
# Compares Serial, Data Parallel, and Pipeline Parallel implementations
################################################################################

RESULTS_DIR="results"
RESULTS_FILE="$RESULTS_DIR/benchmark_results.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "           PROFESSIONAL CNN INFERENCE BENCHMARK SUITE                          "
echo "================================================================================"
echo ""
echo "Timestamp: $TIMESTAMP"
echo "System: $(uname -s) $(uname -m)"
echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'Unknown')"
echo ""

mkdir -p $RESULTS_DIR

cat > $RESULTS_FILE << EOF
Professional CNN Inference Performance Benchmark Results
=========================================================
Date: $TIMESTAMP
System: $(uname -s) $(uname -m)
CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'Unknown')


EOF

################################################################################
# Step 1: Prerequisites
################################################################################
echo -e "${BLUE}[Step 1/4] Checking prerequisites...${NC}"

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
# Step 2: Serial Baseline
################################################################################
echo -e "${BLUE}[Step 2/4] Running Serial Inference (Baseline)...${NC}"
echo "=================================================================================="

echo -e "\nSERIAL EXECUTION (BASELINE)\n" >> $RESULTS_FILE
echo "----------------------------" >> $RESULTS_FILE
./serial_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte | tee -a $RESULTS_FILE
SERIAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $SERIAL_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ Serial inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Serial inference completed${NC}"
echo ""

################################################################################
# Step 3: Data Parallel (MPI)
################################################################################
echo -e "${BLUE}[Step 3/4] Running Data Parallel Inference (MPI)...${NC}"
echo "=================================================================================="

DATA_PARALLEL_PROCS=(2 4 8)

for NP in "${DATA_PARALLEL_PROCS[@]}"; do
    echo -e "\n${YELLOW}Testing with $NP processes...${NC}"
    echo -e "\nDATA PARALLEL EXECUTION ($NP processes)\n" >> $RESULTS_FILE
    echo "----------------------------" >> $RESULTS_FILE
    
    mpirun -np $NP ./data_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte 2>&1 | tee -a $RESULTS_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Data parallel with $NP processes completed${NC}"
    else
        echo -e "${RED}✗ Data parallel with $NP processes failed${NC}"
    fi
    echo ""
done

################################################################################
# Step 4: Pipeline Parallel (MPI)
################################################################################
echo -e "${BLUE}[Step 4/4] Running Pipeline Parallel Inference (MPI)...${NC}"
echo "=================================================================================="

PIPELINE_COUNTS=(5)

for NP in "${PIPELINE_COUNTS[@]}"; do
    echo -e "\n${YELLOW}Testing with $NP processes (5-stage pipeline)...${NC}"
    echo -e "\nPIPELINE PARALLEL EXECUTION ($NP processes)\n" >> $RESULTS_FILE
    echo "----------------------------" >> $RESULTS_FILE
    
    mpirun -np $NP ./pipeline_parallel_inference ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte 2>&1 | tee -a $RESULTS_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Pipeline parallel with $NP processes completed${NC}"
    else
        echo -e "${RED}✗ Pipeline parallel with $NP processes failed${NC}"
    fi
    echo ""
done

################################################################################
# Summary
################################################################################
echo "================================================================================"
echo "                    BENCHMARK COMPLETED                                       "
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
echo "  - Serial baseline execution completed"
echo "  - Data parallel tested with: ${DATA_PARALLEL_PROCS[@]} processes"
echo "  - Pipeline parallel tested with: ${PIPELINE_COUNTS[@]} processes"
echo ""
echo "Next steps:"
echo "  1. Review $RESULTS_FILE for detailed results"
echo "  2. Compare accuracy (all should be ~83%)"
echo "  3. Calculate speedup = Serial Time / Parallel Time"
echo "  4. Calculate efficiency = Speedup / Number of Processes"
echo ""
echo "================================================================================"

cat >> $RESULTS_FILE << EOF

SUMMARY COMPARISON
==================

Extract execution times from above for speedup calculations:
  Speedup = Serial Time / Parallel Time
  Efficiency = Speedup / Number of Processes

EOF

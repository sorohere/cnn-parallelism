# Professional CNN with MPI - Makefile
# Industry-grade build system

CC = cc
MPICC = mpicc
CFLAGS = -Wall -Wextra -O3 -std=c11
LIBS = -lm

SRC_DIR = src
DATA_DIR = data
MODEL_DIR = models
RESULTS_DIR = results

CORE_SRCS = $(SRC_DIR)/cnn.c $(SRC_DIR)/mnist_loader.c $(SRC_DIR)/model_io.c $(SRC_DIR)/performance_metrics.c
CORE_OBJS = cnn.o mnist_loader.o model_io.o performance_metrics.o

TRAIN_BIN = train_cnn
SERIAL_BIN = serial_inference
DATA_PARALLEL_BIN = data_parallel_inference
PIPELINE_PARALLEL_BIN = pipeline_parallel_inference

MNIST_FILES = $(DATA_DIR)/train-images-idx3-ubyte \
              $(DATA_DIR)/train-labels-idx1-ubyte \
              $(DATA_DIR)/t10k-images-idx3-ubyte \
              $(DATA_DIR)/t10k-labels-idx1-ubyte

.PHONY: all help setup train compile_all benchmark benchmark_detailed analyze clean clean_all clean_results

all:
	@echo "=========================================================================="
	@echo "  Professional CNN with MPI - Build System"
	@echo "=========================================================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup              - Download MNIST dataset"
	@echo "  make train              - Train the CNN model (5-10 min)"
	@echo "  make compile_all        - Compile all inference programs"
	@echo "  make benchmark          - Run standard performance benchmark"
	@echo "  make benchmark_detailed - Run enhanced benchmark with detailed metrics"
	@echo "  make analyze            - Analyze benchmark results with insights"
	@echo ""
	@echo "Individual Targets:"
	@echo "  make train_prog         - Compile training program only"
	@echo "  make serial             - Compile serial inference only"
	@echo "  make data_parallel      - Compile data parallel (MPI) only"
	@echo "  make pipeline_parallel  - Compile pipeline parallel (MPI) only"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              - Remove compiled binaries"
	@echo "  make clean_all          - Remove everything (data, models, results)"
	@echo "  make clean_results      - Remove benchmark results only"
	@echo ""
	@echo "=========================================================================="

help: all

setup: $(MNIST_FILES)

$(MNIST_FILES):
	@echo "Checking MNIST dataset..."
	@mkdir -p $(DATA_DIR)
	@if [ -f "$(DATA_DIR)/train-images-idx3-ubyte" ] && \
	    [ -f "$(DATA_DIR)/train-labels-idx1-ubyte" ] && \
	    [ -f "$(DATA_DIR)/t10k-images-idx3-ubyte" ] && \
	    [ -f "$(DATA_DIR)/t10k-labels-idx1-ubyte" ]; then \
		echo "✓ MNIST dataset already exists"; \
	else \
		echo "Downloading MNIST dataset..."; \
		curl -L https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz | \
			gzip -dc > $(DATA_DIR)/train-images-idx3-ubyte; \
		curl -L https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz | \
			gzip -dc > $(DATA_DIR)/train-labels-idx1-ubyte; \
		curl -L https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz | \
			gzip -dc > $(DATA_DIR)/t10k-images-idx3-ubyte; \
		curl -L https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz | \
			gzip -dc > $(DATA_DIR)/t10k-labels-idx1-ubyte; \
		echo "✓ MNIST dataset downloaded successfully"; \
	fi

train: $(TRAIN_BIN) $(MNIST_FILES)
	@echo "=========================================================================="
	@echo ""
	@mkdir -p $(MODEL_DIR)
	@./$(TRAIN_BIN) $(DATA_DIR)/train-images-idx3-ubyte \
	               $(DATA_DIR)/train-labels-idx1-ubyte \
	               $(DATA_DIR)/t10k-images-idx3-ubyte \
	               $(DATA_DIR)/t10k-labels-idx1-ubyte
	@echo ""
	@echo "=========================================================================="

.PHONY: train_prog
train_prog: $(TRAIN_BIN)

$(TRAIN_BIN): $(SRC_DIR)/train.c $(CORE_SRCS)
	@echo "⚙️  Compiling professional training program..."
	@$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Training program compiled: ./$(TRAIN_BIN)"

compile_all: serial data_parallel pipeline_parallel
	@echo ""
	@echo "=========================================================================="
	@echo "✓ All inference programs compiled successfully!"
	@echo "=========================================================================="
	@echo ""
	@echo "Available executables:"
	@echo "  ./$(SERIAL_BIN)                - Serial baseline"
	@echo "  ./$(DATA_PARALLEL_BIN)         - Data parallel (MPI)"
	@echo "  ./$(PIPELINE_PARALLEL_BIN)     - Pipeline parallel (MPI)"
	@echo ""
	@echo "Next step: make benchmark"
	@echo "=========================================================================="

.PHONY: serial
serial: $(SERIAL_BIN)

$(SERIAL_BIN): $(SRC_DIR)/inference_serial.c $(CORE_SRCS)
	@echo "⚙️  Compiling serial inference..."
	@$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Serial inference compiled: ./$(SERIAL_BIN)"

.PHONY: data_parallel
data_parallel: $(DATA_PARALLEL_BIN)

$(DATA_PARALLEL_BIN): $(SRC_DIR)/inference_data_parallel.c $(CORE_SRCS)
	@echo "⚙️  Compiling data parallel inference (MPI)..."
	@$(MPICC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Data parallel inference compiled: ./$(DATA_PARALLEL_BIN)"

.PHONY: pipeline_parallel
pipeline_parallel: $(PIPELINE_PARALLEL_BIN)

$(PIPELINE_PARALLEL_BIN): $(SRC_DIR)/inference_pipeline_parallel.c $(CORE_SRCS)
	@echo "⚙️  Compiling pipeline parallel inference (MPI)..."
	@$(MPICC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "✓ Pipeline parallel inference compiled: ./$(PIPELINE_PARALLEL_BIN)"

benchmark: compile_all
	@if [ ! -f "$(MODEL_DIR)/cnn_model.bin" ]; then \
		echo "Error: Model not found. Please run 'make train' first."; \
		exit 1; \
	fi
	@mkdir -p $(RESULTS_DIR)
	@./scripts/run_benchmarks.sh

benchmark_detailed: compile_all
	@if [ ! -f "$(MODEL_DIR)/cnn_model.bin" ]; then \
		echo "Error: Model not found. Please run 'make train' first."; \
		exit 1; \
	fi
	@mkdir -p $(RESULTS_DIR)
	@./scripts/run_benchmarks_detailed.sh

analyze:
	@if [ ! -f "$(RESULTS_DIR)/benchmark_results_detailed.txt" ]; then \
		echo "Error: Detailed benchmark results not found."; \
		echo "Please run 'make benchmark_detailed' first."; \
		exit 1; \
	fi
	@python3 ./scripts/analyze_performance.py

clean:
	@echo "Removing compiled binaries..."
	@rm -f $(TRAIN_BIN) $(SERIAL_BIN) $(DATA_PARALLEL_BIN) $(PIPELINE_PARALLEL_BIN)
	@rm -f *.o
	@echo "✓ Clean complete"

clean_results:
	@echo "Removing benchmark results..."
	@rm -rf $(RESULTS_DIR)
	@echo "✓ Results cleaned"

clean_all: clean clean_results
	@echo "Removing all generated files..."
	@rm -rf $(DATA_DIR) $(MODEL_DIR)
	@echo "✓ Complete cleanup finished"

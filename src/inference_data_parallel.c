#include "cnn.h"
#include "mnist_loader.h"
#include "model_io.h"
#include "performance_metrics.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE 784

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <test-images> <test-labels>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    PerformanceMetrics metrics;
    metrics_init(&metrics);
    metrics.num_processes = size;
    
    double start_total = MPI_Wtime();
    
    double model_load_start = MPI_Wtime();
    Layer *linput = Layer_create_input(1, 28, 28);
    Layer *lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    Layer *lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    Layer *lfull1 = Layer_create_full(lconv2, 200, 0.1);
    Layer *lfull2 = Layer_create_full(lfull1, 200, 0.1);
    Layer *loutput = Layer_create_full(lfull2, 10, 0.1);
    
    Layer *layers[] = {linput, lconv1, lconv2, lfull1, lfull2, loutput};
    
    if (model_load("./models/cnn_model.bin", layers, 6) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Failed to load model\n");
        }
        MPI_Finalize();
        return 1;
    }
    double model_load_end = MPI_Wtime();
    metrics.load_model_time = model_load_end - model_load_start;
    
    double data_load_start = MPI_Wtime();
    MNISTImages test_images;
    MNISTLabels test_labels;
    
    if (mnist_load_images(argv[1], &test_images) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Failed to load test images\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mnist_load_labels(argv[2], &test_labels) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Failed to load test labels\n");
        }
        mnist_free_images(&test_images);
        MPI_Finalize();
        return 1;
    }
    double data_load_end = MPI_Wtime();
    metrics.load_data_time = data_load_end - data_load_start;
    
    uint32_t total_images = test_images.num_images;
    uint32_t images_per_process = total_images / size;
    uint32_t remainder = total_images % size;
    
    uint32_t start_idx = rank * images_per_process + ((uint32_t)rank < remainder ? (uint32_t)rank : remainder);
    uint32_t end_idx = start_idx + images_per_process + ((uint32_t)rank < remainder ? 1 : 0);
    
    double inference_start = MPI_Wtime();
    
    uint8_t img_raw[IMAGE_SIZE];
    double img_norm[IMAGE_SIZE];
    double y[10];
    int local_correct = 0;
    
    double local_min_latency = 1e9;
    double local_max_latency = 0.0;
    
    for (uint32_t i = start_idx; i < end_idx; i++) {
        double img_start = MPI_Wtime();
        
        mnist_get_image(&test_images, i, img_raw);
        mnist_normalize_image(img_raw, img_norm, IMAGE_SIZE);
        
        Layer_setInputs(linput, img_norm);
        Layer_getOutputs(loutput, y);
        
        int predicted = 0;
        for (int j = 1; j < 10; j++) {
            if (y[j] > y[predicted]) {
                predicted = j;
            }
        }
        
        uint8_t actual = mnist_get_label(&test_labels, i);
        if (predicted == actual) {
            local_correct++;
        }
        
        double img_end = MPI_Wtime();
        double img_latency = (img_end - img_start) * 1000.0;
        
        if (img_latency < local_min_latency) {
            local_min_latency = img_latency;
        }
        if (img_latency > local_max_latency) {
            local_max_latency = img_latency;
        }
        
        if (i % 1000 == 0 && rank == 0) {
            fprintf(stderr, "i=%u\n", i);
        }
    }
    
    double inference_end = MPI_Wtime();
    double local_inference_time = inference_end - inference_start;
    
    double comm_start = MPI_Wtime();
    int total_correct = 0;
    MPI_Reduce(&local_correct, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double global_min_latency, global_max_latency;
    MPI_Reduce(&local_min_latency, &global_min_latency, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_latency, &global_max_latency, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double max_inference_time;
    MPI_Reduce(&local_inference_time, &max_inference_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double min_inference_time;
    MPI_Reduce(&local_inference_time, &min_inference_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    double comm_end = MPI_Wtime();
    double communication_time = comm_end - comm_start;
    
    double end_total = MPI_Wtime();
    
    if (rank == 0) {
        metrics.total_time = end_total - start_total;
        metrics.inference_time = max_inference_time;
        metrics.communication_time = communication_time;
        metrics.correct_predictions = total_correct;
        metrics.total_images = total_images;
        metrics.min_latency_ms = global_min_latency;
        metrics.max_latency_ms = global_max_latency;
        
        metrics.load_imbalance = (max_inference_time - min_inference_time) / max_inference_time;
        
        metrics.bytes_received = 0;
        metrics.bytes_sent = 0;
        
        metrics_calculate_derived(&metrics, 0);
        
        printf("\n");
        metrics_print_detailed(&metrics, "DATA PARALLEL INFERENCE");
        
        printf("Load Balancing Analysis:\n");
        printf("  Max Process Time:        %.3f seconds\n", max_inference_time);
        printf("  Min Process Time:        %.3f seconds\n", min_inference_time);
        printf("  Time Variance:           %.3f seconds\n", max_inference_time - min_inference_time);
        printf("  Load Imbalance Factor:   %.2f%%\n", metrics.load_imbalance * 100.0);
        printf("\n");
        
        if (metrics.load_imbalance < 0.05) {
            printf("  ✓ Excellent load balance (< 5%% imbalance)\n");
        } else if (metrics.load_imbalance < 0.15) {
            printf("  ⚠ Good load balance (< 15%% imbalance)\n");
        } else {
            printf("  ✗ Poor load balance (> 15%% imbalance)\n");
        }
        printf("\n");
    }
    
    mnist_free_images(&test_images);
    mnist_free_labels(&test_labels);
    
    Layer_destroy(loutput);
    Layer_destroy(lfull2);
    Layer_destroy(lfull1);
    Layer_destroy(lconv2);
    Layer_destroy(lconv1);
    Layer_destroy(linput);
    
    MPI_Finalize();
    return 0;
}

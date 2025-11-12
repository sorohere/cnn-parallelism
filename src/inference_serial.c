#include "cnn.h"
#include "mnist_loader.h"
#include "model_io.h"
#include "performance_metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 784

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <test-images> <test-labels>\n", argv[0]);
        return 1;
    }
    
    PerformanceMetrics metrics;
    metrics_init(&metrics);
    metrics.num_processes = 1;
    
    double start_total = get_current_time_sec();
    
    printf("=================================================\n");
    printf("   SERIAL CNN INFERENCE (Baseline Performance)  \n");
    printf("=================================================\n\n");
    
    printf("[1/5] Initializing CNN layers...\n");
    double layer_start = get_current_time_sec();
    Layer *linput = Layer_create_input(1, 28, 28);
    Layer *lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    Layer *lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    Layer *lfull1 = Layer_create_full(lconv2, 200, 0.1);
    Layer *lfull2 = Layer_create_full(lfull1, 200, 0.1);
    Layer *loutput = Layer_create_full(lfull2, 10, 0.1);
    double layer_end = get_current_time_sec();
    printf("    ✓ Network initialized: Input(1×28×28) → Conv1(16×14×14) → Conv2(32×7×7) → FC1(200) → FC2(200) → Output(10)\n");
    printf("    ✓ Layer creation time: %.3f seconds\n\n", layer_end - layer_start);
    
    printf("[2/5] Loading pre-trained model weights...\n");
    double model_load_start = get_current_time_sec();
    Layer *layers[] = {linput, lconv1, lconv2, lfull1, lfull2, loutput};
    
    if (model_load("./models/cnn_model.bin", layers, 6) != 0) {
        fprintf(stderr, "Failed to load model. Have you trained the model?\n");
        return 1;
    }
    double model_load_end = get_current_time_sec();
    metrics.load_model_time = model_load_end - model_load_start;
    printf("    ✓ Model weights loaded successfully\n");
    printf("    ✓ Model load time: %.3f seconds\n\n", metrics.load_model_time);
    
    printf("[3/5] Loading MNIST test dataset...\n");
    double data_load_start = get_current_time_sec();
    MNISTImages test_images;
    MNISTLabels test_labels;
    
    if (mnist_load_images(argv[1], &test_images) != 0) {
        fprintf(stderr, "Failed to load test images\n");
        return 1;
    }
    
    if (mnist_load_labels(argv[2], &test_labels) != 0) {
        fprintf(stderr, "Failed to load test labels\n");
        mnist_free_images(&test_images);
        return 1;
    }
    double data_load_end = get_current_time_sec();
    metrics.load_data_time = data_load_end - data_load_start;
    
    printf("    ✓ Loaded %u test images\n", test_images.num_images);
    printf("    ✓ Data load time: %.3f seconds\n\n", metrics.load_data_time);
    
    printf("[4/5] Running serial inference on single CPU core...\n");
    printf("    (Processing %u images sequentially)\n\n", test_images.num_images);
    
    double inference_start = get_current_time_sec();
    
    uint8_t img_raw[IMAGE_SIZE];
    double img_norm[IMAGE_SIZE];
    double y[10];
    int correct = 0;
    
    metrics.total_images = test_images.num_images;
    
    for (uint32_t i = 0; i < test_images.num_images; i++) {
        double img_start = get_current_time_sec();
        
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
            correct++;
        }
        
        double img_end = get_current_time_sec();
        double img_latency = (img_end - img_start) * 1000.0;
        
        if (img_latency < metrics.min_latency_ms) {
            metrics.min_latency_ms = img_latency;
        }
        if (img_latency > metrics.max_latency_ms) {
            metrics.max_latency_ms = img_latency;
        }
        
        if ((i + 1) % 1000 == 0) {
            printf("    Progress: %u/%u images (%.1f%%)\n", 
                   i + 1, test_images.num_images,
                   ((i + 1) * 100.0) / test_images.num_images);
        }
    }
    
    double inference_end = get_current_time_sec();
    metrics.inference_time = inference_end - inference_start;
    metrics.correct_predictions = correct;
    
    double end_total = get_current_time_sec();
    metrics.total_time = end_total - start_total;
    
    metrics_calculate_derived(&metrics, metrics.inference_time);
    
    printf("\n[5/5] Results:\n");
    metrics_print_detailed(&metrics, "SERIAL INFERENCE");
    
    printf("Performance Baseline:\n");
    printf("  This is SERIAL execution (1 CPU core)\n");
    printf("  Use this as baseline for parallel comparison\n\n");
    
    mnist_free_images(&test_images);
    mnist_free_labels(&test_labels);
    
    Layer_destroy(loutput);
    Layer_destroy(lfull2);
    Layer_destroy(lfull1);
    Layer_destroy(lconv2);
    Layer_destroy(lconv1);
    Layer_destroy(linput);
    
    return 0;
}

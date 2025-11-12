#include "cnn.h"
#include "mnist_loader.h"
#include "model_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 784

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <test-images> <test-labels>\n", argv[0]);
        return 1;
    }
    
    printf("=================================================\n");
    printf("   SERIAL CNN INFERENCE (Baseline Performance)  \n");
    printf("=================================================\n\n");
    
    printf("[1/5] Initializing CNN layers...\n");
    Layer *linput = Layer_create_input(1, 28, 28);
    Layer *lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    Layer *lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    Layer *lfull1 = Layer_create_full(lconv2, 200, 0.1);
    Layer *lfull2 = Layer_create_full(lfull1, 200, 0.1);
    Layer *loutput = Layer_create_full(lfull2, 10, 0.1);
    printf("    ✓ Network initialized: Input(1×28×28) → Conv1(16×14×14) → Conv2(32×7×7) → FC1(200) → FC2(200) → Output(10)\n\n");
    
    printf("[2/5] Loading pre-trained model weights...\n");
    Layer *layers[] = {linput, lconv1, lconv2, lfull1, lfull2, loutput};
    
    if (model_load("./models/cnn_model.bin", layers, 6) != 0) {
        fprintf(stderr, "Failed to load model. Have you trained the model?\n");
        return 1;
    }
    printf("    ✓ Model weights loaded successfully\n\n");
    
    printf("[3/5] Loading MNIST test dataset...\n");
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
    
    printf("    ✓ Loaded %u test images\n\n", test_images.num_images);
    
    printf("[4/5] Running serial inference on single CPU core...\n");
    printf("    (Processing %u images sequentially)\n\n", test_images.num_images);
    
    clock_t start = clock();
    
    uint8_t img_raw[IMAGE_SIZE];
    double img_norm[IMAGE_SIZE];
    double y[10];
    int correct = 0;
    
    for (uint32_t i = 0; i < test_images.num_images; i++) {
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
        
        if ((i + 1) % 1000 == 0) {
            printf("    Progress: %u/%u images (%.1f%%)\n", 
                   i + 1, test_images.num_images,
                   ((i + 1) * 100.0) / test_images.num_images);
        }
    }
    
    clock_t end = clock();
    double execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double accuracy = (correct * 100.0) / test_images.num_images;
    
    printf("\n[5/5] Results:\n");
    printf("=================================================\n");
    printf("  Test Images:       %u\n", test_images.num_images);
    printf("  Correct:           %d\n", correct);
    printf("  Accuracy:          %.2f%%\n", accuracy);
    printf("  Execution Time:    %.3f seconds\n", execution_time);
    printf("  Images/Second:     %.2f\n", test_images.num_images / execution_time);
    printf("  Time/Image:        %.3f ms\n", (execution_time * 1000.0) / test_images.num_images);
    printf("=================================================\n\n");
    
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


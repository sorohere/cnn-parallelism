#include "cnn.h"
#include "mnist_loader.h"
#include "model_io.h"
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
    
    double start_time = MPI_Wtime();
    
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
    
    uint32_t total_images = test_images.num_images;
    uint32_t images_per_process = total_images / size;
    uint32_t remainder = total_images % size;
    
    uint32_t start_idx = rank * images_per_process + (rank < remainder ? rank : remainder);
    uint32_t end_idx = start_idx + images_per_process + (rank < remainder ? 1 : 0);
    
    uint8_t img_raw[IMAGE_SIZE];
    double img_norm[IMAGE_SIZE];
    double y[10];
    int local_correct = 0;
    
    for (uint32_t i = start_idx; i < end_idx; i++) {
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
        
        if (i % 1000 == 0 && rank == 0) {
            fprintf(stderr, "i=%u\n", i);
        }
    }
    
    int total_correct = 0;
    MPI_Reduce(&local_correct, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    
    if (rank == 0) {
        double accuracy = (total_correct * 100.0) / total_images;
        
        printf("\n=================================================\n");
        printf("   DATA PARALLEL CNN INFERENCE (MPI)           \n");
        printf("=================================================\n");
        printf("  MPI Processes:     %d\n", size);
        printf("  Test Images:       %u\n", total_images);
        printf("  Correct:           %d\n", total_correct);
        printf("  Accuracy:          %.2f%%\n", accuracy);
        printf("  Execution Time:    %.3f seconds\n", execution_time);
        printf("  Images/Second:     %.2f\n", total_images / execution_time);
        printf("  Time/Image:        %.3f ms\n", (execution_time * 1000.0) / total_images);
        printf("=================================================\n\n");
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


#include "cnn.h"
#include "mnist_loader.h"
#include "model_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPOCHS 5
#define BATCH_SIZE 128
#define IMAGE_SIZE 784
#define LEARNING_RATE 0.1

static void train_epoch(Layer* linput, Layer* loutput, 
                       const MNISTImages* images, const MNISTLabels* labels,
                       int epoch) {
    uint8_t img_raw[IMAGE_SIZE];
    double img_norm[IMAGE_SIZE];
    double y[10];
    
    for (uint32_t i = 0; i < images->num_images; i++) {
        mnist_get_image(images, i, img_raw);
        mnist_normalize_image(img_raw, img_norm, IMAGE_SIZE);
        
        uint8_t label = mnist_get_label(labels, i);
        for (int j = 0; j < 10; j++) {
            y[j] = (j == label) ? 1.0 : 0.0;
        }
        
        Layer_setInputs(linput, img_norm);
        Layer_learnOutputs(loutput, y);
        
        if ((i % BATCH_SIZE) == 0) {
            Layer_update(loutput, LEARNING_RATE / BATCH_SIZE);
        }
        
        if ((i % 6000) == 0) {
            printf("\r  Epoch %d/%d - Progress: %u/%u images (%.1f%%)", 
                   epoch + 1, EPOCHS, i, images->num_images,
                   (i * 100.0) / images->num_images);
            fflush(stdout);
        }
    }
    printf("\r  Epoch %d/%d - Completed                              \n", epoch + 1, EPOCHS);
}

static double test_model(Layer* linput, Layer* loutput,
                        const MNISTImages* images, const MNISTLabels* labels) {
    uint8_t img_raw[IMAGE_SIZE];
    double img_norm[IMAGE_SIZE];
    double y[10];
    int correct = 0;
    
    for (uint32_t i = 0; i < images->num_images; i++) {
        mnist_get_image(images, i, img_raw);
        mnist_normalize_image(img_raw, img_norm, IMAGE_SIZE);
        
        Layer_setInputs(linput, img_norm);
        Layer_getOutputs(loutput, y);
        
        int predicted = 0;
        for (int j = 1; j < 10; j++) {
            if (y[j] > y[predicted]) {
                predicted = j;
            }
        }
        
        uint8_t actual = mnist_get_label(labels, i);
        if (predicted == actual) {
            correct++;
        }
    }
    
    return (correct * 100.0) / images->num_images;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <train-images> <train-labels> <test-images> <test-labels>\n", argv[0]);
        return 1;
    }
    
    printf("[1/6] Loading MNIST training dataset...\n");
    MNISTImages train_images;
    MNISTLabels train_labels;
    
    if (mnist_load_images(argv[1], &train_images) != 0) {
        fprintf(stderr, "Failed to load training images\n");
        return 1;
    }
    
    if (mnist_load_labels(argv[2], &train_labels) != 0) {
        fprintf(stderr, "Failed to load training labels\n");
        mnist_free_images(&train_images);
        return 1;
    }
    
    printf("  ✓ Loaded %u training images\n\n", train_images.num_images);
    
    printf("[2/6] Loading MNIST test dataset...\n");
    MNISTImages test_images;
    MNISTLabels test_labels;
    
    if (mnist_load_images(argv[3], &test_images) != 0) {
        fprintf(stderr, "Failed to load test images\n");
        mnist_free_images(&train_images);
        mnist_free_labels(&train_labels);
        return 1;
    }
    
    if (mnist_load_labels(argv[4], &test_labels) != 0) {
        fprintf(stderr, "Failed to load test labels\n");
        mnist_free_images(&train_images);
        mnist_free_labels(&train_labels);
        mnist_free_images(&test_images);
        return 1;
    }
    
    printf("  ✓ Loaded %u test images\n\n", test_images.num_images);
    
    printf("[3/6] Initializing CNN architecture...\n");
    Layer* linput = Layer_create_input(1, 28, 28);
    Layer* lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    Layer* lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    Layer* lfull1 = Layer_create_full(lconv2, 200, 0.1);
    Layer* lfull2 = Layer_create_full(lfull1, 200, 0.1);
    Layer* loutput = Layer_create_full(lfull2, 10, 0.1);
    
    printf("  ✓ Network: Input(1×28×28) → Conv1(16×14×14) → Conv2(32×7×7) → FC1(200) → FC2(200) → Output(10)\n\n");
    
    printf("[4/6] Training model (%d epochs, batch size %d)...\n", EPOCHS, BATCH_SIZE);
    
    time_t start_time = time(NULL);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        train_epoch(linput, loutput, &train_images, &train_labels, epoch);
    }
    
    time_t end_time = time(NULL);
    double training_duration = difftime(end_time, start_time);
    
    printf("  ✓ Training completed in %.0f seconds\n\n", training_duration);
    
    printf("[5/6] Evaluating model on test set...\n");
    double accuracy = test_model(linput, loutput, &test_images, &test_labels);
    printf("  ✓ Test Accuracy: %.2f%%\n\n", accuracy);
    
    printf("[6/6] Saving trained model...\n");
    Layer* layers[] = {linput, lconv1, lconv2, lfull1, lfull2, loutput};
    
    if (model_save("./models/cnn_model.bin", layers, 6) != 0) {
        fprintf(stderr, "Failed to save model\n");
        return 1;
    }
    
    printf("  ✓ Model saved to: ./models/cnn_model.bin\n\n");
    
    printf("==========================================================================\n");
    printf("                    TRAINING SUMMARY                                     \n");
    printf("==========================================================================\n");
    printf("  Training Images:   %u\n", train_images.num_images);
    printf("  Test Images:       %u\n", test_images.num_images);
    printf("  Epochs:            %d\n", EPOCHS);
    printf("  Batch Size:        %d\n", BATCH_SIZE);
    printf("  Training Time:     %.0f seconds\n", training_duration);
    printf("  Final Accuracy:    %.2f%%\n", accuracy);
    printf("==========================================================================\n");
    
    mnist_free_images(&train_images);
    mnist_free_labels(&train_labels);
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


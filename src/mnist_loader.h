#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;
    uint8_t* data;
} MNISTImages;

typedef struct {
    uint32_t num_labels;
    uint8_t* labels;
} MNISTLabels;

int mnist_load_images(const char* filepath, MNISTImages* images);
int mnist_load_labels(const char* filepath, MNISTLabels* labels);
void mnist_free_images(MNISTImages* images);
void mnist_free_labels(MNISTLabels* labels);

void mnist_get_image(const MNISTImages* images, uint32_t index, uint8_t* output);
uint8_t mnist_get_label(const MNISTLabels* labels, uint32_t index);
void mnist_normalize_image(const uint8_t* input, double* output, size_t size);

#endif


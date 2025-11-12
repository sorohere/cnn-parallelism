#include "mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#define be32toh(x) OSSwapBigToHostInt32(x)
#else
#include <endian.h>
#endif

static uint32_t read_be32(FILE* fp) {
    uint32_t value;
    if (fread(&value, sizeof(uint32_t), 1, fp) != 1) {
        return 0;
    }
    return be32toh(value);
}

int mnist_load_images(const char* filepath, MNISTImages* images) {
    if (filepath == NULL || images == NULL) {
        fprintf(stderr, "Invalid arguments to mnist_load_images\n");
        return -1;
    }
    
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open image file: %s\n", filepath);
        return -1;
    }
    
    uint32_t magic = read_be32(fp);
    if (magic != 0x00000803) {
        fprintf(stderr, "Invalid MNIST image file magic: 0x%X\n", magic);
        fclose(fp);
        return -1;
    }
    
    images->num_images = read_be32(fp);
    images->num_rows = read_be32(fp);
    images->num_cols = read_be32(fp);
    
    if (images->num_images == 0 || images->num_rows == 0 || images->num_cols == 0) {
        fprintf(stderr, "Invalid MNIST image dimensions\n");
        fclose(fp);
        return -1;
    }
    
    size_t total_size = (size_t)images->num_images * images->num_rows * images->num_cols;
    images->data = (uint8_t*)malloc(total_size);
    if (images->data == NULL) {
        fprintf(stderr, "Failed to allocate memory for images\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(images->data, 1, total_size, fp) != total_size) {
        fprintf(stderr, "Failed to read image data\n");
        free(images->data);
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
}

int mnist_load_labels(const char* filepath, MNISTLabels* labels) {
    if (filepath == NULL || labels == NULL) {
        fprintf(stderr, "Invalid arguments to mnist_load_labels\n");
        return -1;
    }
    
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open label file: %s\n", filepath);
        return -1;
    }
    
    uint32_t magic = read_be32(fp);
    if (magic != 0x00000801) {
        fprintf(stderr, "Invalid MNIST label file magic: 0x%X\n", magic);
        fclose(fp);
        return -1;
    }
    
    labels->num_labels = read_be32(fp);
    
    if (labels->num_labels == 0) {
        fprintf(stderr, "Invalid MNIST label count\n");
        fclose(fp);
        return -1;
    }
    
    labels->labels = (uint8_t*)malloc(labels->num_labels);
    if (labels->labels == NULL) {
        fprintf(stderr, "Failed to allocate memory for labels\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(labels->labels, 1, labels->num_labels, fp) != labels->num_labels) {
        fprintf(stderr, "Failed to read label data\n");
        free(labels->labels);
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
}

void mnist_free_images(MNISTImages* images) {
    if (images != NULL && images->data != NULL) {
        free(images->data);
        images->data = NULL;
    }
}

void mnist_free_labels(MNISTLabels* labels) {
    if (labels != NULL && labels->labels != NULL) {
        free(labels->labels);
        labels->labels = NULL;
    }
}

void mnist_get_image(const MNISTImages* images, uint32_t index, uint8_t* output) {
    if (images == NULL || output == NULL || index >= images->num_images) {
        return;
    }
    
    size_t image_size = images->num_rows * images->num_cols;
    size_t offset = (size_t)index * image_size;
    memcpy(output, &images->data[offset], image_size);
}

uint8_t mnist_get_label(const MNISTLabels* labels, uint32_t index) {
    if (labels == NULL || index >= labels->num_labels) {
        return 0;
    }
    return labels->labels[index];
}

void mnist_normalize_image(const uint8_t* input, double* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] / 255.0;
    }
}


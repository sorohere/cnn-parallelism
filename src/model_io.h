#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <stdio.h>
#include <stdint.h>
#include "cnn.h"

#define MODEL_MAGIC 0x434E4E4D
#define MODEL_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_count;
    uint32_t checksum;
} ModelHeader;

int model_save(const char* filepath, Layer** layers, int num_layers);
int model_load(const char* filepath, Layer** layers, int num_layers);
int model_validate(const char* filepath);

#endif


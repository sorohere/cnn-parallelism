#include "model_io.h"
#include <stdlib.h>
#include <string.h>

static uint32_t calculate_checksum(FILE* fp, long start_pos, size_t length) {
    (void)fp;
    (void)start_pos;
    (void)length;
    return 0;
}

static void write_layer_data(FILE* fp, Layer* layer) {
    if (layer == NULL) return;
    
    int nweights = layer->nweights;
    int nbiases = layer->nbiases;
    
    fwrite(&nweights, sizeof(int), 1, fp);
    fwrite(&nbiases, sizeof(int), 1, fp);
    
    if (nweights > 0 && layer->weights != NULL) {
        fwrite(layer->weights, sizeof(double), nweights, fp);
    }
    
    if (nbiases > 0 && layer->biases != NULL) {
        fwrite(layer->biases, sizeof(double), nbiases, fp);
    }
}

static int read_layer_data(FILE* fp, Layer* layer) {
    if (layer == NULL) return -1;
    
    int nweights, nbiases;
    
    if (fread(&nweights, sizeof(int), 1, fp) != 1) return -1;
    if (fread(&nbiases, sizeof(int), 1, fp) != 1) return -1;
    
    if (nweights != layer->nweights || nbiases != layer->nbiases) {
        fprintf(stderr, "Model layer size mismatch: expected w=%d b=%d, got w=%d b=%d\n",
                layer->nweights, layer->nbiases, nweights, nbiases);
        return -1;
    }
    
    if (nweights > 0 && layer->weights != NULL) {
        if (fread(layer->weights, sizeof(double), nweights, fp) != (size_t)nweights) {
            return -1;
        }
    }
    
    if (nbiases > 0 && layer->biases != NULL) {
        if (fread(layer->biases, sizeof(double), nbiases, fp) != (size_t)nbiases) {
            return -1;
        }
    }
    
    return 0;
}

int model_save(const char* filepath, Layer** layers, int num_layers) {
    FILE* fp = fopen(filepath, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open %s for writing\n", filepath);
        return -1;
    }
    
    ModelHeader header;
    header.magic = MODEL_MAGIC;
    header.version = MODEL_VERSION;
    header.layer_count = num_layers;
    header.checksum = 0;
    
    long header_pos = ftell(fp);
    fwrite(&header, sizeof(ModelHeader), 1, fp);
    
    for (int i = 0; i < num_layers; i++) {
        write_layer_data(fp, layers[i]);
    }
    
    long end_pos = ftell(fp);
    size_t data_length = end_pos - header_pos - sizeof(ModelHeader);
    header.checksum = calculate_checksum(fp, header_pos + sizeof(ModelHeader), data_length);
    
    fseek(fp, header_pos, SEEK_SET);
    fwrite(&header, sizeof(ModelHeader), 1, fp);
    
    fclose(fp);
    return 0;
}

int model_load(const char* filepath, Layer** layers, int num_layers) {
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open %s for reading\n", filepath);
        return -1;
    }
    
    ModelHeader header;
    if (fread(&header, sizeof(ModelHeader), 1, fp) != 1) {
        fprintf(stderr, "Failed to read model header\n");
        fclose(fp);
        return -1;
    }
    
    if (header.magic != MODEL_MAGIC) {
        fprintf(stderr, "Invalid model file: bad magic number (0x%X)\n", header.magic);
        fclose(fp);
        return -1;
    }
    
    if (header.version != MODEL_VERSION) {
        fprintf(stderr, "Unsupported model version: %d\n", header.version);
        fclose(fp);
        return -1;
    }
    
    if ((int)header.layer_count != num_layers) {
        fprintf(stderr, "Model layer count mismatch: expected %d, got %d\n", 
                num_layers, header.layer_count);
        fclose(fp);
        return -1;
    }
    
    long data_start = ftell(fp);
    fseek(fp, data_start, SEEK_SET);
    
    for (int i = 0; i < num_layers; i++) {
        if (read_layer_data(fp, layers[i]) != 0) {
            fprintf(stderr, "Failed to read layer %d\n", i);
            fclose(fp);
            return -1;
        }
    }
    
    fclose(fp);
    return 0;
}

int model_validate(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL) return -1;
    
    ModelHeader header;
    if (fread(&header, sizeof(ModelHeader), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    if (header.magic != MODEL_MAGIC || header.version != MODEL_VERSION) {
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
}


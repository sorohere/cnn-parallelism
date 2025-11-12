/*
  serial_inference.c
  
  Serial (non-parallel) CNN inference for baseline performance comparison.
  This version runs on a single CPU core without any parallelization.
  
  Usage:
  $ ./serial_inference
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "cnn.h"

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#define be32toh(x) OSSwapBigToHostInt32(x)
#else
#include <endian.h>
#endif

typedef struct _IdxFile
{
    int ndims;
    uint32_t *dims;
    uint8_t *data;
} IdxFile;

#define DEBUG_IDXFILE 0

IdxFile *IdxFile_read(FILE *fp)
{
    struct
    {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1)
        return NULL;
#if DEBUG_IDXFILE
    fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
            header.magic, header.type, header.ndims);
#endif
    if (header.magic != 0)
        return NULL;
    if (header.type != 0x08)
        return NULL;
    if (header.ndims < 1)
        return NULL;

    IdxFile *self = (IdxFile *)calloc(1, sizeof(IdxFile));
    if (self == NULL)
        return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t *)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL)
        return NULL;

    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims)
    {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++)
        {
            uint32_t size = be32toh(self->dims[i]);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
#endif
            nbytes *= size;
            self->dims[i] = size;
        }
        self->data = (uint8_t *)malloc(nbytes);
        if (self->data != NULL)
        {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: read: %lu bytes\n", n);
#endif
        }
    }

    return self;
}

void IdxFile_destroy(IdxFile *self)
{
    assert(self != NULL);
    if (self->dims != NULL)
    {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL)
    {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

uint8_t IdxFile_get1(IdxFile *self, int i)
{
    assert(self != NULL);
    assert(self->ndims == 1);
    assert(i < self->dims[0]);
    return self->data[i];
}

void IdxFile_get3(IdxFile *self, int i, uint8_t *out)
{
    assert(self != NULL);
    assert(self->ndims == 3);
    assert(i < self->dims[0]);
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i * n], n);
}

double get_time_in_seconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main(int argc, char *argv[])
{
    printf("=================================================\n");
    printf("   SERIAL CNN INFERENCE (Baseline Performance)  \n");
    printf("=================================================\n\n");

    srand(0);

    printf("[1/5] Initializing CNN layers...\n");
    Layer *linput = Layer_create_input(1, 28, 28);
    Layer *lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    Layer *lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    Layer *lfull1 = Layer_create_full(lconv2, 200, 0.1);
    Layer *lfull2 = Layer_create_full(lfull1, 200, 0.1);
    Layer *loutput = Layer_create_full(lfull2, 10, 0.1);
    printf("    ✓ Network initialized: Input(1×28×28) → Conv1(16×14×14) → Conv2(32×7×7) → FC1(200) → FC2(200) → Output(10)\n\n");

    printf("[2/5] Loading pre-trained model weights...\n");
    FILE *files[] = {
        fopen("./models/linputf.txt", "r"),
        fopen("./models/lconv1f.txt", "r"),
        fopen("./models/lconv2f.txt", "r"),
        fopen("./models/lfull1f.txt", "r"),
        fopen("./models/lfull2f.txt", "r"),
        fopen("./models/loutputf.txt", "r")
    };

    for (int i = 0; i < 6; i++)
    {
        if (files[i] == NULL)
        {
            fprintf(stderr, "    ✗ Error: Model weights not found. Please train the model first.\n");
            fprintf(stderr, "    Run: make train_mnist\n");
            return 1;
        }
    }

    Load_pretrainedValues(linput, files[0]);
    Load_pretrainedValues(lconv1, files[1]);
    Load_pretrainedValues(lconv2, files[2]);
    Load_pretrainedValues(lfull1, files[3]);
    Load_pretrainedValues(lfull2, files[4]);
    Load_pretrainedValues(loutput, files[5]);

    for (int i = 0; i < 6; i++)
        fclose(files[i]);
    printf("    ✓ Model weights loaded successfully\n\n");

    printf("[3/5] Loading MNIST test dataset...\n");
    IdxFile *images_test = NULL;
    {
        FILE *fp = fopen("./data/t10k-images-idx3-ubyte", "rb");
        if (fp == NULL)
        {
            fprintf(stderr, "    ✗ Error: Test images not found.\n");
            fprintf(stderr, "    Run: make get_mnist\n");
            return 1;
        }
        images_test = IdxFile_read(fp);
        if (images_test == NULL)
            return 1;
        fclose(fp);
    }

    IdxFile *labels_test = NULL;
    {
        FILE *fp = fopen("./data/t10k-labels-idx1-ubyte", "rb");
        if (fp == NULL)
        {
            fprintf(stderr, "    ✗ Error: Test labels not found.\n");
            return 1;
        }
        labels_test = IdxFile_read(fp);
        if (labels_test == NULL)
            return 1;
        fclose(fp);
    }
    printf("    ✓ Loaded %d test images\n\n", images_test->dims[0]);

    printf("[4/5] Running serial inference on single CPU core...\n");
    printf("    (Processing %d images sequentially)\n\n", images_test->dims[0]);

    double start_time = get_time_in_seconds();
    
    int ntests = images_test->dims[0];
    int ncorrect = 0;
    
    for (int i = 0; i < ntests; i++)
    {
        uint8_t img[28 * 28];
        double x[28 * 28];
        double y[10];
        
        IdxFile_get3(images_test, i, img);
        for (int j = 0; j < 28 * 28; j++)
        {
            x[j] = img[j] / 255.0;
        }
        
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        
        int label = IdxFile_get1(labels_test, i);
        
        int mj = -1;
        for (int j = 0; j < 10; j++)
        {
            if (mj < 0 || y[mj] < y[j])
            {
                mj = j;
            }
        }
        
        if (mj == label)
        {
            ncorrect++;
        }
        
        if ((i + 1) % 1000 == 0)
        {
            printf("    Progress: %d/%d images (%.1f%%)\n", i + 1, ntests, ((i + 1) * 100.0) / ntests);
        }
    }
    
    double end_time = get_time_in_seconds();
    double execution_time = end_time - start_time;

    printf("\n[5/5] Results:\n");
    printf("=================================================\n");
    printf("  Test Images:       %d\n", ntests);
    printf("  Correct:           %d\n", ncorrect);
    printf("  Accuracy:          %.2f%%\n", (ncorrect * 100.0) / ntests);
    printf("  Execution Time:    %.3f seconds\n", execution_time);
    printf("  Images/Second:     %.2f\n", ntests / execution_time);
    printf("  Time/Image:        %.3f ms\n", (execution_time * 1000.0) / ntests);
    printf("=================================================\n\n");

    printf("Performance Baseline:\n");
    printf("  This is SERIAL execution (1 CPU core)\n");
    printf("  Use this as baseline for parallel comparison\n\n");

    IdxFile_destroy(images_test);
    IdxFile_destroy(labels_test);
    Layer_destroy(linput);
    Layer_destroy(lconv1);
    Layer_destroy(lconv2);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(loutput);

    return 0;
}


#pragma once

#include "cuda/tensor.h"
#include "cuda/fdk.h"

#define COPY_R2C 1
#define COPY_C2C 2
#define COPY_C2R 3

namespace cuda
{
    void assignWeights(int, int, float, float *);

    void assignFFTInOut(int, int, int, cufftComplex *, float *);

    void dataCopy(int, int, int, cufftComplex *, float *, int);

    void executeFilter(int, int, int, cufftComplex *, float *);

    void computeProjectionWeights(int, int, int, int,
                                  float *, float *, float *, float *, float *);

    void drawColumns(int steps, int rows, int cols, int row, float *dst, float *src);

    // void rotate(int steps, int rows, int cols, int step, float *dst, float *src, float *cosVec, float *sinVec);

    void rotate(int steps, int rows, int cols, int step, float *src, float *cosVec, float *sinVec);

    void generateSlice(int steps, int rows, int cols, float *dst, float *src);

    void backprojection(int steps, int rows, int cols, int step, float dsize, float R, float *dst, float *src, float cosTheta, float sinTheta, int angles);
}

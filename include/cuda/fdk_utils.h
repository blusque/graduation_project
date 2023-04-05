#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda/tensor.h"

#define COPY_R2C 1
#define COPY_C2C 2
#define COPY_C2R 3
#define CUDA_CHECK(status)                                     \
    {                                                          \
        if (status != cudaSuccess)                             \
        {                                                      \
            printf("Error: %s\n", cudaGetErrorString(status)); \
            exit(-1);                                          \
        }                                                      \
    }

#define CUDA_SYNC()                          \
    {                                        \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    }

namespace cuda
{
    void check(int nsteps, int nrows, int ncols, float *devPtr, const std::string &text = "checking!", bool output = false);

    void assignWeights(int, int, float, float, float *);

    cudaError_t assignSinVec(float *hostPtr, size_t len);

    cudaError_t assignCosVec(float *hostPtr, size_t len);

    void assignFFTInOut(int, int, int, cufftComplex *, float *);

    void dataCopy(int, int, int, cufftComplex *, float *, int);

    void executeFilter(int, int, int, cufftComplex *, float *);

    void computeProjectionWeights(int, int, int, int,
                                  float *, float *, float *, float *, float *);

    void drawColumns(int steps, int rows, int cols, int row, float *dst, float *src);

    // void rotate(int steps, int rows, int cols, int step, float *dst, float *src, float *cosVec, float *sinVec);

    void rotate(int steps, int rows, int cols, int step, float *src, float *tmp);

    void generateSlice(int steps, int rows, int cols, float *dst, float *src);

    void backprojection(int steps, int rows, int cols, int step, float dsize, float R, float *dst, float *src, float cosTheta, float sinTheta, int angles);
}

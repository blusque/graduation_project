#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda
{
    void setZeroFunc(int rows, int cols, int steps, float *dataPtr);

    void setConstantFunc(int rows, int cols, int steps, float *dataPtr, float constant);

    void setIdentityFunc(int rows, int cols, int steps, float *dataPtr);

    float getElement(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr);

    void setElement(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr, float num);

    void add(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result);

    void add(int rows, int cols, int setps, float *dataPtr, float num, float *result);

    void minus(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result);

    void dividedByNumber(int rows, int cols, int steps, float *dataPtr, float num, float *result);

    void multiply(int rows, int cols, int steps, float *dataPtr, float value, float *result);

    void multiply(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result);

    void multiplyBroadcast(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result);

    void divide(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result);

    void transposeFunc(int rows, int cols, int steps, float *src, float *dst, size_t dim0, size_t dim1);
}
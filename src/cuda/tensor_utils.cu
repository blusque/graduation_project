#include "cuda/tensor_utils.h"

#define blockSize 128

__global__ void setZeroDevice(int rows, int cols, int steps, float *dataPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        dataPtr[i] = 0.f;
    }
}

__global__ void setConstantDevice(int rows, int cols, int steps, float *dataPtr, float constant)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        dataPtr[i] = constant;
    }
}

__global__ void setIdentityDevice(int rows, int cols, int steps, float *dataPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int matIndex = i % (rows * cols);
        if (matIndex / rows == matIndex % rows)
        {
            dataPtr[i] = 1.f;
        }
    }
}

__global__ void getElementDevice(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr, float &num)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        if (i == rowIndex * colIndex * stepIndex)
        {
            num = dataPtr[i];
            break;
        }
    }
}

__global__ void setElementDevice(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr, float num)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        if (i == rowIndex * colIndex * stepIndex)
        {
            dataPtr[i] = num;
            break;
        }
    }
}

__global__ void addDevice(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = dataPtrL[i] + dataPtrR[i];
        // dataPtrL[i] += dataPtrR[i];
    }
}

__global__ void addDevice(int rows, int cols, int steps, float *dataPtr, float num, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = dataPtr[i] + num;
        // dataPtr[i] += num;
    }
}

__global__ void minusDevice(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = dataPtrL[i] + dataPtrR[i];
        // dataPtrL[i] -= dataPtrR[i];
    }
}

__global__ void dividedByNumberDevice(int rows, int cols, int steps, float *dataPtr, float num, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = num / dataPtr[i];
        // dataPtr[i] = num / dataPtr[i];
    }
}

__global__ void multiplyDevice(int rows, int cols, int steps, float *dataPtr, float value, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = dataPtr[i] * value;
        // dataPtr[i] *= value;
    }
}

__global__ void multiplyDevice(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = dataPtrL[i] * dataPtrR[i];
        // dataPtrL[i] *= dataPtrR[i];
    }
}

__global__ void multiplyBroadcastDevice(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int rIndex = i % (rows * cols);
        result[i] = dataPtrL[i] * dataPtrR[rIndex];
        // dataPtrL[i] *= dataPtrR[i];
    }
}

__global__ void divideDevice(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        result[i] = dataPtrL[i] / dataPtrR[i];
        // dataPtrL[i] /= dataPtrR[i];
    }
}

__device__ void swap(float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

__global__ void transpose01Device(int rows, int cols, int steps, float *src, float *dst)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int j = i / (rows * cols);
        int k = i % (rows * cols) / cols;
        int c = i % cols;
        swap(src[i], dst[k * steps * cols + j * cols + c]);
    }
}

__global__ void transpose12Device(int rows, int cols, int steps, float *src, float *dst)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int c = i / (rows * cols);
        int j = i % (rows * cols) / cols;
        int k = i % cols;
        swap(src[i], dst[c * rows * cols + k * rows + j]);
    }
}

__global__ void transpose02Device(int rows, int cols, int steps, float *src, float *dst)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int j = i / (rows * cols);
        int c = i % (rows * cols) / cols;
        int k = i % cols;
        swap(src[i], dst[k * rows * steps + c * steps + j]);
    }
}

namespace cuda
{
    void setZeroFunc(int rows, int cols, int steps, float *dataPtr)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        setZeroDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtr);
    }

    void setConstantFunc(int rows, int cols, int steps, float *dataPtr, float constant)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        setConstantDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtr, constant);
    }

    void setIdentityFunc(int rows, int cols, int steps, float *dataPtr)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        setIdentityDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtr);
    }

    float getElement(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        float result;
        getElementDevice<<<gridSize, blockSize>>>(rows, cols, steps, rowIndex, colIndex, stepIndex, dataPtr, result);
        return result;
    }

    void setElement(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr, float num)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        setElementDevice<<<gridSize, blockSize>>>(rows, cols, steps, rowIndex, colIndex, stepIndex, dataPtr, num);
    }

    void add(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        addDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void add(int rows, int cols, int steps, float *dataPtr, float num, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        addDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtr, num, result);
    }

    void minus(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        minusDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void dividedByNumber(int rows, int cols, int steps, float *dataPtr, float num, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        dividedByNumberDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtr, num, result);
    }

    void multiply(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        multiplyDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void multiply(int rows, int cols, int steps, float *dataPtr, float value, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        multiplyDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtr, value, result);
    }

    void multiplyBroadcast(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        multiplyBroadcastDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void divide(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        divideDevice<<<gridSize, blockSize>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void transposeFunc(int rows, int cols, int steps, float *src, float *dst, size_t dim0, size_t dim1)
    {
        int gridSize = (steps * rows * cols) / blockSize;
        if ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0))
        {
            transpose01Device<<<gridSize, blockSize>>>(rows, cols, steps, src, dst);
        }
        else if ((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1))
        {
            transpose12Device<<<gridSize, blockSize>>>(rows, cols, steps, src, dst);
        }
        else if ((dim0 == 2 && dim1 == 0) || (dim0 == 0 && dim1 == 2))
        {
            transpose02Device<<<gridSize, blockSize>>>(rows, cols, steps, src, dst);
        }
    }
}

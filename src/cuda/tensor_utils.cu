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
    for (int i = index; i < rows * cols; i += stride)
    {
        result[i] = dataPtrL[i] / dataPtrR[i];
        // dataPtrL[i] /= dataPtrR[i];
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
}

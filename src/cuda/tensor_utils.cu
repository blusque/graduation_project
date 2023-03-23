#include "cuda/tensor_utils.h"

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
        setZeroDevice<<<32, 1024>>>(rows, cols, steps, dataPtr);
    }

    void setConstantFunc(int rows, int cols, int steps, float *dataPtr, float constant)
    {
        setConstantDevice<<<rows, cols>>>(rows, cols, steps, dataPtr, constant);
    }

    void setIdentityFunc(int rows, int cols, int steps, float *dataPtr)
    {
        setIdentityDevice<<<rows, cols>>>(rows, cols, steps, dataPtr);
    }

    float getElement(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr)
    {
        float result;
        getElementDevice<<<rows, cols>>>(rows, cols, steps, rowIndex, colIndex, stepIndex, dataPtr, result);
        return result;
    }

    void setElement(int rows, int cols, int steps, int rowIndex, int colIndex, int stepIndex, float *dataPtr, float num)
    {
        setElementDevice<<<rows, cols>>>(rows, cols, steps, rowIndex, colIndex, stepIndex, dataPtr, num);
    }

    void add(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        addDevice<<<rows, cols>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void add(int rows, int cols, int steps, float *dataPtr, float num, float *result)
    {
        addDevice<<<rows, cols>>>(rows, cols, steps, dataPtr, num, result);
    }

    void minus(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        minusDevice<<<rows, cols>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void dividedByNumber(int rows, int cols, int steps, float *dataPtr, float num, float *result)
    {
        dividedByNumberDevice<<<rows, cols>>>(rows, cols, steps, dataPtr, num, result);
    }

    void multiply(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        multiplyDevice<<<rows, cols>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void multiply(int rows, int cols, int steps, float *dataPtr, float value, float *result)
    {
        multiplyDevice<<<rows, cols>>>(rows, cols, steps, dataPtr, value, result);
    }

    void multiplyBroadcast(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        multiplyBroadcastDevice<<<rows, cols>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }

    void divide(int rows, int cols, int steps, float *dataPtrL, float *dataPtrR, float *result)
    {
        divideDevice<<<rows, cols>>>(rows, cols, steps, dataPtrL, dataPtrR, result);
    }
}

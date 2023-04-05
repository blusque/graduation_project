#include "cuda/fdk_utils.h"

#include <stdio.h>
#include <iostream>

#include <cufft.h>
#include <nvtx3/nvToolsExt.h>
#define PI 3.1416
#define blockSize 128
#define THETA_SIZE 3600

__constant__ float cosVec[THETA_SIZE] = {0.f};
__constant__ float sinVec[THETA_SIZE] = {0.f};

__global__ void
assignWeightsDevice(int rows, int cols, float R, float ds, float *dataPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < rows * cols; i += stride)
    {
        float row = (i / cols - rows / 2 + 1) * ds;
        float col = (i % cols - cols / 2 + 1) * ds;
        dataPtr[i] = R / sqrtf(R * R + float(row * row) + float(col * col));
    }
}

__global__ void assignFFTInOutDevice(int rows, int cols, int steps, cufftComplex *fftInOut, float *dataPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < cols * rows * steps; i += stride)
    {
        fftInOut[i].x = dataPtr[i];
        fftInOut[i].y = 0.f;
    }
}

__global__ void downloadFFTInOutDevice(int rows, int cols, int steps, cufftComplex *fftInOut, float *dataPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < cols * rows * steps; i += stride)
    {
        dataPtr[i] = fftInOut[i].x;
    }
}

__global__ void executeFilterDevice(int rows, int cols, int steps, cufftComplex *fftInOut, float *filter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int filterIndex = (i + cols / 2) % cols;
        // int filterIndex = (i % (rows * cols) / cols + rows / 2) % rows;
        fftInOut[i].x *= fabs(filter[filterIndex]);
        fftInOut[i].y *= fabs(filter[filterIndex]);
    }
}

__global__ void computeProjectionWeightsDevice(int rows, int cols, int steps, float R, float *x, float *y,
                                               float *cosVec, float *sinVec, float *dataPtr)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float epsilon = 1e-6;
    int matSize = rows * cols;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int j = i % matSize;
        int theta = i / matSize;
        float sqrtU_1 = x[j] * cosVec[theta] + y[j] * sinVec[theta] + R;
        dataPtr[i] = R * R / (sqrtU_1 * sqrtU_1 + epsilon * epsilon);
    }
}

__global__ void drawColumnsDevice(int rows, int cols, int steps, int dist, float *dst, float *src)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < steps * cols * cols; i += stride)
    {
        int dstIndex = i;
        int step = i / dist;
        // int srcIndex = step * dist + i % rows * cols;
        int srcIndex = step * dist + i % cols;
        // dst[dstIndex] += src[srcIndex];
        dst[dstIndex] = __fadd_rn(dst[dstIndex], src[srcIndex]);
    }
}

/**
 * @brief bilinear interpolate function on device
 *
 *
 * @param val1 input(x0, y0)
 * @param val2 input(x1, y0)
 * @param val3 input(x0, y1)
 * @param val4 input(x1, y1)
 * @param x0
 * @param x1
 * @param y0
 * @param y1
 * @param x
 * @param y
 * @param result
 * @return
 */
__device__ float interpolate(float val1, float val2, float val3, float val4,
                             float x0, float x1, float y0, float y1, float x, float y)
{
    return val1 * (x1 - x) * (y1 - y) + val2 * (x - x0) * (y1 - y) + val3 * (x1 - x) * (y - y0) + val4 * (x - x0) * (y - y0);
}

__global__ void rotateDevice(int rows, int cols, int steps, float *dst, float *src)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float centerX = rows / 2 + 1;
    float centerY = cols / 2 + 1;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        float x = float((i / cols) % rows);
        float y = float(i % cols);
        int step = i / (rows * cols);
        float cosTheta = -sinVec[step];
        float sinTheta = cosVec[step];
        // float cosTheta = cosVec[step];
        // float sinTheta = sinVec[step];
        float rotateX = __fmaf_rn(__fadd_rn(x, -centerX), cosTheta,
                                   __fmaf_rn(-__fadd_rn(y, -centerY), sinTheta, centerX));
        float rotateY = __fmaf_rn(__fadd_rn(x, -centerX), sinTheta,
                                   __fmaf_rn(__fadd_rn(y, -centerY), cosTheta, centerY));
        // float rotateX = (x - centerX) * cosTheta - (y - centerY) * sinTheta + centerX;
        // float rotateY = (x - centerX) * sinTheta + (y - centerY) * cosTheta + centerY;
        float x0 = floorf(rotateX), x1 = ceilf(rotateX),
              y0 = floorf(rotateY), y1 = ceilf(rotateY);
        if (rotateX >= 0 && rotateY >= 0 && rotateX < rows - 1 && rotateY < cols - 1)
        {
            float val1 = src[step * cols * rows + int(x0) * cols + int(y0)],
                  val2 = src[step * cols * rows + int(x1) * cols + int(y0)],
                  val3 = src[step * cols * rows + int(x0) * cols + int(y1)],
                  val4 = src[step * cols * rows + int(x1) * cols + int(y1)];
            dst[i] = interpolate(val1, val2, val3, val4, x0, x1, y0, y1, rotateX, rotateY);
        }
        else
        {
            dst[i] = dst[i];
        }
    }
}

__global__ void generateSliceDevice(int rows, int cols, int steps, float *dst, float *src)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float steps_1 = __frcp_rn(steps);
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int dstIndex = i % (rows * cols);
        int srcIndex = i;
        // dst[dstIndex] += src[srcIndex] * steps_1;
        dst[dstIndex] = __fmaf_rn(src[srcIndex], steps_1, dst[dstIndex]);
    }
}

__global__ void backprojectionDevice(int rows, int cols, int steps, float dsize, float R, float *dst, float *src,
                                     float cosTheta, float sinTheta, int angleNums, float *count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int centerX = rows / 2 + 1;
    int centerY = cols / 2 + 1;
    int centerZ = steps / 2 + 1;
    for (int i = index; i < rows * cols * steps; i += stride)
    {
        int k1 = (i / cols) % rows;
        int k2 = i % cols;
        int k3 = i / (rows * cols);
        float x = dsize * (k1 - centerX);
        float y = dsize * (k2 - centerY);
        float z = dsize * (k3 - centerZ);
        float U_1 = R / (R + x * sinTheta - y * cosTheta);
        float a = (x * cosTheta + y * sinTheta) * U_1;
        float xx = a / dsize + centerX;
        float x0 = floorf(xx), x1 = ceilf(xx);
        float b = z * U_1;
        float yy = b / dsize + centerY;
        float y0 = floorf(yy), y1 = ceilf(yy);
        if (xx >= 0 && xx < rows && yy >= 0 && yy < cols)
        {
            count[i] = 1.f;
            float val1 = src[int(y0) * cols + int(x0)], val2 = src[int(y1) * cols + int(x0)],
                  val3 = src[int(y0) * cols + int(x1)], val4 = src[int(y1) * cols + int(x1)];
            float temp = interpolate(val1, val2, val3, val4, y0, y1, x0, x1, yy, xx);
            dst[i] += temp * U_1 * U_1 / float(angleNums);
        }
        else
        {
            dst[i] = dst[i];
        }
    }
}

namespace cuda
{
    void check(int nsteps, int nrows, int ncols, float *devPtr, const std::string &text, bool output)
    {
        float *hostPtr = new float[nsteps * nrows * ncols];
        CUDA_CHECK(cudaMemcpy(hostPtr, devPtr, nsteps * nrows * ncols * sizeof(float), cudaMemcpyDeviceToHost));
        int num = 0;
        std::cout << text << std::endl;
        for (int i = 0; i < nsteps * nrows * ncols; i++)
        {
            if (std::abs(hostPtr[i]) > 1e-15)
            {
                num++;
                if (output)
                {
                    std::cout << "position: " << i / (nrows * ncols) << ", " << (i % (nrows * ncols)) / ncols
                              << ", " << (i % (nrows * ncols)) % ncols << '\t';
                    std::cout << "value: " << hostPtr[i] << '\n';
                }
            }
        }
        std::cout << "Total: " << num << " none-zero nums!\n";
        delete[] hostPtr;
    }

    void assignWeights(int rows, int cols, float R, float ds, float *result)
    {
        int gridSize = (rows * cols) / blockSize;
        assignWeightsDevice<<<gridSize, blockSize>>>(rows, cols, R, ds, result);
        // CUDA_SYNC();
    }

    cudaError_t assignSinVec(float *hostPtr, size_t len)
    {
        return cudaMemcpyToSymbol(sinVec, hostPtr, len);
    }

    cudaError_t assignCosVec(float *hostPtr, size_t len)
    {
        return cudaMemcpyToSymbol(cosVec, hostPtr, len);
    }

    void dataCopy(int steps, int rows, int cols, cufftComplex *complexData, float *floatData, int direction)
    {
        int gridSize = (rows * cols * steps) / blockSize;
        if (direction == COPY_C2R)
        {
            downloadFFTInOutDevice<<<gridSize, blockSize>>>(rows, cols, steps, complexData, floatData);
            // CUDA_SYNC();
        }
        else if (direction == COPY_R2C)
        {
            assignFFTInOutDevice<<<gridSize, blockSize>>>(rows, cols, steps, complexData, floatData);
            // CUDA_SYNC();
        }
    }

    void assignFFTInOut(int steps, int rows, int cols, cufftComplex *fftInOut, float *dataPtr)
    {
        int gridSize = (rows * cols * steps) / blockSize;
        assignFFTInOutDevice<<<gridSize, blockSize>>>(rows, cols, steps, fftInOut, dataPtr);
        // CUDA_SYNC();
    }

    void executeFilter(int steps, int rows, int cols, cufftComplex *fftInOut, float *filter)
    {
        int gridSize = (rows * cols * steps) / blockSize;
        executeFilterDevice<<<gridSize, blockSize>>>(rows, cols, steps, fftInOut, filter);
        // CUDA_SYNC();
    }

    void computeProjectionWeights(int steps, int rows, int cols, int R,
                                  float *x, float *y, float *cosVec, float *sinVec, float *result)
    {
        int gridSize = (rows * cols * steps) / blockSize;
        computeProjectionWeightsDevice<<<gridSize, blockSize>>>(rows, cols, steps, R, x, y, cosVec, sinVec, result);
        // CUDA_SYNC();
    }

    void drawColumns(int steps, int rows, int cols, int row, float *dst, float *src)
    {
        nvtxRangePushA(__FUNCTION__);
        nvtxRangePushA("drawColumns");
        int dist = rows * cols;
        int gridSize = (cols * cols * steps) / blockSize;
        float *startPtr = src + row * cols;
        // std::string checkStr = "checking start: ";
        // checkStr += std::to_string(row);
        // std::cout << "src: " << src << " start: " << startPtr << " dst: " << dst << std::endl;
        // check(1, 1, cols, startPtr, checkStr);
        drawColumnsDevice<<<gridSize, blockSize>>>(rows, cols, steps, dist, dst, startPtr);
        // CUDA_SYNC();
        nvtxRangePop();
        nvtxRangePop();
    }

    void rotate(int steps, int rows, int cols, int step, float *src, float *tmp)
    {
        nvtxRangePushA(__FUNCTION__);
        nvtxRangePushA("rotate");
        int gridSize = (rows * cols * steps) / blockSize;
        rotateDevice<<<gridSize, blockSize>>>(rows, cols, steps, tmp, src);
        // CUDA_SYNC();
        cudaMemcpy((void *)src, (void *)tmp, rows * cols * steps * sizeof(float), cudaMemcpyDeviceToDevice);
        nvtxRangePop();
        nvtxRangePop();
    }

    void generateSlice(int steps, int rows, int cols, float *dst, float *src)
    {
        nvtxRangePushA(__FUNCTION__);
        nvtxRangePushA("generateSlice");
        int gridSize = (cols * rows) / blockSize;
        generateSliceDevice<<<gridSize, blockSize>>>(rows, cols, steps, dst, src);
        // CUDA_SYNC();
        nvtxRangePop();
        nvtxRangePop();
    }

    void backprojection(int steps, int rows, int cols, int step, float dsize, float R,
                        float *dst, float *src, float cosTheta, float sinTheta, int angles)
    {
        Tensor count(rows, cols, steps);
        count.setZero();
        std::string checkText = "check count: " + std::to_string(step);
        float *startPtr = src + cols * rows * step;
        backprojectionDevice<<<rows * steps, cols>>>(rows, cols, steps, dsize, R, dst, startPtr, cosTheta, sinTheta, angles, count.getData());
        // CUDA_SYNC();
        check(steps, rows, cols, count.getData(), checkText);
    }
}

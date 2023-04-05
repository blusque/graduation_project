#pragma once

#include <vector>
#include <cublas_v2.h>

namespace cuda
{
    class Tensor
    {
    public:
        Tensor() {}
        Tensor(size_t rowSize, size_t colSize, size_t stepSize);
        Tensor(size_t rowSize, size_t colSize, size_t stepSize, float *data);
        Tensor(const Tensor&);
        Tensor &operator=(const Tensor&);
        
        ~Tensor();

    private:
        size_t _rows;
        size_t _cols;
        size_t _steps;
        size_t _byteSize;
        float *_dataDev; // data pointer on device, which is a row-major Tensor
        int _gpuNums{1};

    public:
        cudaError_t download(float *dataHost);

        // wait for the device to be ready
        cudaError_t synchronize();

        // set the Tensor to be a zero Tensor
        cudaError_t setZero();

        // set all elements of the Tensor to a constant value
        cudaError_t setConstant(float);

        // set the Tensor to be an identity Tensor
        cudaError_t setIdentity();

        __inline__ size_t rows() const { return _rows; }
        __inline__ size_t cols() const { return _cols; }
        __inline__ size_t steps() const { return _steps; }

        float *getData() const { return _dataDev; }

        float get(int stepIndex, int rowIndex, int colIndex) const;
        void set(int stepIndex, int rowIndex, int colIndex, float value);

        Tensor operator*(const Tensor &rValue);

        Tensor operator*(float rValue);

        friend Tensor operator*(float lValue, const Tensor &rValue);

        Tensor operator+(const Tensor &rValue);

        Tensor operator+(float rValue);

        friend Tensor operator+(float lValue, const Tensor &rValue);

        Tensor operator/(const Tensor &rValue);

        Tensor operator/(float rValue);

        friend Tensor operator/(float lValue, const Tensor &rValue);

        Tensor operator-(const Tensor &rValue);

        Tensor operator-(float rValue);

        friend Tensor operator-(float lValue, const Tensor &rValue);

        cublasStatus_t matmul(const Tensor &rValue, int step);

        void Tensor::transpose(int dim0 = 1, int dim1 = 2);
    };
}

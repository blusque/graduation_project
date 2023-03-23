#include "cuda/tensor.h"
#include "cuda/tensor_utils.h"

#include <iostream>

namespace cuda
{
    Tensor::Tensor(size_t rowSize, size_t colSize, size_t stepSize)
    {
        _rows = rowSize;
        _cols = colSize;
        _steps = stepSize;
        _byteSize = sizeof(float) * _rows * _cols * _steps;
        cudaMalloc((void **)&_dataDev, _byteSize);
        setZero();
    }

    Tensor::Tensor(size_t rowSize, size_t colSize, size_t stepSize, float *data)
    {
        _rows = rowSize;
        _cols = colSize;
        _steps = stepSize;
        _byteSize = sizeof(float) * _rows * _cols * _steps;
        cudaMalloc((void **)&_dataDev, _byteSize);
        cudaMemcpy((void *)_dataDev, (void *)data, _byteSize, cudaMemcpyHostToDevice);
        _handles = (cublasHandle_t *)std::malloc(_gpuNums * sizeof(cublasHandle_t));
        for (int i = 0; i < _gpuNums; i++)
        {
            cublasCreate_v2(&(_handles[i]));
        }
    }

    Tensor::Tensor(const Tensor &other)
        : _rows(other._rows)
        , _cols(other._cols)
        , _steps(other._steps)
        , _byteSize(other._byteSize)
        , _handles(other._handles)
        , _gpuNums(other._gpuNums)
    {
        cudaMalloc((void **)&_dataDev, _byteSize);
        cudaMemcpy((void *)_dataDev, (void *)other._dataDev, _byteSize, cudaMemcpyDeviceToDevice);
    }

    Tensor &Tensor::operator=(const Tensor &other)
    {
        _rows = other._rows;
        _cols = other._cols;
        _steps = other._steps;
        _byteSize = other._byteSize;
        _handles = other._handles;
        _gpuNums = other._gpuNums;
        cudaMalloc((void **)&_dataDev, _byteSize);
        cudaMemcpy((void *)_dataDev, (void *)other._dataDev, _byteSize, cudaMemcpyDeviceToDevice);
        return *this;
    }

    Tensor::~Tensor()
    {
        cudaFree(_dataDev);
        _dataDev = nullptr;
    }

    cudaError_t Tensor::download(float *dataHost)
    {
        return cudaMemcpy((void *)dataHost, (void *)_dataDev, _byteSize, cudaMemcpyDeviceToHost);
    }

    cudaError_t Tensor::synchronize()
    {
        return cudaSuccess;
    }

    cudaError_t Tensor::setZero()
    {
        setZeroFunc(_rows, _cols, _steps, _dataDev);
        return cudaSuccess;
    }

    cudaError_t Tensor::setConstant(float constant)
    {
        setConstantFunc(_rows, _cols, _steps, _dataDev, constant);
        return cudaSuccess;
    }

    cudaError_t Tensor::setIdentity()
    {
        setIdentityFunc(_rows, _cols, _steps, _dataDev);
        return cudaSuccess;
    }

    float Tensor::get(int stepIndex, int rowIndex, int colIndex) const
    {
        float num = 0.f;
        num = getElement(_rows, _cols, _steps, rowIndex, colIndex, stepIndex, _dataDev);
        return num;
    }

    void Tensor::set(int stepIndex, int rowIndex, int colIndex, float value)
    {
        setElement(_rows, _cols, _steps, rowIndex, colIndex, stepIndex, _dataDev, value);
    }

    Tensor Tensor::operator*(const Tensor& rValue)
    {
        if (_steps != rValue._steps && rValue._steps == 1)
        {
            int resultSteps = std::max(_steps, rValue._steps);
            Tensor result(_rows, _cols, resultSteps);
            multiplyBroadcast(_rows, _cols, _steps, _dataDev, rValue._dataDev, result._dataDev);
            return result;
        }
        else if (_steps != rValue._steps && _steps == 1)
        {
            int resultSteps = std::max(_steps, rValue._steps);
            Tensor result(_rows, _cols, resultSteps);
            multiplyBroadcast(_rows, _cols, rValue._steps, rValue._dataDev, _dataDev, result._dataDev);
            return result;
        }
        else if (_steps == rValue._steps)
        {
            Tensor result(_rows, _cols, _steps);
            multiply(_rows, _cols, _steps, _dataDev, rValue._dataDev, result._dataDev);
            return result;
        }
        
    }

    Tensor Tensor::operator*(float rValue)
    {
        Tensor result(_rows, _cols, _steps);
        multiply(_rows, _cols, _steps, _dataDev, rValue, result._dataDev);
        return result;
    }

    Tensor operator*(float lValue, const Tensor &rValue)
    {
        Tensor result(rValue._rows, rValue._cols, rValue._steps);
        multiply(rValue._rows, rValue._cols, rValue._steps, rValue._dataDev, lValue, result._dataDev);
        return result;
    }

    Tensor Tensor::operator+(const Tensor &rValue)
    {
        Tensor result(_rows, _cols, _steps);
        add(_rows, _cols, _steps, _dataDev, rValue._dataDev, result._dataDev);
        return result;
    }

    Tensor Tensor::operator+(float rValue)
    {
        Tensor result(_rows, _cols, _steps);
        add(_rows, _cols, _steps, _dataDev, rValue, result._dataDev);
        return result;
    }

    Tensor operator+(float lValue, const Tensor &rValue)
    {
        Tensor result(rValue._rows, rValue._cols, rValue._steps);
        add(rValue._rows, rValue._cols, rValue._steps, rValue._dataDev, lValue, result._dataDev);
        return result;
    }

    Tensor Tensor::operator/(const Tensor &rValue)
    {
        Tensor result(_rows, _cols, _steps);
        divide(_rows, _cols, _steps, _dataDev, rValue._dataDev, result._dataDev);
        return result;
    }

    Tensor Tensor::operator/(float rValue)
    {
        float rValue1 = 1 / rValue;
        Tensor result(_rows, _cols, _steps);
        multiply(_rows, _cols, _steps, _dataDev, rValue1, result._dataDev);
        return result;
    }

    Tensor operator/(float lValue, const Tensor &rValue)
    {
        Tensor result(rValue._rows, rValue._cols, rValue._steps);
        dividedByNumber(rValue._rows, rValue._cols, rValue._steps, rValue._dataDev, lValue, result._dataDev);
        return result;
    }

    Tensor Tensor::operator-(const Tensor &rValue)
    {
        Tensor result(_rows, _cols, _steps);
        minus(_rows, _cols, _steps, _dataDev, rValue._dataDev, result._dataDev);
        return result;
    }

    Tensor Tensor::operator-(float rValue)
    {
        Tensor result(_rows, _cols, _steps);
        add(_rows, _cols, _steps, _dataDev, -rValue, result._dataDev);
        return result;
    }

    Tensor operator-(float lValue, const Tensor &rValue)
    {
        Tensor result(rValue._rows, rValue._cols, rValue._steps);
        add(rValue._rows, rValue._cols, rValue._steps, rValue._dataDev, -lValue, result._dataDev);
        return result;
    }

    cublasStatus_t Tensor::matmul(const Tensor &rValue, int step)
    {
        float alpha = 1.f;
        float beta = 0.f;
        auto status = cublasSgemm_v2(
            _handles[0],
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            _cols,
            rValue._rows,
            _rows,
            &alpha,
            rValue._dataDev,
            rValue._cols,
            _dataDev,
            _cols,
            &beta,
            _dataDev,
            _cols);
        return status;
    }
}
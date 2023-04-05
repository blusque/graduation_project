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
        cudaError_t state = cudaMalloc((void **)&_dataDev, _byteSize);
        if (state != cudaSuccess)
        {
            std::cout << "cudaMalloc failed! Error Num: " << state << std::endl;
        }
        std::cout << "tensor allocated!" << std::endl;
        setZero();
    }

    Tensor::Tensor(size_t rowSize, size_t colSize, size_t stepSize, float *data)
    {
        _rows = rowSize;
        _cols = colSize;
        _steps = stepSize;
        _byteSize = sizeof(float) * _rows * _cols * _steps;
        cudaError_t state = cudaMalloc((void **)&_dataDev, _byteSize);
        if (state != cudaSuccess)
        {
            std::cout << "cudaMalloc failed! Error Num: " << state << std::endl;
        }
        cudaMemcpy((void *)_dataDev, (void *)data, _byteSize, cudaMemcpyHostToDevice);
    }

    Tensor::Tensor(const Tensor &other)
        : _rows(other._rows)
        , _cols(other._cols)
        , _steps(other._steps)
        , _byteSize(other._byteSize)
        , _gpuNums(other._gpuNums)
    {
        std::cout << "Tensor copying!" << std::endl;
        if (_dataDev != nullptr)
            cudaFree(_dataDev);
        cudaError_t state = cudaMalloc((void **)&_dataDev, _byteSize);
        if (state != cudaSuccess)
        {
            std::cout << "cudaMalloc failed! Error Num: " << state << std::endl;
        }
        cudaMemcpy((void *)_dataDev, (void *)other._dataDev, _byteSize, cudaMemcpyDeviceToDevice);
        std::cout << "Tensor copied!" << std::endl;
    }

    Tensor &Tensor::operator=(const Tensor &other)
    {
        std::cout << "Tensor assigning!" << std::endl;
        if (this != &other && _dataDev != other._dataDev)
        {
            _rows = other._rows;
            _cols = other._cols;
            _steps = other._steps;
            _byteSize = other._byteSize;
            _gpuNums = other._gpuNums;
            cudaError_t state;
            if (_dataDev != nullptr)
            {
                state = cudaFree(_dataDev);
                if (state != cudaSuccess)
                {
                    std::cout << "cudaMalloc failed! Error Num: " << state << std::endl;
                }
                std::cout << "Tensor freed!" << std::endl;
            }
            state = cudaMalloc((void **)&_dataDev, _byteSize);
            if (state != cudaSuccess)
            {
                std::cout << "cudaMalloc failed! Error Num: " << state << std::endl;
            }
            std::cout << "Tensor reallocated!" << std::endl;
            cudaMemcpy((void *)_dataDev, (void *)other._dataDev, _byteSize, cudaMemcpyDeviceToDevice);
        }
        std::cout << "Tensor assigned!" << std::endl; 
        return *this;
    }

    Tensor::~Tensor()
    {
        if (_dataDev != nullptr)
        {
            cudaFree(_dataDev);
            _dataDev = nullptr;
        }
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
        cublasHandle_t handle;
        cublasCreate_v2(&handle);
        float alpha = 1.f;
        float beta = 0.f;
        auto status = cublasSgemm_v2(
            handle,
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

    void Tensor::transpose(int dim0, int dim1)
    {
        transposeFunc(_rows, _cols, _steps, _dataDev, _dataDev, dim0, dim1);
        if ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0))
        {
            size_t tmp = _steps;
            _steps = _rows;
            _rows = tmp;
        }
        else if ((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1))
        {
            size_t tmp = _cols;
            _cols = _rows;
            _rows = tmp;
        }
        else if ((dim0 == 2 && dim1 == 0) || (dim0 == 0 && dim1 == 2))
        {
            size_t tmp = _steps;
            _steps = _cols;
            _cols = tmp;
        }
    }
}
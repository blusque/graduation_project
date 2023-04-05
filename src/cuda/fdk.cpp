#include "cuda/fdk.h"
#include "cuda/fdk_utils.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#define NDEBUG
#include <assert.h>
#ifdef _UNIX
#include <cmath>
#elif defined _WIN32
#include <corecrt_math_defines.h>
#endif
#include <chrono>
#include <stdlib.h>
#include <malloc.h>

using std::string;
using std::vector;

namespace cuda
{
    FDK::FDK(float R, float ds, const string &filterType, const vectorFPtr &projections, int sizeX, int sizeY)
        : _R(R)
        , _detectorSize(ds)
        , _filterType(filterType)
        , _projections(sizeX, sizeY, projections.size())
        , _projectionWeights(sizeX, sizeY, projections.size())
        , _weights(sizeX, sizeY, 1)
    {
        int layerSize = sizeX * sizeY;
        int nrows = sizeX;
        int ncols = sizeY;
        int nsteps = projections.size();
        for (int i = 0; i < projections.size(); i++)
        {
            int offset = layerSize * i;
            float *dstPtr = this->_projections.getData() + offset;
            cudaMemcpy((void *)dstPtr, (void *)projections[i], layerSize * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMalloc((void **)&_fftInOut, sizeof(cufftComplex) * ncols * nrows * nsteps);
        cudaMalloc((void **)&_filter, nrows * sizeof(float));
    }

    FDK::~FDK()
    {
        cudaFree(this->_filter);
        this->_filter = nullptr;
        cudaFree(this->_fftInOut);
        this->_fftInOut = nullptr;
    }

    void FDK::linspace(float start, float end, int steps, float *vec)
    {
        try
        {
            int len = _msize(vec) / sizeof(float);
            if (len != steps)
            {
                throw std::exception("Len of vec should be the same size as steps");
            }
        }
        catch (std::exception &e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            exit(-1);
        }
        float stride = (end - start) / float(steps);
        float now = start;
        for (int i = 0; i < steps; i++)
        {
            vec[i] = now;
            now += stride;
        }
    }

    void FDK::initialize()
    {
        // _projections.transpose();
        // initialize filter
        auto nrows = _projections.rows();
        auto ncols = _projections.cols();
        auto nsteps = _projections.steps();

        float start = 0.f;
        float end = 0.f;
        if (nrows % 2 == 0)
        {
            start = -float(ncols / 2);
            end = float(ncols / 2);
        }
        else
        {
            start = -float(ncols / 2);
            end = float(ncols / 2 + 1);
        }
        std::cout << "filter type: " << _filterType << std::endl;
        float *filterHost = new float[ncols];
        std::cout << "filterHost: " << filterHost << std::endl;
        if (_filterType == "r-l")
        {
            linspace(start, end, ncols, filterHost);
            for (int i = 0; i < ncols; i++)
            {
                filterHost[i] /= std::max(std::abs(start), std::abs(end));
            }

            std::cout << "filter size: " << ncols << std::endl;
        }
        else if (_filterType == "s-l")
        {
            // s-l filter
        }
        cudaMemcpy((void *)_filter, (void *)filterHost, ncols * sizeof(float), cudaMemcpyHostToDevice);
        delete[] filterHost;
        filterHost = nullptr;
        std::cout << "filter initialized!" << std::endl;

        // initialize pre-weights
        // w(x, y) = r / sqrt(r ** 2 + x ** 2 + y ** 2)
        _weights.setZero();
        assignWeights(nrows, ncols, _R, _detectorSize, _weights.getData());
        std::cout << "weights initialized!" << std::endl;

        // initialize projection weights
        // p_w(theta, x, y) = r ** 2 / [(r + xcos(theta) + ysin(theta)) ** 2 + epsilon ** 2]
        float *cosVecHost, *sinVecHost;
        cosVecHost = (float *)malloc(nsteps * sizeof(float));
        sinVecHost = (float *)malloc(nsteps * sizeof(float));
        float *thetaVec = (float *)malloc(nsteps * sizeof(float));
        linspace(0, 2 * M_PI, nsteps, thetaVec);
        for (auto i = 0; i < nsteps; i++)
        {
            cosVecHost[i] = std::cos(thetaVec[i]);
            sinVecHost[i] = std::sin(thetaVec[i]);
        }
        // cudaMemcpy(_cosVec, cosVecHost, sizeof(float) * nsteps, cudaMemcpyHostToDevice);
        // cudaMemcpy(_sinVec, sinVecHost, sizeof(float) * nsteps, cudaMemcpyHostToDevice);
        CUDA_CHECK(assignCosVec(cosVecHost, nsteps * sizeof(float)));
        CUDA_CHECK(assignSinVec(sinVecHost, nsteps * sizeof(float)));
        free(thetaVec);
        thetaVec = nullptr;

        float *xHost, *yHost;
        xHost = (float *)malloc(sizeof(float) * nrows * ncols);
        yHost = (float *)malloc(sizeof(float) * nrows * ncols);
        for (int i = 0; i < nrows; i++)
        {
            float x = i - int(nrows) / 2 + 1;
            for (int j = 0; j < ncols; j++)
            {
                float y = j - int(ncols) / 2 + 1;
                xHost[i * ncols + j] = x;
                yHost[i * ncols + j] = y;
            }
        }

        float *projectionsWeightHost = new float[nsteps * nrows * ncols];
        for (int k = 0; k < nsteps; k++)
        {
            float cosTheta = cosVecHost[k], sinTheta = sinVecHost[k];
            for (int i = 0; i < nrows; i++)
            {
                for (int j = 0; j < ncols; j++)
                {
                    float x = xHost[i * ncols + j] * _detectorSize, y = yHost[i * ncols + j] * _detectorSize;
                    float sqrtU_1 = (_R + x * sinTheta - y * cosTheta);
                    projectionsWeightHost[k * nrows * ncols + i * ncols + j] = (_R * _R) / (sqrtU_1 * sqrtU_1 + 1e-6 * 1e-6);
                }
            }
        }
        cudaMemcpy((void *)_projectionWeights.getData(), (void *)projectionsWeightHost,
                   nsteps * ncols * nrows * sizeof(float), cudaMemcpyHostToDevice);
        std::cout << "projection weights initialized!" << std::endl;
        free(xHost);
        xHost = nullptr;
        free(yHost);
        yHost = nullptr;
        free(cosVecHost);
        cosVecHost = nullptr;
        free(sinVecHost);
        sinVecHost = nullptr;
        delete[] projectionsWeightHost;
        projectionsWeightHost = nullptr;

        // initialize FFTW plan
        int n[1] = {ncols};
        int inembed[1] = {nrows * ncols * nsteps};
        int onembed[1] = {nrows * ncols * nsteps};
        cufftPlanMany(&_plan, 1, n, inembed, 1, ncols,
                      onembed, 1, ncols, CUFFT_C2C, nrows * nsteps);
        std::cout << "fft plan initialized!" << std::endl;
    }

    void FDK::weighting()
    {
        // multiply every projected data with corresponding weight
        _projections = _projections * _weights;
        std::cout << "weighted!" << std::endl;
    }

    void FDK::filtering()
    {
        auto nrows = _projections.rows();
        auto ncols = _projections.cols();
        auto nsteps = _projections.steps();
        int offset = 0;
        dataCopy(nsteps, nrows, ncols, _fftInOut, _projections.getData(), COPY_R2C);
        cufftExecC2C(_plan, _fftInOut, _fftInOut, CUFFT_FORWARD);
        executeFilter(nsteps, nrows, ncols, _fftInOut, _filter);
        cufftExecC2C(_plan, _fftInOut, _fftInOut, CUFFT_INVERSE);
        dataCopy(nsteps, nrows, ncols, _fftInOut, _projections.getData(), COPY_C2R);
        _projections = _projections / float(ncols) * 2.f;
        std::cout << "filtered!" << std::endl;
    }

#include <nvtx3/nvToolsExt.h>
    Tensor FDK::backprojecting()
    {
        nvtxRangePushA(__FUNCTION__);
        auto nrows = _projections.rows();
        auto ncols = _projections.cols();
        auto nsteps = _projections.steps();

        // backprojection weighting
        _projections = _projections * _projectionWeights;
        std::cout << "backprojection weighted!" << std::endl;

        // backprojection
        Tensor result(ncols, ncols, nrows);

        // float *cosVecHost = new float[nsteps];
        // float *sinVecHost = new float[nsteps];
        // cudaMemcpy((void *)cosVecHost, (void *)_cosVec, nsteps * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy((void *)sinVecHost, (void *)_sinVec, nsteps * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int step = 0; step < nsteps; step++)
        // {
        //     // std::cout << "\rprogress: " << std::setw(6) << std::setprecision(4) << float(step + 1) / float(nsteps) * 100.f << "%";
        //     backprojection(nrows, nrows, ncols, step, _detectorSize, _R, result.getData(),
        //         _projections.getData(), cosVecHost[step], sinVecHost[step], nsteps);
        // }
        // delete [] cosVecHost;
        // delete [] sinVecHost;

        int offset = 0;
        int stride = ncols * ncols;
        std::cout << "using policy: bilinear" << std::endl;
        Tensor slice(ncols, ncols, 1);
        Tensor drawnCols(ncols, ncols, nsteps);
        Tensor temp(nrows, ncols, nsteps);
        // check(nsteps, nrows, ncols, _projections.getData(), "Checking Porjections!");
        std::string checkTextBase = "Checking result: ";
        std::string checkText;
        auto start = std::chrono::steady_clock::now();
        for (int row = 0; row < nrows; row++)
        {
            std::cout << "\rprogress: " << std::setw(6) << std::setprecision(4) << float(row + 1) / float(nrows) * 100.f << "%";
            // generate slice[row]
            nvtxRangePushA("slice setZero");
            slice.setZero();
            nvtxRangePop();
            nvtxRangePushA("draw setZero");
            drawnCols.setZero();
            nvtxRangePop();
            nvtxRangePushA("temp setZero");
            temp.setZero();
            nvtxRangePop();
            // checkText = checkTextBase + std::to_string(row);
            drawColumns(nsteps, nrows, ncols, row, drawnCols.getData(), _projections.getData());
            // check(nsteps, ncols, ncols, drawnCols.getData(), checkText);
            rotate(nsteps, nrows, ncols, row, drawnCols.getData(), temp.getData());
            generateSlice(nsteps, ncols, ncols, slice.getData(), drawnCols.getData());
            cudaMemcpy((result.getData() + offset), slice.getData(), stride * sizeof(float), cudaMemcpyDeviceToDevice);
            offset += stride;
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << '\n';
        // check(nsteps, nrows, ncols, _projections.getData(), "Checking Porjections 2!");
        result = result / _detectorSize;
        std::cout << "over!" << std::endl;
        // check(nrows, ncols, ncols, result.getData(), "Checking result!");
        std::cout << "result size: " << result.steps() << ", " << result.rows() << ", " << result.cols() << std::endl;
        std::cout << "back projection time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
        nvtxRangePop();
        return result;
    }
}

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
    FDK::FDK(float R, const string &filterType, const vectorFPtr &projections, int sizeX, int sizeY)
    {
        this->_R = R;
        this->_filterType = filterType;
        this->_projections = Tensor(sizeX, sizeY, projections.size());
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
        this->_projectionWeights = Tensor(sizeX, sizeY, projections.size());
        this->_weights = Tensor(nrows, ncols, 1);
        cudaMalloc((void **)&_fftInOut, sizeof(float) * ncols * nrows * nsteps);
        cudaMalloc((void **)&_filter, ncols * sizeof(float));
        cudaMalloc((void **)&_sinVec, nsteps * sizeof(float));
        cudaMalloc((void **)&_cosVec, nsteps * sizeof(float));
    }

    FDK::FDK(FDK &&other)
        : _R(std::move(other._R))
        , _filterType(std::move(other._filterType))
        , _projections(std::move(other._projections))
        , _plan(std::move(other._plan))
        , _weights(std::move(other._weights))
        , _projectionWeights(std::move(other._projectionWeights))
    {
        this->_filter = other._filter;
        other._filter = nullptr;
        this->_cosVec = other._cosVec;
        other._cosVec = nullptr;
        this->_sinVec = other._sinVec;
        other._sinVec = nullptr;
        this->_fftInOut = other._fftInOut;
        other._fftInOut = nullptr;
    }

    FDK &FDK::operator=(FDK &&other)
    {
        this->_R = std::move(other._R);
        this->_filterType = std::move(other._filterType);
        this->_projections = std::move(other._projections);
        this->_plan = std::move(other._plan);
        this->_weights = std::move(other._weights);
        this->_projectionWeights = std::move(other._projectionWeights);
        this->_filter = other._filter;
        other._filter = nullptr;
        this->_cosVec = other._cosVec;
        other._cosVec = nullptr;
        this->_sinVec = other._sinVec;
        other._sinVec = nullptr;
        this->_fftInOut = other._fftInOut;
        other._fftInOut = nullptr;
        return *this;
    }

    FDK::~FDK()
    {
        cudaFree(this->_filter);
        this->_filter = nullptr;
        cudaFree(this->_cosVec);
        this->_cosVec = nullptr;
        cudaFree(this->_sinVec);
        this->_sinVec = nullptr;
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
        delete [] filterHost;
        filterHost = nullptr;
        std::cout << "filter initialized!" << std::endl;

        // initialize pre-weights
        // w(x, y) = r / sqrt(r ** 2 + x ** 2 + y ** 2)
        _weights.setZero();
        assignWeights(nrows, ncols, _R, _weights.getData());
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
        cudaMemcpy(_cosVec, cosVecHost, sizeof(float) * nsteps, cudaMemcpyHostToDevice);
        cudaMemcpy(_sinVec, sinVecHost, sizeof(float) * nsteps, cudaMemcpyHostToDevice);
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
                    float x = xHost[i * ncols + j] / 64.f, y = yHost[i * ncols + j] / 64.f;
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
        delete [] projectionsWeightHost;
        projectionsWeightHost = nullptr;

        // initialize FFTW plan
        cudaFree(_fftInOut);
        cudaMalloc((void **)&_fftInOut, sizeof(float) * ncols * nrows * nsteps);
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

    Tensor FDK::backprojecting()
    {
        auto nrows = _projections.rows();
        auto ncols = _projections.cols();
        auto nsteps = _projections.steps();

        // backprojection weighting
        _projections = _projections * _projectionWeights;
        std::cout << "backprojection weighted!" << std::endl;

        // backprojection
        Tensor result(ncols, ncols, nrows);
        int offset = 0;
        int stride = ncols * ncols;
        std::cout << "using policy: bilinear" << std::endl;
        Tensor slice(ncols, ncols, 1);
        Tensor drawnCols(ncols, ncols, nsteps);
        // float *cosVecHost = new float[nsteps];
        // float *sinVecHost = new float[nsteps];
        // float *resultHost = new float[nrows * nrows * ncols];
        // float *projHost = new float[nrows * ncols];
        // float *projHostAll = new float[nrows * ncols * nsteps];
        // cudaMemcpy((void *)cosVecHost, (void *)_cosVec, nsteps * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy((void *)sinVecHost, (void *)_sinVec, nsteps * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy((void *)projHostAll, (void *)_projections.getData(), nsteps * nrows * ncols * sizeof(float), cudaMemcpyDeviceToHost);
        // check(nsteps, nrows, ncols, projHostAll, "checking projection all!");
        // for (int step = 0; step < nsteps; step++)
        // {
        //     std::cout << "step: " << step << '\t';
        //     backprojection(nrows, nrows, ncols, step, 1.f, _R, result.getData(), 
        //         _projections.getData(), cosVecHost[step], sinVecHost[step], nsteps);
        //     // cudaMemcpy((void *)projHostAll, (void *)_projections.getData(), nsteps * nrows * ncols * sizeof(float), cudaMemcpyDeviceToHost);
        //     // check(nsteps, nrows, ncols, projHostAll, "checking projection!");
        // }
        // cudaMemcpy((void *)projHostAll, (void *)_projections.getData(), nsteps * nrows * ncols * sizeof(float), cudaMemcpyDeviceToHost);
        // check(nsteps, nrows, ncols, projHostAll, "checking projection all!");
        // cudaMemcpy((void *)resultHost, (void *)result.getData(), nrows * nrows * ncols * sizeof(float), cudaMemcpyDeviceToHost);
        // check(nrows, nrows, ncols, resultHost, "checking result host!");
        // delete [] cosVecHost;
        // delete [] sinVecHost;
        // delete [] resultHost;
        // delete [] projHost;
        // delete [] projHostAll;
        check(nsteps, nrows, ncols, _projections.getData(), "Checking Porjections!");
        std::string checkTextBase = "Checking result: ";
        std::string checkText;
        for (int row = 0; row < nrows; row++)
        {
            std::cout << "\rprogress: " << std::setw(6) << std::setprecision(4) << float(row + 1) / float(nrows) * 100.f << "%";
            // generate slice[row]
            slice.setZero();
            drawnCols.setZero();
            // checkText = checkTextBase + std::to_string(row);
            drawColumns(nsteps, nrows, ncols, row, drawnCols.getData(), _projections.getData());
            // check(nsteps, ncols, ncols, drawnCols.getData(), checkText);
            rotate(nsteps, nrows, ncols, row, drawnCols.getData(), _cosVec, _sinVec);
            generateSlice(nsteps, ncols, ncols, slice.getData(), drawnCols.getData());
            cudaMemcpy((result.getData() + offset), slice.getData(), stride * sizeof(float), cudaMemcpyDeviceToDevice);
            offset += stride;
        }
        // check(nsteps, nrows, ncols, _projections.getData(), "Checking Porjections 2!");
        result = result * 64.f;
        std::cout << "\nover!" << std::endl;
        check(nrows, ncols, ncols, result.getData(), "Checking result!");
        std::cout << "result size: " << result.steps() << ", " << result.rows() << ", " << result.cols() << std::endl;
        return result;
    }
}

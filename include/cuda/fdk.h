#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda/tensor.h"

/**
 * @brief
 *
 */
namespace cuda
{
    void check(int nsteps, int nrows, int ncols, float *devPtr, const std::string &text);
    
    class FDK
    {
    public:
        using string = std::string;
        using vector = std::vector<float>;
        using vectorFPtr = std::vector<float *>;

    public:
        FDK() = delete;
        explicit FDK(float R, const string &filter, const vectorFPtr &projections, int sizeX, int sizeY);
        FDK(const FDK &other) = delete;
        FDK &operator=(const FDK &other) = delete;
        FDK(FDK &&other);
        FDK &operator=(FDK &&other);

        ~FDK();

    // private:
    //     static __constant__ float filter[1024]; // filter in constant memory on device
    //     static __constant__ float cosVec[7200]; // cos vector on constant memory on device
    //     static __constant__ float sinVec[7200]; // sin vector on constant memory on device

    private:
        float _R{100.f};                  // radius of the detector circle
        string _filterType{"r-l"};        // filter name (default 'r-l', optional 's-l')
        int _filterLength;                // length of the filter
        Tensor _projections;             // projected images (in size of degrees)
        Tensor _weights;                  // weights
        Tensor _projectionWeights;       // projection weights
        int _rows;
        int _cols;
        int _steps;                       // number of steps
        float *_filter;
        float *_cosVec;
        float *_sinVec;
        cufftHandle _plan;                // the fft/ifft plan
        cufftComplex *_fftInOut;          // the data array

    public:
        void initialize();

        void weighting();

        void filtering();

        Tensor backprojecting();

        // static float *filter();

        // static float *cosVec();

        // static float *sinVec();

    private:
        void linspace(float start, float end, int steps, float *vec);
    };
}

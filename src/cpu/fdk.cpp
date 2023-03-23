#include "cpu/fdk.h"

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
#include <windows.h>

using std::string;
using std::vector;

FDK::FDK(float R, const string &filter, const vectorM &projections)
{
    this->R = R;
    this->filterType = filter;
    this->projections = projections;
    fftInOut = nullptr;
}

FDK::FDK(FDK &&other)
    : R(std::move(other.R)), filterType(std::move(other.filterType)), projections(std::move(other.projections)), forwardPlan(std::move(other.forwardPlan)), backwardPlan(std::move(other.backwardPlan)), filter(std::move(other.filter)), weights(std::move(other.weights)), projectionWeights(std::move(other.projectionWeights)), cosVec(std::move(other.cosVec)), sinVec(std::move(other.sinVec))
{
    this->fftInOut = other.fftInOut;
    other.fftInOut = nullptr;
}

FDK &FDK::operator=(FDK &&other)
{
    this->R = std::move(other.R);
    this->filterType = std::move(other.filterType);
    this->projections = std::move(other.projections);
    this->forwardPlan = std::move(other.forwardPlan);
    this->backwardPlan = std::move(other.backwardPlan);
    this->filter = std::move(other.filter);
    this->weights = std::move(other.weights);
    this->projectionWeights = std::move(other.projectionWeights);
    this->cosVec = std::move(other.cosVec);
    this->sinVec = std::move(other.sinVec);
    this->fftInOut = other.fftInOut;
    other.fftInOut = nullptr;
    return *this;
}

FDK::~FDK()
{
    fftwf_free(fftInOut);
    fftInOut = nullptr;
}

void FDK::linspace(float start, float end, int steps, vector &vec)
{
    vec.resize(steps);
    float stride = (end - start) / float(steps);
    float now = start;
    for (int i = 0; i < steps; i++)
    {
        vec[i] = now;
        now += stride;
    }
}

Matrix FDK::add(float l, const Matrix &r)
{
    auto nrows = r.rows();
    auto ncols = r.cols();
    Matrix result(r);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            result(i, j) += l;
        }
    }
    return result;
}

Matrix FDK::add(const Matrix &l, float r)
{
    return add(r, l);
}

Matrix FDK::multiply(const Matrix &l, const Matrix &r)
{
    try
    {
        if (l.rows() != r.rows() || l.cols() != r.rows())
        {
            throw std::runtime_error("RuntimeError: You should keep l.rows() == r.rows() and l.cols() == r.cols() when you use multiply, but now they are not equal.");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    auto nrows = r.rows();
    auto ncols = r.cols();
    Matrix result(r);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            result(i, j) *= l(i, j);
        }
    }
    return result;
}

Matrix FDK::divide(float l, const Matrix &r)
{
    auto nrows = r.rows();
    auto ncols = r.cols();
    Matrix result(Matrix::Zero(nrows, ncols));
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            result(i, j) = l / r(i, j);
        }
    }
    return result;
}

Matrix FDK::rotate(const Matrix &input, int step, const string &interp = "bilinear")
{
    using Eigen::placeholders::all;
    using Eigen::placeholders::last;
    int steps = projections.size() - 1;
    // special degree: 0, 180, 270, 90
    if (step == 0)
    {
        return input;
    }
    else if (step == steps / 2)
    {
        return input(Eigen::seq(last, 0, Eigen::fix<-1>), all);
    }
    else if (step == steps * 3 / 4)
    {
        return input.transpose();
    }
    else if (step == steps / 4)
    {
        return input.transpose()(all, Eigen::seq(last, 0, Eigen::fix<-1>));
    }
    Matrix result(Matrix::Zero(input.rows(), input.cols()));
    int centerX = input.rows() / 2 - 1;
    int centerY = input.cols() / 2 - 1;
    for (int i = 0; i < result.rows(); i++)
    {
        for (int j = 0; j < result.cols(); j++)
        {
            float s = float(i - centerX) * cosVec[step] - float(j - centerY) * sinVec[step] + float(centerX);
            float t = float(i - centerX) * sinVec[step] + float(j - centerY) * cosVec[step] + float(centerY);
            if (s >= 0 && s < input.rows() - 1 && t >= 0 && t < input.cols() - 1)
            {
                result(i, j) = interpolate(input, s, t, interp);
            }
            else
            {
                result(i, j) = 0.f;
            }
        }
    }
    return result;
}

float FDK::interpolate(const Matrix &input, float x, float y, const string &interp = "bilinear")
{
    if (interp == "nearest")
    {
        return input(int(x + 0.5), int(y + 0.5));
    }
    else if (interp == "bilinear")
    {
        int x0 = std::floor(x), x1 = std::ceil(x), y0 = std::floor(y), y1 = std::ceil(y);
        return input(x0, y0) * (x1 - x) * (y1 - y) + input(x0, y1) * (x1 - x) * (y - y0) + input(x1, y0) * (x - x0) * (y1 - y) + input(x1, y1) * (x - x0) * (y - y0);
    }
    else if (interp == "bicubic")
    {
        // bicubic interpolation
    }
}

void FDK::initialize()
{
    // initialize filter
    auto nrows = projections[0].rows();
    auto ncols = projections[0].cols();
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
    std::cout << "filter type: " << filterType << std::endl;
    if (filterType == "r-l")
    {
        linspace(start, end, ncols, filter);
        for (auto &&num : filter)
        {
            num /= max(start, end);
        }
        std::cout << "filter size: " << filter.size() << std::endl;
    }
    else if (filterType == "s-l")
    {
        // s-l filter
    }
    std::cout << "filter initialized!" << std::endl;

    // initialize pre-weights
    // w(x, y) = r / sqrt(r ** 2 + x ** 2 + y ** 2)
    weights.resize(nrows, ncols);
    for (auto i = 0; i < nrows; i++)
    {
        for (auto j = 0; j < ncols; j++)
        {
            weights(i, j) = R / std::sqrt(R * R + std::pow(float(i - nrows / 2 + 1) / 64.f, 2) + std::pow(float(j - ncols / 2 + 1) / 64.f, 2));
        }
    }
    std::cout << "weights initialized!" << std::endl;

    // initialize projection weights
    // p_w(theta, x, y) = r ** 2 / [(r + xcos(theta) + ysin(theta)) ** 2 + epsilon ** 2]
    auto steps = projections.size();
    projectionWeights.resize(steps);
    cosVec.resize(steps);
    sinVec.resize(steps);
    vector thetaVec;
    linspace(0, 2 * M_PI, steps, thetaVec);
    Matrix mat(nrows, ncols);
    for (auto i = 0; i < steps; i++)
    {
        cosVec[i] = std::cos(thetaVec[i]);
        sinVec[i] = std::sin(thetaVec[i]);
    }
    float epsilon = 1e-6f;
    auto xOnes = Matrix::Constant(nrows, 1, 1);
    Matrix xBase(1, ncols);
    xBase = Eigen::ArrayXf::LinSpaced(ncols, -ncols / 2 + 1, ncols / 2).transpose();
    auto yOnes = Matrix::Constant(1, ncols, 1);
    Matrix yBase(nrows, 1);
    yBase = Eigen::ArrayXf::LinSpaced(nrows, -nrows / 2 + 1, nrows / 2);
    // std::cout << "xOnes size: " << xOnes.rows() << ", " << xOnes.cols() << std::endl;
    // std::cout << "xBase size: " << xBase.rows() << ", " << xBase.cols() << std::endl;
    // std::cout << "yOnes size: " << yOnes.rows() << ", " << yOnes.cols() << std::endl;
    // std::cout << "yBase size: " << yBase.rows() << ", " << yBase.cols() << std::endl;
    auto x = xOnes * xBase;
    auto y = yBase * yOnes;
    std::cout << "x size: " << x.rows() << ", " << x.cols() << std::endl;
    std::cout << "y size: " << y.rows() << ", " << y.cols() << std::endl;
    for (int i = 0; i < steps; i++)
    {
        auto sqrtU1 = add(R, x * cosVec[i] + y * sinVec[i]);
        // std::cout << "add finished!" << std::endl;
        projectionWeights[i] = divide(R * R, add(multiply(sqrtU1, sqrtU1), epsilon * epsilon));
    }
    std::cout << "projection weights initialized!" << std::endl;

    // initialize FFTW plan
    fftwf_free(fftInOut);
    fftInOut = (fftwf_complex *)fftwf_alloc_complex(ncols);
    forwardPlan = fftwf_plan_dft_1d(ncols, fftInOut, fftInOut, FFTW_FORWARD, FFTW_MEASURE);
    backwardPlan = fftwf_plan_dft_1d(ncols, fftInOut, fftInOut, FFTW_BACKWARD, FFTW_MEASURE);
    std::cout << "fft plan initialized!" << std::endl;
}

void FDK::weighting()
{
    // multiply every projected data with corresponding weight
    for (auto &data : projections)
    {
        data = multiply(data, weights);
    }
    std::cout << "weighted!" << std::endl;
}

void FDK::filtering()
{
    int size = projections.size();
    int nrows = projections[0].rows();
    int ncols = projections[1].cols();
    float max = -1e15;
    float min = 1e15;
    for (int k = 0; k < size; k++)
    {
        for (int i = 0; i < nrows; i++)
        {
            // 1D FFT for each projection row
            for (int j = 0; j < ncols; j++)
            {
                fftInOut[j][0] = projections[k](i, j);
                fftInOut[j][1] = 0.f;
            }
            fftwf_execute_dft(forwardPlan, fftInOut, fftInOut);

            // use filter to weight the frequency data
            for (int j = 0; j < ncols; j++)
            {
                fftInOut[j][0] *= std::abs(filter[(j + ncols / 2) % ncols]);
                fftInOut[j][1] *= std::abs(filter[(j + ncols / 2) % ncols]);
            }

            // 1D IFFT for each frequency data
            fftwf_execute_dft(backwardPlan, fftInOut, fftInOut);
            for (int j = 0; j < ncols; j++)
            {
                projections[k](i, j) = fftInOut[j][0] / float(ncols);
            }
        }
    }
    std::cout << "filtered!" << std::endl;
}

FDK::vectorM FDK::backprojecting()
{
    // backprojection weighting
    int size = projections.size();
    // for (int i = 0; i < size; i++)
    // {
    //     projections[i] = multiply(projections[i], projectionWeights[i]);
    // }
    // std::cout << "backprojection weighted!" << std::endl;

    // backprojection
    int nrows = projections[0].rows();
    int ncols = projections[0].cols();
    vectorM result(nrows, Matrix::Zero(ncols, ncols));
    std::cout << "using policy: bilinear" << std::endl;
    for (int step = 0; step < size; step++)
    {
        std::cout << "progress: " << std::setw(6) << std::setprecision(4) << float(step + 1) / float(size) * 100.f << "%";
        for (int k1 = 0; k1 < nrows; k1++)
        {
            float x = (k1 - nrows / 2 - 1);
            for (int k2 = 0; k2 < ncols; k2++)
            {
                float y = (k2 - ncols / 2 - 1);
                float U_1 = R / (R + x / 64.f * sinVec[step] - y / 64.f * cosVec[step]);
                float a = (x * cosVec[step] + y * sinVec[step]) * U_1;
                int xx = round(a + nrows / 2 + 1);
                float u1 = a + nrows / 2 + 1 - float(xx);
                for (int k3 = 0; k3 < nrows; k3++)
                {
                    float z = k3 - nrows / 2 - 1;
                    float b = z * U_1;
                    int yy = round(b + nrows / 2 + 1);
                    float u2 = b + nrows / 2 - float(yy);
                    if (xx >= 0 && xx < nrows && yy >= 0 && yy < nrows)
                    {
                        float temp = (1 - u1) * (1 - u2) * projections[step](xx, yy) + (1 - u1) * u2 * projections[step](xx, yy + 1)
                            + u1 * (1 - u2) * projections[step](xx + 1, yy) + u1 * u2 * projections[step](xx + 1, yy + 1);
                        result[k3](k1, k2) += temp * U_1 * U_1 * 2 * M_PI / float(size);
                    }
                    else
                    {
                        result[k3](k1, k2) = result[k3](k1, k2);
                    }
                }
            }
        }
        std::cout << '\r';
    }
    // for (int i = 0; i < nrows; i++)
    // {
    //     std::cout << "progress: " << std::setw(6) << std::setprecision(4) << float(i + 1) / float(nrows) * 100.f << "%";
    //     for (int k = 0; k < size; k++)
    //     {
    //         Eigen::ArrayXf array = projections[k](i, Eigen::placeholders::all);
    //         Matrix temp(result[i]);
    //         for (int j = 0; j < ncols; j++)
    //         {
    //             temp(j, Eigen::placeholders::all) = array;
    //         }
    //         // temp(Eigen::placeholders::all, Eigen::placeholders::all) = array;
    //         result[i] += rotate(temp, k, "bilinear");
    //     }
    //     std::cout << '\r';
    // }
    std::cout << "\nover!" << std::endl;
    std::cout << "result size: " << result.size() << std::endl;
    return result;
}

float Q_rsqrt(float number)
{
    long i;
    float x2, y;
    const float threeHalfs = 1.5f;

    x2 = number * 0.5f;
    y = number;
    i = *(long *)&y;           // evil floating point bit level hacking
    i = 0x5f3759df - (i >> 1); // What the fuck?
    y = *(float *)&i;
    y = y * (threeHalfs - (x2 * y * y)); // 1st iteration
    // y  = y * ( threeHalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    return y;
}
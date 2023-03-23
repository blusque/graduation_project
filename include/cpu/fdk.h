#pragma once

#include <Eigen/Core>
#include <vector>
#include <string>
#include <fftw3.h>

using Matrix = Eigen::MatrixXf;

/**
 * @brief
 *
 */
class FDK
{
public:
    using string = std::string;
    using complex = std::complex<float>;
    using vector = std::vector<float>;
    using vectorC = std::vector<complex>;
    using vectorM = std::vector<Matrix>;

public:
    FDK() = default;
    explicit FDK(float R, const string &filter, const vectorM &projections);
    FDK(const FDK &other) = delete;
    FDK &operator=(const FDK &other) = delete;
    FDK(FDK &&other);
    FDK &operator=(FDK &&other);

    ~FDK();

private:
    float R{100.f};        // radius of the detector circle
    string filterType{"r-l"};   // filter name (default 'r-l', optional 's-l')
    vector filter; // filter
    vectorM projections; // projected images (in size of degrees)
    Matrix weights; // weights
    vectorM projectionWeights; // projection weights
    vector cosVec; // cos vector
    vector sinVec; // sin vector
    fftwf_plan forwardPlan; // the fft plan
    fftwf_plan backwardPlan; // the ifft plan
    fftwf_complex *fftInOut; // the data array

public:
    void initialize();

    void weighting();

    void filtering();

    vectorM backprojecting();

private:
    Matrix add(float l, const Matrix &r);

    Matrix add(const Matrix &l, float r);

    Matrix multiply(const Matrix &l, const Matrix &r);

    Matrix divide(float l, const Matrix &r);

    void linspace(float start, float end, int steps, vector &vec);

    Matrix rotate(const Matrix &input, int step, const string &interpType);

    float interpolate(const Matrix &input, float x, float y, const string &interpType);
};
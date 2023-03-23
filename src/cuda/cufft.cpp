#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include "device_launch_parameters.h"

#define pi 3.1415926535
#define LENGTH 100 //signal sampling points

int main()
{
    // data gen
    float Data1[LENGTH] = {0};
    float Data2[LENGTH] = {0};
    float fs = 1000000.000;//sampling frequency
    float f0 = 200000.00;// signal frequency
    for (int i = 0; i < LENGTH; i++)
    {
        Data1[i] = 1.35 * cos(2 * pi * f0 * i / fs);//signal gen,
        Data2[i] = 1.35 * sin(2 * pi * f0 * i / fs); // signal gen,
    }

    cufftComplex *CompData = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));//allocate memory for the data in host
    cufftComplex *result = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));
    int i;
    for (i = 0; i < LENGTH; i++)
    {
        CompData[i].x = Data1[i];
        CompData[i].y = 0;
    }

    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_C2C, 1);//declaration
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    for (int i = -LENGTH / 2; i < LENGTH / 2; i++)
    {
        CompData[i].x *= abs(float(i) / float(LENGTH) * 2.f);
        CompData[i].y *= abs(float(i) / float(LENGTH) * 2.f);
    }

    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice); // copy data from host to device
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result
    for (i = 0; i < LENGTH / 2; i++)
    {
        printf("i=%d\tf= %6.1fHz\tRealAmp=%3.2f\t", i, fs * i / LENGTH, CompData[i].x);
        printf("ImagAmp=%3.1fi", CompData[i].y);
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < LENGTH; i++)
    {
        printf("i=%d\tRealRes=%3.2f\t", i, result[i].x / (float)LENGTH);
        printf("ImagRes=%3.2fi\t", result[i].y / (float)LENGTH);
        printf("RealAmp=%3.2f\t", Data1[i]);
        // printf("ImagAmp=%3.2fi\t", Data2[i]);
        printf("\n");
    }
    cufftDestroy(plan);
    free(CompData);
    free(result);
    cudaFree(d_fftData);
    return 0;
}
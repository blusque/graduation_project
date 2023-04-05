#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int len, int scale, float *a, float *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < len * scale; i += stride)
    {
        int aIndex = i % len;
        int bIndex = i;
        a[aIndex] = __fadd_rn(a[aIndex], b[bIndex]);
    }
}

__device__ void swap(float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

__global__ void swap_wrapper(int len, int scale, float *a, float *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < len * scale; i += stride)
    {
        int aIndex = i % len;
        int bIndex = i;
        swap(a[aIndex], b[bIndex]);
    }
}

int main()
{
    constexpr int len = 100;
    constexpr int scale = 1;
    float *b, *a;
    float bHost[len * scale];
    for (int i = 0; i < len * scale; i++)
    {
        bHost[i] = 1.33;
    }
    float aHost[len] = {0};
    cudaMalloc((void **)&a, len * sizeof(float));
    cudaMalloc((void **)&b, len * scale * sizeof(float));
    cudaMemcpy(a, aHost, sizeof(float) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(b, bHost, sizeof(float) * len * scale, cudaMemcpyHostToDevice);
    swap_wrapper<<<1, len>>>(len, scale, a, b);
    cudaMemcpy(aHost, a, sizeof(float) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(bHost, b, sizeof(float) * len * scale, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; i++)
    {
        std::cout << aHost[i] << ' ';
    }
    cudaFree(a);
    cudaFree(b);
    std::cout << std::endl;
    return 0;
}

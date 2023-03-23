#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int len, int scale, int *a, int *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < len * scale; i += stride)
    {
        int aIndex = i % len;
        int bIndex = i;
        a[aIndex] += b[bIndex];
    }
}

int main()
{
    constexpr int len = 100;
    constexpr int scale = 20;
    int *b, *a;
    int bHost[len * scale];
    for (int i = 0; i < len * scale; i++)
    {
        bHost[i] = 1;
    }
    int aHost[len] = {0};
    cudaMalloc((void **)&a, len * sizeof(int));
    cudaMalloc((void **)&b, len * scale * sizeof(int));
    cudaMemcpy(a, aHost, sizeof(int) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(b, bHost, sizeof(int) * len * scale, cudaMemcpyHostToDevice);
    add<<<10, len>>>(len, scale, a, b);
    cudaMemcpy(aHost, a, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaMemcpy(bHost, b, sizeof(int) * len * scale, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; i++)
    {
        std::cout << aHost[i] << ' ';
    }
    cudaFree(a);
    cudaFree(b);
    std::cout << std::endl;
    return 0;
}

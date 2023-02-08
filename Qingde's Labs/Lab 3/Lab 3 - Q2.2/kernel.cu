#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__ __managed__ int a[5], b[5], c[5];

__global__ void dotKernel(int* c, int* a, int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int main()
{
    for (int i = 0; i < 5; i++)
    {
        a[i] = i + 1;
        b[i] = (i + 1) * 10;
    }

    cudaError_t cudaStatus;

    dotKernel << <1, 5 >> > (c, a, b);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
    }

    printf("{1,2,3,4,5} . {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    return 0;
}
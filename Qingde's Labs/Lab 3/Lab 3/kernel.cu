#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void dotKernel(int *c, int *a, int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

void dot(int* c, int* a, int* b, const int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] * b[i];
    }
}

void sumOfDot(int *c, int &sum, const int size)
{
    for (int i = 0; i < size; i++)
    {
        sum += c[i];
    }
}

int main()
{
    const int arraySize = 5;
    int a[arraySize] = { 1, 2, 3, 4, 5 };
    int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    int sum = 0;

    cudaError_t cudaStatus;

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dotKernel << <1, arraySize >> > (dev_c, dev_a, dev_b);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    sumOfDot(c, sum, arraySize);

    printf("{1,2,3,4,5} . {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    printf("\n");
    printf("Sum of {%d,%d,%d,%d,%d} = %d\n", c[0], c[1], c[2], c[3], c[4], sum);

    return 0;
}
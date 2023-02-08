#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void dotKernel(int* c, int* a, int* b)
{
    __shared__ int dataPerBlock[4];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];

    dataPerBlock[threadIdx.x] = c[i];

    __syncthreads();

    float subtotal = 0;

    for (int j = 0; j < blockDim.x; j++)
    {
        subtotal += dataPerBlock[j];
    }

   c[blockIdx.x] = subtotal;//total = 2040
}

void sumOfDot(int* c, int& sum)
{
    sum = c[0] + c[1];
}

int main()
{
    const int arraySize = 8;
    int a[arraySize] = { 1, 2, 3, 4, 5, 6, 7, 8};
    int b[arraySize] = { 10, 20, 30, 40, 50, 60, 70, 80 };
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

    dotKernel << <4, 2 >> > (dev_c, dev_a, dev_b);

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

    sumOfDot(c, sum);

    printf("{1,2,3,4,5,6,7,8} . {10,20,30,40,50,60,70,80} = {%d,%d,%d,%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    printf("\n");
    printf("10 + 40 + 90 + 160 + 250 + 360 + 490 + 640 = %d\n", sum);

    return 0;
}
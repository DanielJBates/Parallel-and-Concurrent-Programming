#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 32

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, int* a, int* b)
{
    //int blockSize = blockDim.x * blockDim.y * blockDim.z;

    //int i = threadIdx.x + blockIdx.x * blockSize; //multiple 1D
    //int i = threadIdx.x + threadIdx.y * blockDim.x; //one 2D
    int i = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y);
    
    c[i] = a[i] + b[i];
}

__global__ void matAddKernel(int C[N][N], int A[N][N], int B[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;

    C[i][j] = A[i][j] + B[i][j];
}

void add(int* c, const int* a, const int* b)
{
    for (int i = 0; i < 5; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int arraySize = 5;
    int a[arraySize] = {1,2,3,4,5};
    int b[arraySize] = {10,20,30,40,50};
    int c[arraySize] = { 0 };

    int A[N][N];
    int B[N][N];
    int C[N][N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = i + j;
            B[i][j] = (i + j) * 10;
        }
    }

    //for (int i = 0; i < arraySize; i++)
    //{
    //    a[i] = (i + 1);
    //    b[i] = ((i + 1) * 10);
    //}

    //cudaEvent_t start, stop;

    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //add(c, a, b);

    cudaError_t cudaStatus;

    //int* dev_a = 0;
    //int* dev_b = 0;
    //int* dev_c = 0;

    int (*dA)[N];
    int (*dB)[N];
    int (*dC)[N];

    //cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMalloc((void**)&dA, (N * N) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMalloc((void**)&dB, (N * N) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMalloc((void**)&dC, (N * N) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMemcpy(dA, A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMemcpy(dB, B, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //cudaStatus = cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMemcpy(dC, C, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //cudaEventRecord(start, 0);
    //addKernel << <dim3(2,2), dim3(2,3)>> > (dev_c, dev_a, dev_b);
    matAddKernel << <1, dim3(32, 32) >> > (dC, dA, dB);
    //cudaEventRecord(stop, 0);

    //cudaEventSynchronize(stop);
    //float elapsedTime;
    //cudaEventElapsedTime(&elapsedTime, start, stop);

    //printf("Time elapsed the execution of kernal %fn", elapsedTime);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }

    //cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    cudaStatus = cudaMemcpy(C, dC, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    //cudaFree(dev_c);
    //cudaFree(dev_a);
    //cudaFree(dev_b);

    cudaFree(dC);
    cudaFree(dA);
    cudaFree(dB);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
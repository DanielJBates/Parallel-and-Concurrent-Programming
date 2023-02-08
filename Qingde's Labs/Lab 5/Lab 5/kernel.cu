
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define rowsA 512
#define columnsA 512
#define rowsB 512
#define columnsB 512

//__shared__ int d_A[rowsA][columnsA], int d_B[rowsB][columnsB], int d_C[rowsA][columnsB];

__global__ void kernalMatMultipy(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    int j = threadIdx.y;

    int C_ij = i * blockDim.x + j;

    int temp = 0;
    for (int k = 0; k < rowsA; k++)
    {
        int i_A = i * columnsA + k;
        int i_B = k * columnsB + j;

        temp += a[i_A] * b[i_B];
    }

    c[C_ij] = temp;
}

void matMultiply(int a[rowsA][columnsA], int b[rowsB][columnsB], int c[rowsA][columnsB])
{
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < columnsB; j++)
        {
            for (int k = 0; k < rowsB; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main()
{
    int A[rowsA][columnsA];
    int B[rowsB][columnsB];
    int C[rowsA][columnsB] = { 0 };

    int x = 1;

    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < columnsA; j++)
        {
            A[i][j] = x;
            x++;
        }
    }

    for (int i = 0; i < rowsB; i++)
    {
        for (int j = 0; j < columnsB; j++)
        {
            B[i][j] = x;
            x++;
        }
    }
    const int arraySizeA = rowsA * columnsA;
    const int arraySizeB = rowsB * columnsB;
    const int arraySizeC = rowsA * columnsB;

    int a[arraySizeA];

    x = 1;

    for (int i = 0; i < arraySizeA; i++)
    {
        a[i] = x;
        x++;
    }

    int b[arraySizeB];

    for (int i = 0; i < arraySizeB; i++)
    {
        b[i] = x;
        x++;
    }
    
    int c[arraySizeC] = { 0 };

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc((void**)&d_A, arraySizeA * sizeof(int));
    cudaMalloc((void**)&d_B, arraySizeB * sizeof(int)); 
    cudaMalloc((void**)&d_C, arraySizeC * sizeof(int));

    cudaMemcpy(d_A, a, arraySizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, arraySizeB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, c, arraySizeC * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 gridShape(1);
    dim3 blockShape(columnsA, rowsB);

    cudaEventRecord(start, 0); 
    //matMultiply(A, B, C);
    kernalMatMultipy << <gridShape, blockShape>> > (d_C, d_A, d_B);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    printf("Time: %fn", time);

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_C, arraySizeC * sizeof(int), cudaMemcpyDeviceToHost);

    return 0;
}
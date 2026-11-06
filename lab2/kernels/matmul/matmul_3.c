#include "matmul_vs.h"

void matmul(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}
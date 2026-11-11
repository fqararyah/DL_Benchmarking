#include "matmul_vs.h"

void matmul(float *A, float *B, float *C, int N)
{
    const int FACTOR = 8;

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float a = A[i * N + k];
            int j;

            for (j = 0; j <= N - FACTOR; j += FACTOR)
            {
                C[i * N + j + 0] += a * B[k * N + j + 0];
                C[i * N + j + 1] += a * B[k * N + j + 1];
                C[i * N + j + 2] += a * B[k * N + j + 2];
                C[i * N + j + 3] += a * B[k * N + j + 3];
                C[i * N + j + 4] += a * B[k * N + j + 4];
                C[i * N + j + 5] += a * B[k * N + j + 5];
                C[i * N + j + 6] += a * B[k * N + j + 6];
                C[i * N + j + 7] += a * B[k * N + j + 7];
            }
        }
    }
}

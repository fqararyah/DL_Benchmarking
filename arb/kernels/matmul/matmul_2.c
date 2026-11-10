#include "matmul_vs.h"
#include <stdlib.h>

void matmul(float *A, float *B, float *C, int N)
{
    float *tmp = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float a = A[i * N + k];

            for (int j = 0; j < N; j++)
            {
                tmp[j] = a * B[k * N + j];
            }

            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += tmp[j];
            }
        }
    }

    free(tmp);
}

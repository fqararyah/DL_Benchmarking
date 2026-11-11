#include "matmul_vs.h"

void matmul(float *A, float *B, float *C, int N)
{
    const int FACTOR = 8;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += FACTOR)
        {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

            for (int k = 0; k < N; k++)
            {
                float a = A[i * N + k];
                sum0 += a * B[(j + 0) * N + k];
                sum1 += a * B[(j + 1) * N + k];
                sum2 += a * B[(j + 2) * N + k];
                sum3 += a * B[(j + 3) * N + k];
                sum4 += a * B[(j + 4) * N + k];
                sum5 += a * B[(j + 5) * N + k];
                sum6 += a * B[(j + 6) * N + k];
                sum7 += a * B[(j + 7) * N + k];
            }

            C[i * N + j + 0] = sum0;
            C[i * N + j + 1] = sum1;
            C[i * N + j + 2] = sum2;
            C[i * N + j + 3] = sum3;
            C[i * N + j + 4] = sum4;
            C[i * N + j + 5] = sum5;
            C[i * N + j + 6] = sum6;
            C[i * N + j + 7] = sum7;
        }
    }
}

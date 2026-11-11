#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matmul_vs.h"

void zero_matrix(float *C, int N)
{
    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;
}

#define FLUSH_SIZE (32 * 1024 * 1024) // 32 MB

// Flush CPU caches by reading/writing a large buffer
void flush_cache()
{
    static volatile char *flush_buffer = NULL;
    if (!flush_buffer)
    {
        flush_buffer = malloc(FLUSH_SIZE);
        if (!flush_buffer)
            return; // failed allocation
    }

    volatile char sink = 0;

    // Touch every cache line to evict old data
    for (size_t i = 0; i < FLUSH_SIZE; i += 64)
    { // 64B typical cache line
        flush_buffer[i] = (char)i;
        sink += flush_buffer[i];
    }

    // Prevent compiler from optimizing away
    (void)sink;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <version_number> <matrix_size>\n", argv[0]);
        return 1;
    }

    int version = atoi(argv[1]);
    int N = atoi(argv[2]);
    const int RUNS = 20;

    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));

    srand(0);
    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    float *B_T;
    if (version == 5)
    {
        B_T = malloc(N * N * sizeof(float));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                B_T[j * N + i] = B[i * N + j]; // B_T[j,i] = B[i,j]
    }

    double total_gflops = 0.0;

    for (int r = 0; r < RUNS; r++)
    {
        zero_matrix(C, N);
        flush_cache();
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (version == 5)
        {
            matmul(A, B_T, C, N);
        }
        else
        {
            matmul(A, B, C, N);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;

        double gflops = 2.0 * N * N * N / elapsed / 1e9;
        total_gflops += gflops;

        FILE *fout = fopen("results.txt", "w");
        if (!fout)
        {
            perror("Failed to open results.txt");
            return 1;
        }
        int mid = (N * N) / 2;
        int last = N * N - 1;
        fprintf(fout, "Run %d (Version %d, N=%d): C[0]=%f, C[mid]=%f, C[last]=%f\n",
                r + 1, version, N, C[0], C[mid], C[last]);
        fclose(fout);
    }

    double avg_gflops = total_gflops / RUNS;
    printf("-----------------------------------\n");
    printf("Version %d | N = %d | Avg GFLOP/s = %.2f\n",
           version, N, avg_gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}

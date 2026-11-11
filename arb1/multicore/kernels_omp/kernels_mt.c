#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void triad(float *a, float *b, float *c, float *d, int n, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
}


float dot_product(float *a, float *b, int n, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    double sum = 0.0;
    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for reduction(+:sum) schedule(static)
        for (int i = 0; i < n; i++) {
            sum += (double)a[i] * (double)b[i];
        }
    }
    return (float)sum;
}


void fma_chain(float *a, float *b, float *c, int n, int iters, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            float x = a[i];
            const float y = b[i];
            const float coeff = c[i];
            for (int k = 0; k < iters; k++) {
                x = x * y + coeff;
            }
            a[i] = x;
        }
    }
}


void poly_eval(float *a, float *b, int n, int degree, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            const float x = a[i];
            float y = b[degree];
            for (int d = degree - 1; d >= 0; d--) {
                y = y * x + b[d];
            }
            a[i] = y;
        }
    }
}


#define FLUSH_SIZE (32 * 1024 * 1024) // 32 MB

void flush_cache() {
    static volatile char *flush_buffer = NULL;
    if (!flush_buffer) {
        flush_buffer = malloc(FLUSH_SIZE);
        if (!flush_buffer)
            return; // failed allocation
    }

    volatile char sink = 0;

    for (size_t i = 0; i < FLUSH_SIZE; i += 64) {
        flush_buffer[i] = (char)i;
        sink += flush_buffer[i];
    }

    (void)sink;
}


int main(int argc, char *argv[]) {

    const char *kernel = argv[1];
    int N = atoi(argv[2]);
    int param = (argc > 3) ? atoi(argv[3]) : 200; // iters for fma, degree for poly

    int active_threads = 1;

    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c = (float *)malloc(N * sizeof(float));
    float *d = (float *)malloc(N * sizeof(float));

    if (!a || !b || !c || !d) {
        printf("Allocation failed.\n");
        free(a); free(b); free(c); free(d);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 3.0f;
        d[i] = 4.0f;
    }

    const int RUNS = 20;
    double total_time = 0.0;
    for (int run = 0; run < RUNS; run++) {
        flush_cache();
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (strcmp(kernel, "dot") == 0) {
            volatile float result = dot_product(a, b, N, &active_threads);
            (void)result;
        } else if (strcmp(kernel, "triad") == 0) {
            triad(a, b, c, d, N, &active_threads);
        } else if (strcmp(kernel, "fma") == 0) {
            fma_chain(a, b, c, N, param, &active_threads);
        } else if (strcmp(kernel, "poly") == 0) {
            poly_eval(a, b, N, param, &active_threads);
        } else {
            printf("Unknown kernel: %s\n", kernel);
            free(a); free(b); free(c); free(d);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        total_time += elapsed;
    }

    double avg_time = total_time / (double)RUNS;

    double flops = 0.0;
    if (strcmp(kernel, "dot") == 0)
        flops = 2.0 * N;
    else if (strcmp(kernel, "triad") == 0)
        flops = 2.0 * N;
    else if (strcmp(kernel, "fma") == 0)
        flops = 2.0 * N * param;
    else if (strcmp(kernel, "poly") == 0)
        flops = 2.0 * N * param;

    double gflops = flops / (avg_time * 1e9);

    FILE *fout = fopen("output_sample.txt", "w");
    if (fout) {
        for (int i = 0; i < 5 && i < N; i++) {
            fprintf(fout, "%f\n", a[i]);
        }
        fclose(fout);
    }

    printf("Kernel: %-5s | N: %d | Threads: %d | Avg Time: %.6f s | Avg GFLOPs: %.3f\n",
           kernel, N, active_threads, avg_time, gflops);

    free(a); free(b); free(c); free(d);
    return 0;
}

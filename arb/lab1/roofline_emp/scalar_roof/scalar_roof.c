#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    const long N = 1000000000; // iterations
    const int RUNS = 10;

    if (argc < 17) {
        fprintf(stderr, "Usage: %s a1 a2 a3 a4 a5 a6 a7 a8 b1 b2 b3 b4 b5 b6 b7 b8\n", argv[0]);
        return 1;
    }

    // Read a1..a8 and b1..b8 from argv
    float a1 = atof(argv[1]);
    float a2 = atof(argv[2]);
    float a3 = atof(argv[3]);
    float a4 = atof(argv[4]);
    float a5 = atof(argv[5]);
    float a6 = atof(argv[6]);
    float a7 = atof(argv[7]);
    float a8 = atof(argv[8]);

    float b1 = atof(argv[9]);
    float b2 = atof(argv[10]);
    float b3 = atof(argv[11]);
    float b4 = atof(argv[12]);
    float b5 = atof(argv[13]);
    float b6 = atof(argv[14]);
    float b7 = atof(argv[15]);
    float b8 = atof(argv[16]);

    double total_gflops = 0.0;

    for (int run = 0; run < RUNS; run++) {
        float c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0, c8=0;

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (long i = 0; i < N; i++) {
            c1 = a1*b1 + c1;
            c2 = a2*b2 + c2;
            c3 = a3*b3 + c3;
            c4 = a4*b4 + c4;
            c5 = a5*b5 + c5;
            c6 = a6*b6 + c6;
            c7 = a7*b7 + c7;
            c8 = a8*b8 + c8;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        // Print results to prevent dead-code elimination
        printf("%f %f %f %f %f %f %f %f\n",
               c1,c2,c3,c4,c5,c6,c7,c8);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec)/1e9;

        double gflops = 2.0 * 8 * N / elapsed / 1e9;
        total_gflops += gflops;
    }

    printf("Peak GFLOP/s over %d runs: %.2f\n", RUNS, total_gflops / RUNS);

    return 0;
}

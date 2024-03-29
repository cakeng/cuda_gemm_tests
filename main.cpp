#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cblas.h>

#include "util.h"
#include "mat_mul.h"

static void print_help(const char *prog_name)
{
    printf("Usage: %s [-pvh] [-n num_iterations] M N K\n", prog_name);
    printf("Options:\n");
    printf("  -p : print matrix data. (default: off)\n");
    printf("  -v : validate matrix multiplication. (default: off)\n");
    printf("  -h : print this page.\n");
    printf("  -n : number of iterations (default: 1)\n");
    printf("   M : number of rows of matrix A and C. (default: 8)\n");
    printf("   N : number of columns of matrix B and C. (default: 8)\n");
    printf("   K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_matrix = false;
static bool validation = false;
static bool skip_data_movement = false;
static int M = 8, N = 8, K = 8;
static int num_iterations = 1;

void cpu_mat_mul(float *A, float *B, float *C, int M, int N, int K, int skip_data_movement)
{
    cblas_sgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
        N, M, K, 1, B, N, A, K, 0, C, N);
}

static void parse_opt(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "pvhst:n:")) != -1)
    {
        switch (c)
        {
        case 'p':
            print_matrix = true;
            break;
        case 'v':
            validation = true;
            break;
        case 'n':
            num_iterations = atoi(optarg);
            break;
        case 's':
            skip_data_movement = true;
            break;
        case 'h':
        default:
            print_help(argv[0]);
            exit(0);
        }
    }
    for (int i = optind, j = 0; i < argc; ++i, ++j)
    {
        switch (j)
        {
        case 0:
            M = atoi(argv[i]);
            break;
        case 1:
            N = atoi(argv[i]);
            break;
        case 2:
            K = atoi(argv[i]);
            break;
        default:
            break;
        }
    }
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Skip data movement: %s\n", skip_data_movement ? "on" : "off");
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
}

int main(int argc, char **argv)
{
    parse_opt(argc, argv);

    printf("Initializing matrix... ");
    fflush(stdout);
    float *A, *B, *C, *C_ans;
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    if (validation)
    {
        alloc_mat(&C_ans, M, N);
        calculate_mat_mul(A, B, C_ans, M, N, K);
    }
    if (print_matrix)
    {
        printf("MATRIX A:\n");
        print_mat(A, M, K);
        printf("MATRIX B:\n");
        print_mat(B, K, N);
        printf("MATRIX C:\n");
        print_mat(C, M, N);
    }
    printf("done!\n");

    printf("Initializing CPU...\n");
    fflush(stdout);
    double elapsed_time_sum = 0;
    for (int i = -3; i < num_iterations; ++i)
    {
        if (i < 0)
        {
            printf("Warming up CPU...");
            fflush(stdout);
        }
        else
        {
            printf("Calculating...(iter=%d) ", i);
            fflush(stdout);
        }
        timer_start(0);
        cpu_mat_mul (A, B, C, M, N, K, skip_data_movement);
        double elapsed_time = timer_stop(0);
        printf("%f sec\n", elapsed_time);
        if (i >= 0)
        {
            elapsed_time_sum += elapsed_time;
        }
    }

    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    if (validation)
    {
        check_mat_mul(C_ans, C, M, N, K);
        rand_mat (C, M, N);
    }

    printf("Initializing CUDA...\n");
    fflush(stdout);
    mat_mul_init(A, B, C, M, N, K);
    elapsed_time_sum = 0;
    for (int i = -3; i < num_iterations; ++i)
    {
        if (i < 0)
        {
            printf("Warming up GPU...");
            fflush(stdout);
        }
        else
        {
            printf("Calculating...(iter=%d) ", i);
            fflush(stdout);
        }
        if (skip_data_movement)
            mat_mul_write_to_gpu(A, B, C, M, N, K);
        timer_start(0);
        mat_mul(A, B, C, M, N, K, skip_data_movement);
        double elapsed_time = timer_stop(0);
        if (skip_data_movement)
            mat_mul_read_from_gpu(A, B, C, M, N, K);
        printf("%f sec\n", elapsed_time);
        if (i >= 0)
        {
            elapsed_time_sum += elapsed_time;
        }
    }
    
    elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    if (validation)
    {
        check_mat_mul(C_ans, C, M, N, K);
        rand_mat (C, M, N);
    }

    printf("Initializing cuBLAS...\n");
    fflush(stdout);
    cublas_mat_mul_init(A, B, C, M, N, K);
    elapsed_time_sum = 0;
    for (int i = -3; i < num_iterations; ++i)
    {
        if (i < 0)
        {
            printf("Warming up GPU...");
            fflush(stdout);
        }
        else
        {
            printf("Calculating...(iter=%d) ", i);
            fflush(stdout);
        }
        if (skip_data_movement)
            cublas_mat_mul_write_to_gpu(A, B, C, M, N, K);
        timer_start(0);
        cublas_mat_mul(A, B, C, M, N, K, skip_data_movement);
        double elapsed_time = timer_stop(0);
        if (skip_data_movement)
            cublas_mat_mul_read_from_gpu(A, B, C, M, N, K);
        printf("%f sec\n", elapsed_time);
        if (i >= 0)
        {
            elapsed_time_sum += elapsed_time;
        }
    }
    if (print_matrix)
    {
        printf("MATRIX C:\n");
        print_mat(C, M, N);
    }

    elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    if (validation)
    {
        check_mat_mul(C_ans, C, M, N, K);
        rand_mat (C, M, N);
    }

    return 0;
}

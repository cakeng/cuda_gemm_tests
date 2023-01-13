#include "mat_mul.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cublas.h>

#define CUBLAS_CHECK(val) { \
	if (val != CUBLAS_STATUS_SUCCESS) { \
		fprintf(stderr, "Error at line %d in file %s\n", __LINE__, __FILE__); \
		exit(1); \
	} \
}

static float *a_d, *b_d, *c_d;

void cublas_mat_mul_write_to_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    CUBLAS_CHECK(cublasSetVector(M*K, sizeof(float), A, 1, a_d, 1));

    CUBLAS_CHECK(cublasSetVector(K*N, sizeof(float), B, 1, b_d, 1));
}

void cublas_mat_mul_read_from_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    CUBLAS_CHECK(cublasGetVector(M*N, sizeof(float), c_d, 1, C, 1));
    cudaDeviceSynchronize();
}

void cublas_mat_mul(float *A, float *B, float *C, int M, int N, int K, int skip_data_movement)
{
    if (!skip_data_movement)
        cublas_mat_mul_write_to_gpu (A, B, C, M, N, K);

    cublasSgemm('T', 'T', M, N, K, 1.0, a_d, K, b_d, N, 0, c_d, M);

    CUBLAS_CHECK(cublasGetError());

    if (!skip_data_movement)
        cublas_mat_mul_read_from_gpu (A, B, C, M, N, K);
}

void cublas_mat_mul_init(float *A, float *B, float *C, int M, int N, int K)
{
    // Allocate the device input matrixs for A, B, C;

    CUBLAS_CHECK(cublasAlloc(M*K, sizeof(float), (void**)&a_d));

    CUBLAS_CHECK(cublasAlloc(K*N, sizeof(float), (void**)&b_d));

    CUBLAS_CHECK(cublasAlloc(M*N, sizeof(float), (void**)&c_d));

    CUBLAS_CHECK(cublasInit());
}
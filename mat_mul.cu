#include "mat_mul.h"

#include <cuda_runtime.h>
#include <cstdio>

#define __NOPRINT 1
#define __TILE_SIZE_K 8
#define __TILE_SIZE_M 128
#define __TILE_SIZE_N 128
#define __VEC_SIZE_M 8
#define __VEC_SIZE_N 8

#define __THREAD_NUM_M (__TILE_SIZE_M / __VEC_SIZE_M) 
#define __THREAD_NUM_N (__TILE_SIZE_N / __VEC_SIZE_N) 
#define __THREAD_NUM (__THREAD_NUM_N*__THREAD_NUM_M)

#define __MAX_GPU_NUM 4

int num_devices = 0;

static float *a_d[__MAX_GPU_NUM], *b_d[__MAX_GPU_NUM], *c_d[__MAX_GPU_NUM];

__global__ void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    const int mLocal = threadIdx.x*__VEC_SIZE_M;
    const int nLocal = threadIdx.y*__VEC_SIZE_N; 
    const int id = threadIdx.x*__THREAD_NUM_N + threadIdx.y;
    const int mGroup = blockIdx.x*__TILE_SIZE_M;
    const int nGroup = blockIdx.y*__TILE_SIZE_N;
    const int m = mGroup + mLocal;
    const int n = nGroup + nLocal;
    const int kRem = K%__TILE_SIZE_K;
    __shared__ float ACache [__TILE_SIZE_K*__TILE_SIZE_M];
    __shared__ float BCache [__TILE_SIZE_K*__TILE_SIZE_N];
    float cout[__VEC_SIZE_M][__VEC_SIZE_N];
    for (int vecM = 0; vecM < __VEC_SIZE_M; vecM++)
    {
        for (int vecN = 0; vecN < __VEC_SIZE_N; vecN++)
        {
            cout[vecM][vecN] = 0;
        }   
    }
    const int kEnd = K - kRem;

    int kIdx = 0;  
    for (; kIdx < kEnd; kIdx += __TILE_SIZE_K)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < ((__TILE_SIZE_K*__TILE_SIZE_M)/__THREAD_NUM); aIdx++)
        {
            const int cache_idx = id*((__TILE_SIZE_K*__TILE_SIZE_M)/__THREAD_NUM) + aIdx;
            const int cache_m = cache_idx%__TILE_SIZE_M;
            const int cache_k = cache_idx/__TILE_SIZE_M;
            ACache[cache_idx] = A[K*(mGroup + cache_m) + (kIdx + cache_k)];
        }
        for (int bIdx = 0; bIdx < ((__TILE_SIZE_K*__TILE_SIZE_N)/__THREAD_NUM); bIdx++)
        {
            const int cache_idx = id*((__TILE_SIZE_K*__TILE_SIZE_N)/__THREAD_NUM) + bIdx;
            const int cache_n = cache_idx%__TILE_SIZE_N;
            const int cache_k = cache_idx/__TILE_SIZE_N;
            BCache[cache_idx] = B[N*(kIdx + cache_k) + (nGroup + cache_n)];
        }
        __syncthreads();
        for (int kk = 0; kk < __TILE_SIZE_K; kk++)
        {
            // Calculate.
            for (int vecM = 0; vecM < __VEC_SIZE_M; vecM++)
            {
                for (int vecN = 0; vecN < __VEC_SIZE_N; vecN++)
                {
                    cout[vecM][vecN] += ACache[kk*__TILE_SIZE_M + mLocal + vecM] * BCache[kk*__TILE_SIZE_N + nLocal + vecN];
                }   
            }
        }
        // Sync threads.
        __syncthreads();
    }
    // Remainning K
    if (kRem)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < ((__TILE_SIZE_K*__TILE_SIZE_M)/__THREAD_NUM); aIdx++)
        {
            const int cache_idx = id*((__TILE_SIZE_K*__TILE_SIZE_M)/__THREAD_NUM) + aIdx;
            const int cache_m = cache_idx%__TILE_SIZE_M;
            const int cache_k = cache_idx/__TILE_SIZE_M;
            ACache[cache_idx] = A[K*(mGroup + cache_m) + (kIdx + cache_k)];
        }
        for (int bIdx = 0; bIdx < ((__TILE_SIZE_K*__TILE_SIZE_N)/__THREAD_NUM); bIdx++)
        {
            const int cache_idx = id*((__TILE_SIZE_K*__TILE_SIZE_N)/__THREAD_NUM) + bIdx;
            const int cache_n = cache_idx%__TILE_SIZE_N;
            const int cache_k = cache_idx/__TILE_SIZE_N;
            BCache[cache_idx] = B[N*(kIdx + cache_k) + (nGroup + cache_n)];
        }
        // Sync threads.
        __syncthreads();
  
        for (int kk = 0; kk < kRem; kk++)
        {    
            // Calculate.
            for (int vecM = 0; vecM < __VEC_SIZE_M; vecM++)
            {
                for (int vecN = 0; vecN < __VEC_SIZE_N; vecN++)
                {
                    cout[vecM][vecN] += ACache[kk*__TILE_SIZE_M + mLocal + vecM] * BCache[kk*__TILE_SIZE_N + nLocal + vecN];
                }   
            } 
        }
        // Sync threads.
        __syncthreads();
    }
    // Save results
    for (int vecM = 0; vecM < __VEC_SIZE_M; vecM++)
    {
        for (int vecN = 0; vecN < __VEC_SIZE_N; vecN++)
        {
            if (m + vecM < M &&  n + vecN < N)
                C[N*(m + vecM) + (n + vecN)] = cout[vecM][vecN];
        }   
    }

}

void mat_mul(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(a_d[i], A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        dim3 blockDim (__THREAD_NUM_M, __THREAD_NUM_N, 1);
        dim3 gridDim (M/__TILE_SIZE_M + ((M%__TILE_SIZE_M) > 0)
            , N/__TILE_SIZE_N + ((N%__TILE_SIZE_N) > 0), 1);
        sgemm<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
    }
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(C, c_d[i], M * N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K)
{
    printf ("Block Settings: M: %d, N: %d, K: %d, VecM: %d, VecN: %d, Thread Num: %d, ACache: %d, BCache: %d\n",
        __TILE_SIZE_M, __TILE_SIZE_N, __TILE_SIZE_K, __VEC_SIZE_M, __VEC_SIZE_N, __THREAD_NUM_M*__THREAD_NUM_N, __TILE_SIZE_M*__TILE_SIZE_K, __TILE_SIZE_N*__TILE_SIZE_K);
    printf ("Num blocks: %d\n", (M/__TILE_SIZE_M)*(N/__TILE_SIZE_N));
    if (!((((__TILE_SIZE_K)*__TILE_SIZE_M)%__THREAD_NUM) == 0 && (((__TILE_SIZE_K)*__TILE_SIZE_N)%__THREAD_NUM) == 0))
        exit(0);
    cudaGetDeviceCount(&num_devices);

    printf("Using %d devices\n", num_devices);
    for (int i = 0; i < num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("[GPU %d] %s\n", i, prop.name);
    }

    if (num_devices <= 0)
    {
        printf("No CUDA device found. Aborting\n");
        exit(1);
    }

    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&a_d[i], (M+__TILE_SIZE_M) * (K+__TILE_SIZE_K) * sizeof(float));
        cudaMalloc(&b_d[i], (K+__TILE_SIZE_K) * (N+__TILE_SIZE_N) * sizeof(float));
        cudaMalloc(&c_d[i], M * N * sizeof(float));
        cudaDeviceSynchronize();
    }
}
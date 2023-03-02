# cuda_gemm_tests
Toy implementation of CUDA accelerated GEMM.

Base code is taken from SNU CS Multicore Processor's coursework HW6.

Requires CUDA capable graphics card, Nvidia Drivers, CUDA Toolkit, cuBLAS, and OpenBLAS for full operation.

On RTX 3090 with CUDA version 11.7, custom kernel reaches 11TFLOPs, cuBLAS reaches 15TFLOPs with M=N=K=8192.

On RTX 3070 with CUDA version 12.0, custom kernel reaches 4.5TFLOPs, cuBLAS reaches 5.7TFLOPs with M=N=K=8192.

On RTX 2070 SUPER with CUDA version 11.5, custom kernel reaches 3.5TFLOPs, cuBLAS reaches 5.9TFLOPs with M=N=K=8192.

CPU results: i7-10700: 130GFLOPs(?), i7-7700K: 469GLFOPs, Threadripper-3990X: 2.1TFLOPs using OpenBLAS

//// INSTRUCTIONS ////

"make" to build. 

"./main [M] [N] [K]" to run the GEneral Matrix Multiplication (GEMM) A(MxK) x B(KxN) = C(MxN).

Use the "-v" option to validate results.

Use the "-n [NUM]" option to run the code NUM times and take the average time as the result.

ex. "./main -v -n 20 8192 8192 8912" to run GEMM of two 8192x8192 matrices 20 times, and validate the result.

Modify mat_mul.cu's "sgemm" kernel and "mat_mul" function to implement your own GEMM algorithm and test its performance.


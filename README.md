# cuda_gemm_tests
Toy implementation of CUDA accelerated GEMM.

Base code is taken from SNU CS Multicore Processor's coursework HW6.

Requires CUDA capable graphics card and CUDA toolkit.

Tested on RTX 3090 with CUDA version 11.7. Reaches 11TFLOPs with M=N=K=8192.

Also tested on RTX 2070 SUPER with CUDA version 11.5. Reaches 3.5TFLOPs with M=N=K=8192.


//// INSTRUCTIONS ////

"make" to build. 

"./main [M] [N] [K]" to run the GEneral Matrix Multiplication (GEMM) A(MxK) x B(KxN) = C(MxN).

Use the "-v" option to validate results.

Use the "-n [NUM]" option to run the code NUM times and take the average time as the result.

ex. "./main -v -n 20 8192 8192 8912" to run GEMM of two 8192x8192 matrices 20 times, and validate the result.

Modify mat_mul.cu's "sgemm" kernel and "mat_mul" function to implement your own GEMM algorithm and test its performance.


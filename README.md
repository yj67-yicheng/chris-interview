## Task:

(said by Chris)

What we want to do is to implement “matrix multiply followed by subtract max” over two square matrices. You know matrix multiply; what we want is a code that also finds the max value in each row, and subtracts it from each entry in that row.

Here's exactly what I'd like you to do. Start with the "stupid" implementation, where you just do a nested loops matrix multiply, you write out the result, and then you do a row-by-row subtract-max. See how fast it is on a matrix multiply of two matrices both of size M by M, for M in {256 \* x for x in 1, 2, 3, ..., 10}. Then try to make it faster. Write a little report. Each time you try something, describe what you did, and if it helped!

## Env setup

WSL, Git, CPU flag ensure, Makefile, try to run dgemm_x86[0 - 19]

## Review

[Github repo reference](https://github.com/yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F#how-to-optimize-dgemm-on-x86-cpu-platforms)

## Play around for opt (C++)

### 1. naive

![1.1](./self-solution/static/1.1.png)

comments: quite slow

### 2. use register

C(i,j) is irrelevant to the innermost loop index k, therefore, one can load it into the register before entering the k-loop to avoid unnecessary memory access.

An array element, it must reside in memory. cij is a local scalar variable, it can be placed in a register.

use pointer

![2.1](./self-solution/static/2.1.png)

comments: theoretically it should be improved. But it seems no use in practice.

### 3. register blocking

update the whole 2x2 block of C matrix by loading a 2x1 slice of A and an 1x2 slice of B. Then we conduct an outer-product rank-1 update on the 2x2 C block.

![3.1](./self-solution/static/3.1.png)

comments: great improvement !

### 4. register blocking plus

just as same as 3., add more register blocking. (4 \* 4)

![4.1](./self-solution/static/4.1.png)

comments: Obviously, it should be improved, because there are more register blockings.

### 5. SIMD

process data using Single Instruction Multiple Data (SIMD) instructions. SIMD can process 4 float elements at a lane, and improve cache utilization. (\_\_m128)

![5.1](./self-solution/static/5.1.png)

comments: because we use the SIMD instructions instead of scalar instructions, we get great great improvement.

### 6. Kernel 5 plus

update \_\_m128 to \_\_m256, which can process 8 floats in a lane. \
Register blocking 4 \* 8 \
Unroll the loop by 4 folds. \
Using \_\_mm256_fmadd_ps, instead of mul + add

![6.1](./self-solution/static/6.1.png)

comments: slightly improves the performance. \
ps: should modify 'Code Runner' or use 'bash'

### 7. Kernel 6 + 8 \* 4 blocking

We changed the previous kernel from 4x4 to the current 8x4 so that we obtain a better utilization on all 256-bit YMM registers

![7.1](./self-solution/static/7.1.png)

comments: we have more blocking, so it should be improved

### 8. Kernel 7 + cache blocking

At this point, we have `matmul_v8 (cache blocking) + macro_kernel (same as kernel 7 implementation) + scalar_kernel (edges within block)` to deal with the tasks

```
m_blocking: 192
n_blocking: 2048
k_blocking: 384
```

![8.1](./self-solution/static/8.1.png)

### 9. Kernel 8 + Packing

Packing the data into continous buffers, eliminating the performance penalty of irregular memory access patterns found in vectors

![9.1](./self-solution/static/9.1.png)

comments: fast! But code becomes complicated coz it's greatly related to the hardware layer (cache, TBL, memory, pointers stuff)

## Some other resources

![o1](./self-solution/static/o1.png)

```General Matrix Multiplication
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float sum = beta * (float)C[row * N + col];
    for (int i = 0; i < K; i++) {
        sum += alpha * (float)A[row * K + i] * (float)B[i * N + col];
    }
    C[row * N + col] = (half)sum;
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(16, 16);
    dim3 blockPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul<<<blockPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
```

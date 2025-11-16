A repo for Rice 25fall chris research interview

## Task: 
(said by Chris)

What we want to do is to implement “matrix multiply followed by subtract max” over two square matrices.  You know matrix multiply; what we want is a code that also finds the max value in each row, and subtracts it from each entry in that row. 

Here's exactly what I'd like you to do. Start with the "stupid" implementation, where you just do a nested loops matrix multiply, you write out the result, and then you do a row-by-row subtract-max. See how fast it is on a matrix multiply of two matrices both of size M by M, for M in {256 * x for x in 1, 2, 3, ..., 10}.  Then try to make it faster. Write a little report. Each time you try something, describe what you did, and if it helped!


## Env setup
WSL, Github, Makefile, make, try to run dgemm_x86[0 - 19] 

## Review
[Github repo reference](https://github.com/yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F#how-to-optimize-dgemm-on-x86-cpu-platforms) 

kernel 0: Intel MKL benchmark \
kernel 1: naive version \
kernel 2: register re-use \
kernel 3: 2 * 2 register blocking \
kernel 4: 4 * 4 register blocking \
kernel 5: kernel 4 + AVX2 \
kernel 6: kernel 5 + loop unrolling * 4 \
kernel 7: 8 * 4 kernel + AVX2 + loop unrolling * 4 \
kernel 8: kernel 7 + cache blocking \
kernel 9: kernel 8 + packing \
kernel 10: 24 * 8 kernel + AVX512 + blocking + packing \
kernel 11: kernel 10 + discontinuous packing on B \
kernel 12: kernel 11: from instrinsics to inline ASM \
kernel 13: kernel 12 + changing the whole macro kernel into inline ASM \
kernel 14: kernel 13 + software prefetching on A \
kernel 15: kernel 14 + software prefetching on B \
kernel 16: kernel 15 + software prefetching on C \
kernel 17: kernel 16 + fine-tuned matrix scaling routine on C \
kernel 18: kernel 17 fine-grained packing for B to benefit the CPU frequency boosting (best serial version) 

kernel 19: same as kernel 18 (while best parallel version) ps: segmentation fault, can't find the bug right now


## Play around for opt (C++)
### 1. naive 
``` cpp
void matmul_naive(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0;
            for (int k = 0; k < m; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    for (int i = 0; i < m; i++) {
        float maxx = c[i][0];
        for (int j = 1; j < m; j++) 
            maxx = max(maxx, c[i][j]);
        for (int j = 0; j < m; j++)
            c[i][j] -= maxx;
    }
}
```
![1.1](./self-solution/static/1.1.png)

comments: quite slow

### 2. use register

try to load it into the regsiter before entering the k-loop to avoid unnecessary memory access

c[i][j] is an array element, it must reside in memory. cij is a local scalar variable, it can be placed in a register. registers can only hold scalars, not array elements, which are memory locations.

use pointer 

``` cpp
void matmul_v2(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            float cij = 0.0f;  

            for (int k = 0; k < m; k++) {
                cij += a[i][k] * b[k][j];
            }

            c[i][j] = cij;
        }
    }

    for (int i = 0; i < m; i++) {
        float* row = c[i].data();
        float maxx = -INFINITY;

        for (int j = 0; j < m; j++) {
            float cij = row[j];
            maxx = max(maxx, cij);
        }

        for (int j = 0; j < m; j++) {
            float cij = row[j];
            cij -= maxx;
            row[j] = cij;
        }
    }
}
```
![2.1](./self-solution/static/2.1.png)

comments: theoretically it should be improved. But it seems no use in practice.

### 3. 2 * 2 register blocking



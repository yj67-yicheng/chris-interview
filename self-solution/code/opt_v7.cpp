// 8x8 Register Blocking + Loop Unrolling (K=4) + AVX2/FMA
// 适配 float 类型：利用 8 个寄存器累加 8 行结果

#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

// 核心优化函数：8x8 Blocking
void matmul_avx2_v7(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    // 块大小定义
    // M维度步长为8：一次计算 C 的 8 行
    // N维度步长为8：一次计算 C 的 8 列 (正好填满一个 __m256)
    int m_row_block = 8;
    int m_col_block = 8;
    
    // 边界处理
    int m_end_row = m - (m % m_row_block); 
    int m_end_col = m - (m % m_col_block);
    int k_end = m - (m % 4); // K 维度展开4次

    for (int i = 0; i < m_end_row; i += m_row_block) { // Row loop (Step 8)
        for (int j = 0; j < m_end_col; j += m_col_block) { // Col loop (Step 8)
            
            // 初始化 8 个累加寄存器，分别对应 C[i...i+7][j...j+7]
            __m256 c0 = _mm256_setzero_ps(); // Row i
            __m256 c1 = _mm256_setzero_ps(); // Row i+1
            __m256 c2 = _mm256_setzero_ps(); // Row i+2
            __m256 c3 = _mm256_setzero_ps(); // Row i+3
            __m256 c4 = _mm256_setzero_ps(); // Row i+4
            __m256 c5 = _mm256_setzero_ps(); // Row i+5
            __m256 c6 = _mm256_setzero_ps(); // Row i+6
            __m256 c7 = _mm256_setzero_ps(); // Row i+7

            // K loop (Unrolled by 4)
            for (int k = 0; k < k_end; k += 4) {
                // 预取 B 的行指针，减少 vector[][] 的寻址开销
                const float* b_ptr0 = b[k].data();
                const float* b_ptr1 = b[k+1].data();
                const float* b_ptr2 = b[k+2].data();
                const float* b_ptr3 = b[k+3].data();

                // --- K + 0 ---
                __m256 b_vec = _mm256_loadu_ps(&b_ptr0[j]); // Load B row once
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k]), b_vec, c7);

                // --- K + 1 ---
                b_vec = _mm256_loadu_ps(&b_ptr1[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k+1]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k+1]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k+1]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k+1]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k+1]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k+1]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k+1]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k+1]), b_vec, c7);

                // --- K + 2 ---
                b_vec = _mm256_loadu_ps(&b_ptr2[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k+2]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k+2]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k+2]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k+2]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k+2]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k+2]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k+2]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k+2]), b_vec, c7);

                // --- K + 3 ---
                b_vec = _mm256_loadu_ps(&b_ptr3[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k+3]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k+3]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k+3]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k+3]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k+3]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k+3]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k+3]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k+3]), b_vec, c7);
            }

            // Cleanup loop for K (处理不能被4整除的剩余K)
            for (int k = k_end; k < m; k++) {
                __m256 b_vec = _mm256_loadu_ps(&b[k][j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k]), b_vec, c7);
            }

            // 写回 C 矩阵
            _mm256_storeu_ps(&c[i][j],   _mm256_add_ps(c0, _mm256_loadu_ps(&c[i][j])));
            _mm256_storeu_ps(&c[i+1][j], _mm256_add_ps(c1, _mm256_loadu_ps(&c[i+1][j])));
            _mm256_storeu_ps(&c[i+2][j], _mm256_add_ps(c2, _mm256_loadu_ps(&c[i+2][j])));
            _mm256_storeu_ps(&c[i+3][j], _mm256_add_ps(c3, _mm256_loadu_ps(&c[i+3][j])));
            _mm256_storeu_ps(&c[i+4][j], _mm256_add_ps(c4, _mm256_loadu_ps(&c[i+4][j])));
            _mm256_storeu_ps(&c[i+5][j], _mm256_add_ps(c5, _mm256_loadu_ps(&c[i+5][j])));
            _mm256_storeu_ps(&c[i+6][j], _mm256_add_ps(c6, _mm256_loadu_ps(&c[i+6][j])));
            _mm256_storeu_ps(&c[i+7][j], _mm256_add_ps(c7, _mm256_loadu_ps(&c[i+7][j])));
        }
    }

    // === 边界情况处理 (Edge Cases) ===
    
    // 1. 处理剩余的列 (Columns j from m_end_col to m)
    for (int i = 0; i < m; i++) {
        for (int j = m_end_col; j < m; j++) {
            float sum = c[i][j]; // Load existing value if needed
            for (int k = 0; k < m; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    // 2. 处理剩余的行 (Rows i from m_end_row to m)
    // 只计算 0 到 m_end_col 的列
    for (int i = m_end_row; i < m; i++) {
        for (int j = 0; j < m_end_col; j++) {
            float sum = c[i][j];
            for (int k = 0; k < m; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

double test(int m) {
    vector<vector<float>> a(m, vector<float>(m));
    vector<vector<float>> b(m, vector<float>(m));
    vector<vector<float>> c(m, vector<float>(m)); // 默认为0

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            a[i][j] = (float)(rand() % 100) / 100.0f;
            b[i][j] = (float)(rand() % 100) / 100.0f;
        }
    
    auto start = chrono::high_resolution_clock::now();
    matmul_avx2_v7(a, b, c, m);
    auto end = chrono::high_resolution_clock::now();

    return chrono::duration<double>(end - start).count();
}

int main() {
    int tot = 0;
    cout << "Running Matrix Multiplication (Float) with AVX2 8x8 Blocking..." << endl;

    for (int x = 1; x <= 10; x++) {
        int m = 256 * x;
        double t = test(m);
        cout << "m = " << m << ", time = " << t << "sec" << endl;
        
        tot += t;
    }

    cout << "tot: " << tot << " sec" << endl;
    return 0;
}
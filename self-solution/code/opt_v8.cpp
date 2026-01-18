#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

#define m_blocking 192
#define n_blocking 2048
#define k_blocking 384



void scalar_kernel(int m_start, int m_end, int n_start, int n_end, int k_start, int k_end, vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c) {
    for (int i = m_start; i < m_end; i ++) {
        for (int j = n_start; j < n_end; j ++) {
            float sum = c[i][j];
            for (int k = k_start; k < k_end; k ++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

void macro_kernel(int m_start, int m_end, int n_start, int n_end, int k_start, int k_end, vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c) {
    int m_step = 8;
    int n_step = 8;

    int m_align_end = m_start +  ((m_end - m_start) / m_step) * m_step;
    int n_align_end = n_start + ((n_end - n_start) / n_step) * n_step;
    int k_unroll_end = k_start + ((k_end - k_start) / 4) * 4;

    for (int i = m_start; i < m_align_end; i += m_step) {
        for (int j = n_start; j < n_align_end; j += n_step) {
            
            // 8个累加器，对应 C[i...i+7][j...j+7]
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();
            __m256 c6 = _mm256_setzero_ps();
            __m256 c7 = _mm256_setzero_ps();

            // K 循环展开 (Unroll 4)
            for (int k = k_start; k < k_unroll_end; k += 4) {
                const float* b_ptr0 = b[k].data();
                const float* b_ptr1 = b[k+1].data();
                const float* b_ptr2 = b[k+2].data();
                const float* b_ptr3 = b[k+3].data();

                // K+0
                __m256 b_vec = _mm256_loadu_ps(&b_ptr0[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k]), b_vec, c7);

                // K+1
                b_vec = _mm256_loadu_ps(&b_ptr1[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k+1]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k+1]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k+1]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k+1]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k+1]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k+1]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k+1]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k+1]), b_vec, c7);

                // K+2
                b_vec = _mm256_loadu_ps(&b_ptr2[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k+2]),   b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+1][k+2]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+2][k+2]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+3][k+2]), b_vec, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+4][k+2]), b_vec, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+5][k+2]), b_vec, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+6][k+2]), b_vec, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[i+7][k+2]), b_vec, c7);

                // K+3
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

            // K Loop Cleanup
            for (int k = k_unroll_end; k < k_end; k++) {
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

            // Write back
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

    if (n_align_end < n_end) 
        scalar_kernel(m_start, m_end, n_align_end, n_end, k_start, k_end, a, b, c);
    if (m_align_end < m_end) 
        scalar_kernel(m_align_end, m_end, n_start, n_end, k_start, k_end, a, b, c);
}

void matmul_v8(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    for (int j = 0; j < m; j += n_blocking) {
        int j_len = min(n_blocking, m - j);
        
        for (int k = 0; k < m; k += k_blocking) {
            int k_len = min(k_blocking, m - k);
            
            for (int i = 0; i < m; i += m_blocking) {
                int i_len = min(m_blocking, m - i);

                macro_kernel(i, i + i_len, j, j + j_len, k, k + k_len, a, b, c);
            }

        }
    }

}

double test(int m) {
    vector<vector<float>> a(m, vector<float>(m));
    vector<vector<float>> b(m, vector<float>(m));
    vector<vector<float>> c(m, vector<float>(m));

    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < m; j ++) {
            a[i][j] = (float)(rand() % 100) / 100.0f;
            b[i][j] = (float)(rand() % 100) / 100.0f;
        }
    }

    auto start = chrono::high_resolution_clock::now();
    matmul_v8(a, b, c, m);
    auto end = chrono::high_resolution_clock::now();

    return chrono::duration<double>(end - start).count();
}

int main() {
    int tot = 0;
    cout << "Running MatMul ... " << endl;

    for (int x = 1; x <= 10; x ++) {
        int m = 256 * x;
        double t = test(m);
        tot += t;
        cout << "m = " << m << ", time = " << t <<  " sec" << endl;

    }
    cout << "tot = " << tot << " sec" << endl;
    
    return 0;
}
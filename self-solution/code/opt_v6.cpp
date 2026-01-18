#include <bits/stdc++.h>
#include <immintrin.h> // AVX2 需要包含此头文

using namespace std;

void matmul_v6(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    // block 4 * 8
    int m_row_block = 4;  // use 4 register 
    int m_col_block = 8;  // b_vec has 8 float

    int m_end_row = m - (m % m_row_block);
    int m_end_col = m - (m % m_col_block);
    int k_end = m - (m % 4);  // unroll k dim 4 times
    
    for (int i = 0; i < m_end_row; i += m_row_block) {
        for (int j = 0; j < m_end_col; j += m_col_block) {
            // c[i][j ... j + 7], c[i + 1][j ... j + 7] ... 
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            
            // unroll by 4
            for (int k = 0; k < k_end; k += 4) {
                const float* b_ptr0 = b[k].data();
                const float* b_ptr1 = b[k + 1].data();
                const float* b_ptr2 = b[k + 2].data();
                const float* b_ptr3 = b[k + 3].data();
                
                // k + 0
                __m256 b_vec = _mm256_loadu_ps(&b_ptr0[j]);  // eight float
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k]), b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 1][k]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 2][k]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 3][k]), b_vec, c3);

                // k + 1
                b_vec = _mm256_loadu_ps(&b_ptr1[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k + 1]), b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 1][k + 1]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 2][k + 1]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 3][k + 1]), b_vec, c3);

                // k + 2
                b_vec = _mm256_loadu_ps(&b_ptr2[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k + 2]), b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 1][k + 2]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 2][k + 2]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 3][k + 2]), b_vec, c3);

                // k + 3
                b_vec = _mm256_loadu_ps(&b_ptr3[j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k + 3]), b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 1][k + 3]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 2][k + 3]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 3][k + 3]), b_vec, c3);
                
            }

            for (int k = k_end; k < m; k ++) {
                __m256 b_vec = _mm256_loadu_ps(&b[k][j]);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[i][k]), b_vec, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 1][k]), b_vec, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 2][k]), b_vec, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[i + 3][k]), b_vec, c3);

            }

            _mm256_storeu_ps(&c[i][j], c0);
            _mm256_storeu_ps(&c[i + 1][j], c1);
            _mm256_storeu_ps(&c[i + 2][j], c2);
            _mm256_storeu_ps(&c[i + 3][j], c3);
        }
    }

    for (int i = 0; i < m; i ++) {
        for (int j = m_end_col; j < m; j ++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k ++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    for (int i = m_end_row; i < m; i ++) {
        for (int j = 0; j < m; j ++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k ++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

double test(int m) {
    vector<vector<float>> a(m, vector<float>(m));
    vector<vector<float>> b(m, vector<float>(m));
    vector<vector<float>> c(m, vector<float>(m));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = (float)(rand() % 100) / 100.0f;
            b[i][j] = (float)(rand() % 100) / 100.0f;
        }
    }

    auto start = chrono::high_resolution_clock::now();
    matmul_v6(a, b, c, m);
    auto end = chrono::high_resolution_clock::now();

    return chrono::duration<double>(end - start).count();
}

int main() {
    int tot = 0;
    for (int x = 1; x <= 10; x ++) {
        int m = 256 * x;
        double t = test(m);
        cout << "m = " << m << ", time = " << t << "sec" << endl;
        tot += t;
    }

    cout << "tot: " << tot  << " sec" << endl;
    return 0;
}
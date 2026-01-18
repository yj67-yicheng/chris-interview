#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

#define M_BLOCKING 192
#define N_BLOCKING 2048 
#define K_BLOCKING 384

void packing_a_k9(vector<vector<float>>& src, float* dst, int m_start, int k_start, int m_inc, int k_inc) {
    for (int k = 0; k < k_inc; ++k) {
        float* dst_ptr = dst + k * 8;
        int real_k = k_start + k;
        
        dst_ptr[0] = src[m_start + 0][real_k];
        dst_ptr[1] = src[m_start + 1][real_k];
        dst_ptr[2] = src[m_start + 2][real_k];
        dst_ptr[3] = src[m_start + 3][real_k];
        dst_ptr[4] = src[m_start + 4][real_k];
        dst_ptr[5] = src[m_start + 5][real_k];
        dst_ptr[6] = src[m_start + 6][real_k];
        dst_ptr[7] = src[m_start + 7][real_k];
    }
}

void packing_b_k9(vector<vector<float>>& src, float* dst, int k_start, int n_start, int k_inc, int n_inc) {
    for (int k = 0; k < k_inc; ++k) {
        int real_k = k_start + k;
        _mm256_store_ps(dst + k * 8, _mm256_loadu_ps(&src[real_k][n_start]));
    }
}

void macro_kernel_k9(float* pack_a, float* pack_b, vector<vector<float>>& c, 
                     int m_start, int n_start, int k_inc) {
    
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    float* ptr_a = pack_a;
    float* ptr_b = pack_b;

    int k_end = k_inc - (k_inc % 4);
    
    for (int k = 0; k < k_end; k += 4) {
        __m256 b_vec = _mm256_load_ps(ptr_b);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 0), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 1), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 2), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 3), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 4), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 5), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 6), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 7), b_vec, c7);
        
        ptr_b += 8; ptr_a += 8;

        b_vec = _mm256_load_ps(ptr_b);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 0), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 1), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 2), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 3), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 4), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 5), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 6), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 7), b_vec, c7);

        ptr_b += 8; ptr_a += 8;

        b_vec = _mm256_load_ps(ptr_b);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 0), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 1), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 2), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 3), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 4), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 5), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 6), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 7), b_vec, c7);

        ptr_b += 8; ptr_a += 8;

        b_vec = _mm256_load_ps(ptr_b);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 0), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 1), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 2), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 3), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 4), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 5), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 6), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 7), b_vec, c7);

        ptr_b += 8; ptr_a += 8;
    }

    for (int k = k_end; k < k_inc; k++) {
        __m256 b_vec = _mm256_load_ps(ptr_b);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 0), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 1), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 2), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 3), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 4), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 5), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 6), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(ptr_a + 7), b_vec, c7);
        ptr_b += 8; ptr_a += 8;
    }

    _mm256_storeu_ps(&c[m_start + 0][n_start], _mm256_add_ps(c0, _mm256_loadu_ps(&c[m_start + 0][n_start])));
    _mm256_storeu_ps(&c[m_start + 1][n_start], _mm256_add_ps(c1, _mm256_loadu_ps(&c[m_start + 1][n_start])));
    _mm256_storeu_ps(&c[m_start + 2][n_start], _mm256_add_ps(c2, _mm256_loadu_ps(&c[m_start + 2][n_start])));
    _mm256_storeu_ps(&c[m_start + 3][n_start], _mm256_add_ps(c3, _mm256_loadu_ps(&c[m_start + 3][n_start])));
    _mm256_storeu_ps(&c[m_start + 4][n_start], _mm256_add_ps(c4, _mm256_loadu_ps(&c[m_start + 4][n_start])));
    _mm256_storeu_ps(&c[m_start + 5][n_start], _mm256_add_ps(c5, _mm256_loadu_ps(&c[m_start + 5][n_start])));
    _mm256_storeu_ps(&c[m_start + 6][n_start], _mm256_add_ps(c6, _mm256_loadu_ps(&c[m_start + 6][n_start])));
    _mm256_storeu_ps(&c[m_start + 7][n_start], _mm256_add_ps(c7, _mm256_loadu_ps(&c[m_start + 7][n_start])));
}

void scalar_kernel_k9(int m_start, int m_end, int n_start, int n_end, int k_start, int k_end,
                      vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c) {
    for (int i = m_start; i < m_end; i++) {
        for (int j = n_start; j < n_end; j++) {
            float sum = c[i][j];
            for (int k = k_start; k < k_end; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

void mydgemm_cpu_v9(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    float* b_buffer = (float*)_mm_malloc(K_BLOCKING * N_BLOCKING * sizeof(float), 4096);
    float* a_buffer = (float*)_mm_malloc(M_BLOCKING * K_BLOCKING * sizeof(float), 4096);

    for (int n = 0; n < m; n += N_BLOCKING) {
        int n_len = min(N_BLOCKING, m - n);
        
        for (int k = 0; k < m; k += K_BLOCKING) {
            int k_len = min(K_BLOCKING, m - k);
            
            int n_aligned = n + (n_len / 8) * 8;
            for (int j = n; j < n_aligned; j += 8) {
                packing_b_k9(b, b_buffer + (j - n) * k_len, k, j, k_len, 8);
            }

            for (int i = 0; i < m; i += M_BLOCKING) {
                int i_len = min(M_BLOCKING, m - i);
                
                int i_aligned = i + (i_len / 8) * 8;

                for (int ii = i; ii < i_aligned; ii += 8) {
                    packing_a_k9(a, a_buffer, ii, k, 8, k_len);

                    for (int jj = n; jj < n_aligned; jj += 8) {
                        macro_kernel_k9(a_buffer, b_buffer + (jj - n) * k_len, c, ii, jj, k_len);
                    }
                }
                
                if (i_aligned < i + i_len) {
                    scalar_kernel_k9(i_aligned, i + i_len, n, n + n_len, k, k + k_len, a, b, c);
                }
            }
            
            if (n_aligned < n + n_len) {
                 scalar_kernel_k9(0, m, n_aligned, n + n_len, k, k + k_len, a, b, c);
            }
        }
    }

    _mm_free(a_buffer);
    _mm_free(b_buffer);
}

double test(int m) {
    vector<vector<float>> a(m, vector<float>(m));
    vector<vector<float>> b(m, vector<float>(m));
    vector<vector<float>> c(m, vector<float>(m));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            a[i][j] = (float)(rand() % 100) / 100.0f;
            b[i][j] = (float)(rand() % 100) / 100.0f;
        }
    
    auto start = chrono::high_resolution_clock::now();
    mydgemm_cpu_v9(a, b, c, m);
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

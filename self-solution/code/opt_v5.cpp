// 4 * 4 register blocking

#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

void matmul_v5(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    int m4 = m - (m % 4);  // 4的整数倍

    for (int i = 0; i < m4; i += 4) {  // row
        for (int j = 0; j < m4; j += 4) {  // col
            // 4 blocks of c
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            
            for (int k = 0; k < m; k++) {  // m个矩阵相加
                __m128 b_vec = _mm_loadu_ps(&b[k][j]);

                // a col
                float a0 = a[i][k];
                float a1 = a[i + 1][k];
                float a2 = a[i + 2][k];
                float a3 = a[i + 3][k];

                // broadcast a to 4 lanes for b_vec
                __m128 va0 = _mm_set1_ps(a0);
                __m128 va1 = _mm_set1_ps(a1);
                __m128 va2 = _mm_set1_ps(a2);
                __m128 va3 = _mm_set1_ps(a3);

                // sum up to c, c is row vector
                c0 = _mm_add_ps(c0, _mm_mul_ps(va0, b_vec));
                c1 = _mm_add_ps(c1, _mm_mul_ps(va1, b_vec));
                c2 = _mm_add_ps(c2, _mm_mul_ps(va2, b_vec));
                c3 = _mm_add_ps(c3, _mm_mul_ps(va3, b_vec));
                
            }

            // write back
            _mm_storeu_ps(&c[i][j], c0);
            _mm_storeu_ps(&c[i + 1][j], c1);
            _mm_storeu_ps(&c[i + 2][j], c2);
            _mm_storeu_ps(&c[i + 3][j], c3);
            

        }
    }

        // there some edge cases

        // for row
        if (m4 < m) {
            for (int i = m4; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    float cij = 0.0f;
                    for (int k = 0; k < m; k++) {
                        cij += a[i][k] * b[k][j];
                    }
                    c[i][j] = cij;
                }
            }
        }

        // for col
        if (m4 < m) {
            for (int i = 0; i < m; i++) {
                for (int j = m4; j < m; j++) {
                    float cij = 0.0f;
                    for (int k = 0; k < m; k++) {
                        cij += a[i][k] * b[k][j];
                    }
                    c[i][j] = cij;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            float* row = c[i].data();
            float maxx = row[0];

            for (int j = 1; j < m; j++) {
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

double test(int m) {
    vector<vector<float>> a(m, vector<float>(m));
    vector<vector<float>> b(m, vector<float>(m));
    vector<vector<float>> c(m, vector<float>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            a[i][j] = rand() / float(RAND_MAX);
            b[i][j] = rand() / float(RAND_MAX);
        }
    
    auto start = chrono::high_resolution_clock::now();
    matmul_v5(a, b, c, m);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

int main() {
    for (int x = 1; x <= 10; x++) {
        int m = 256 * x;
        double t = test(m);
        cout << "m = " << m << ", time = " << t << "sec" << endl;
    }
}


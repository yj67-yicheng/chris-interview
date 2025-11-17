// 4 * 4 register blocking + SIMD style (float version of the double code you showed)

#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;

void matmul_v4(vector<vector<float>>& a,
               vector<vector<float>>& b,
               vector<vector<float>>& c,
               int m) {
    // 主块大小：按 4 对齐
    int m4 = m - (m % 4);

    // ------------------------------------
    //  4x4 block + SIMD（按 mydgemm_v5 思路）
    // ------------------------------------
    for (int i = 0; i < m4; i += 4) {
        for (int j = 0; j < m4; j += 4) {

            // 每一行一个 4 元向量：c[i][j..j+3]
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();

            for (int k = 0; k < m; k++) {
                // B 的一行切片：b[k][j..j+3] 是连续的
                __m128 b_vec = _mm_loadu_ps(&b[k][j]);

                // A 的 4 个标量：a[i+r][k]
                float a0 = a[i][k];
                float a1 = a[i + 1][k];
                float a2 = a[i + 2][k];
                float a3 = a[i + 3][k];

                // 广播 A 的标量到 4-lane 向量，然后做外积更新
                __m128 va0 = _mm_set1_ps(a0);
                __m128 va1 = _mm_set1_ps(a1);
                __m128 va2 = _mm_set1_ps(a2);
                __m128 va3 = _mm_set1_ps(a3);

                c0 = _mm_add_ps(c0, _mm_mul_ps(va0, b_vec));
                c1 = _mm_add_ps(c1, _mm_mul_ps(va1, b_vec));
                c2 = _mm_add_ps(c2, _mm_mul_ps(va2, b_vec));
                c3 = _mm_add_ps(c3, _mm_mul_ps(va3, b_vec));
            }

            // 写回 4 行各自的 4 个元素
            _mm_storeu_ps(&c[i][j], c0);
            _mm_storeu_ps(&c[i + 1][j], c1);
            _mm_storeu_ps(&c[i + 2][j], c2);
            _mm_storeu_ps(&c[i + 3][j], c3);
        }
    }

    // ------------------------------------
    //  边界处理：如果 m 不是 4 的倍数
    //  （思路和 mydgemm_cpu_v5 里的 opt_k5 一样）
    // ------------------------------------

    // 下面这些边界循环在你的测试里 m=256*x，本质上不会触发
    // 但写上更完整一些

    // 处理剩余的行（从 m4 到 m-1），全部列
    if (m4 < m) {
        for (int i = m4; i < m; i++) {
            for (int j = 0; j < m; j++) {
                float sum = 0.0f;
                for (int k = 0; k < m; k++)
                    sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        }
    }

    // 处理剩余的列（从 m4 到 m-1），前 m4 行
    if (m4 < m) {
        for (int i = 0; i < m4; i++) {
            for (int j = m4; j < m; j++) {
                float sum = 0.0f;
                for (int k = 0; k < m; k++)
                    sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        }
    }

    // ------------------------------------
    //  行最大值减法：保留你原来的结构
    // ------------------------------------
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
    matmul_v4(a, b, c, m);
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

// 2 * 2 register blocking

#include <bits/stdc++.h>
using namespace std;

void matmul_v3(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    for (int i = 0; i < m; i += 2) {
        for (int j = 0; j < m; j += 2) {
            // a: 2*1, b: 1*2, c: 2*2
            // outer product 
            
            float c00 = 0.0f, c01 = 0.0f;
            float c10 = 0.0f, c11 = 0.0f;

            for (int k = 0; k < m; k++) {
                float a0 = a[i][k];
                float a1 = (i + 1 < m) ? a[i + 1][k] : 0.0f;

                float b0 = b[k][j];
                float b1 = (j + 1 < m) ? b[k][j + 1] : 0.0f;

                c00 += a0 * b0;
                c01 += a0 * b1;
                c10 += a1 * b0;
                c11 += a1 * b1;
            }

            c[i][j] = c00;
            if (j + 1 < m) c[i][j + 1] = c01;
            if (i + 1 < m) {
                c[i + 1][j] = c10;
                if (j + 1 < m) c[i + 1][j + 1] = c11;
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
    matmul_v3(a, b, c, m);
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


// 4 * 4 register blocking

#include <bits/stdc++.h>
using namespace std;

void matmul_v4(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < m; j += 4) {

            float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
            float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
            float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
            float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

            for (int k = 0; k < m; k++) {

                float a0 = a[i][k];
                float a1 = (i + 1 < m) ? a[i + 1][k] : 0.0f;
                float a2 = (i + 2 < m) ? a[i + 2][k] : 0.0f;
                float a3 = (i + 3 < m) ? a[i + 3][k] : 0.0f;


                float b0 = b[k][j];
                float b1 = (j + 1 < m) ? b[k][j + 1] : 0.0f;
                float b2 = (j + 2 < m) ? b[k][j + 2] : 0.0f;
                float b3 = (j + 3 < m) ? b[k][j + 3] : 0.0f;

                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;

    
                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;


                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;

            
                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
            }


            if (i < m && j < m) c[i][j] = c00;
            if (i < m && j + 1 < m) c[i][j + 1] = c01;
            if (i < m && j + 2 < m) c[i][j + 2] = c02;
            if (i < m && j + 3 < m) c[i][j + 3] = c03;

            if (i + 1 < m && j < m) c[i + 1][j] = c10;
            if (i + 1 < m && j + 1 < m) c[i + 1][j + 1] = c11;
            if (i + 1 < m && j + 2 < m) c[i + 1][j + 2] = c12;
            if (i + 1 < m && j + 3 < m) c[i + 1][j + 3] = c13;

            if (i + 2 < m && j < m) c[i + 2][j] = c20;
            if (i + 2 < m && j + 1 < m) c[i + 2][j + 1] = c21;
            if (i + 2 < m && j + 2 < m) c[i + 2][j + 2] = c22;
            if (i + 2 < m && j + 3 < m) c[i + 2][j + 3] = c23;

            if (i + 3 < m && j < m) c[i + 3][j] = c30;
            if (i + 3 < m && j + 1 < m) c[i + 3][j + 1] = c31;
            if (i + 3 < m && j + 2 < m) c[i + 3][j + 2] = c32;
            if (i + 3 < m && j + 3 < m) c[i + 3][j + 3] = c33;
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


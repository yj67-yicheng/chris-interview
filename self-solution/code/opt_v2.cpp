// register reuse

#include <bits/stdc++.h>
using namespace std;

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
    matmul_v2(a, b, c, m);
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


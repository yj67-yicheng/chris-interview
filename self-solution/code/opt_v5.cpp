// 4 * 4 register blocking

#include <bits/stdc++.h>
using namespace std;

void matmul_v5(vector<vector<float>>& a, vector<vector<float>>& b, vector<vector<float>>& c, int m) {
    
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


#include <vector>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;
using Clock = chrono::high_resolution_clock;

// Размерности
const int IN_DIM = 128;
const int HIDDEN_DIM = 32;
const int OUT_DIM = 1;
const int BATCH = 8;

// Функция ReLU и её производная
inline float relu(float x) { return x > 0 ? x : 0; }
inline float relu_deriv(float x) { return x > 0 ? 1 : 0; }

int main() {
    // Инициализация RNG
    mt19937 gen(0);
    normal_distribution<float> dist(0.0f, 1.0f);

    // Веса и смещения
    vector<float> W1(IN_DIM * HIDDEN_DIM), b1(HIDDEN_DIM);
    vector<float> W2(HIDDEN_DIM * OUT_DIM), b2(OUT_DIM);
    
    for (auto &w : W1) w = dist(gen);
    for (auto &w : W2) w = dist(gen);

    // Данные
    vector<float> X(BATCH * IN_DIM), Y(BATCH * OUT_DIM);
    for (auto &x : X) x = dist(gen);
    for (auto &y : Y) y = dist(gen);

    // Буферы
    vector<float> Z1(BATCH * HIDDEN_DIM), H(BATCH * HIDDEN_DIM);
    vector<float> Y_pred(BATCH * OUT_DIM);
    vector<float> dW1(IN_DIM * HIDDEN_DIM), db1(HIDDEN_DIM);
    vector<float> dW2(HIDDEN_DIM * OUT_DIM), db2(OUT_DIM);

    // Разогрев
    auto run_once = [&]() {
        // Forward
        for (int n=0; n<BATCH; ++n) {
            for (int j=0; j<HIDDEN_DIM; ++j) {
                float sum = b1[j];
                for (int i=0; i<IN_DIM; ++i)
                    sum += X[n*IN_DIM + i] * W1[i*HIDDEN_DIM + j];
                Z1[n*HIDDEN_DIM + j] = sum;
                H[n*HIDDEN_DIM + j] = relu(sum);
            }
            for (int k=0; k<OUT_DIM; ++k) {
                float sum = b2[k];
                for (int j=0; j<HIDDEN_DIM; ++j)
                    sum += H[n*HIDDEN_DIM + j] * W2[j*OUT_DIM + k];
                Y_pred[n*OUT_DIM + k] = sum;
            }
        }
        // Backward (MSE loss)
        for (int n=0; n<BATCH; ++n) {
            for (int k=0; k<OUT_DIM; ++k) {
                float dy = (Y_pred[n*OUT_DIM + k] - Y[n*OUT_DIM + k]);
                db2[k] += dy;
                for (int j=0; j<HIDDEN_DIM; ++j)
                    dW2[j*OUT_DIM + k] += H[n*HIDDEN_DIM + j] * dy;
                for (int j=0; j<HIDDEN_DIM; ++j) {
                    float dz = W2[j*OUT_DIM + k] * dy * relu_deriv(Z1[n*HIDDEN_DIM + j]);
                    db1[j] += dz;
                    for (int i=0; i<IN_DIM; ++i)
                        dW1[i*HIDDEN_DIM + j] += X[n*IN_DIM + i] * dz;
                }
            }
        }
    };

    // Разогрев
    run_once();
    run_once();

    // Замеры
    vector<double> times;
    for (int it=0; it<10; ++it) {
        // Сброс градиентов
        fill(dW1.begin(), dW1.end(), 0);
        fill(db1.begin(), db1.end(), 0);
        fill(dW2.begin(), dW2.end(), 0);
        fill(db2.begin(), db2.end(), 0);

        auto t0 = Clock::now();
        run_once();
        auto t1 = Clock::now();
        times.push_back(chrono::duration<double, milli>(t1 - t0).count());
    }

    cout << "C++ no-libs ms per iter:
";
    for (auto t : times) cout << t << ' ';
    cout << '
';
    return 0;
}
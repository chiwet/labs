#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <omp.h>
#include <string>
#include <thread>

using std::rand;
const int COUT_N = 5;
const int n = 5000;

void print_matrix_part(std::vector<float> M) {
    for (int i = 0; i < COUT_N; i++) {
        for (int j = 0; j < COUT_N; j++) {
            std::cout << M[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}


int main() {
    setlocale(LC_ALL, "");
    srand(static_cast<unsigned int>(time(0)));
    std::cout << "Hi\n";
    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = rand() % 10 * 10;
            B[i * n + j] = rand() % 10 * 10;
        }
    }
    std::cout << "Matrix 1 part\n";
    print_matrix_part(A);
    std::cout << "\n\n";
    std::cout << "Matrix 2 part\n";
    print_matrix_part(B);
    std::cout << "\n\n";
    std::cout << "Multiplying will start in 5 secs\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));
    auto begin = std::chrono::steady_clock::now();

    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        // Указатель на i-ю строку матрицы C
        float* c = C.data() + i * n;
        for (int j = 0; j < n; ++j)
            c[j] = 0;
        for (int k = 0; k < n; ++k)
        {
            // Указатель на k-ю строку матрицы B
            const float* b = B.data() + k * n;
            float a = A[i * n + k];

            for (int j = 0; j < n; ++j)
                c[j] += a * b[j];
        }
        #pragma omp critical
        {
            std::cout << "На очереди i-" << i + 1 << " Строка (поток "
                << omp_get_thread_num() << ")\n";
        }
    }
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "The time: " << elapsed_ms.count() << " ms\n";
    std::cout << "Multiplied matrix part\n";
    print_matrix_part(C);
    std::cout << "\n\n";
    return 0;
}
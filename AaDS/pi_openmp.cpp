#include <iostream>
#include <string>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <iomanip>

long double compute_pi(unsigned long long n) {
    long double sum = 0.0;
    long double step = 1.0 / static_cast<double>(n);
    long double x;
    #pragma omp parallel for private(x), reduction(+:sum) num_threads(8)
    for (long long i = 0; i < n; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    return sum * step;
}

int main()
{
    long double eps;
    std::string in;
    std::cout << "Enter eps value ";
    std::cin >> in;
    eps = atof(in.c_str());
    std::cout << eps;
    std::cout << std::setprecision(20);
    if (eps >= 1.0 || eps <= 0.0)
    {
        std::cout << ("Invalid input for eps.\n");
        return 1;
    }

    unsigned long long n = 100;
    long double pi_prev = 3.0;
    long double pi_curr = compute_pi(n);
    std::cout << "Calculating pi...\n";
    auto start = std::chrono::steady_clock::now();
    while (fabs(pi_curr - pi_prev) > eps)
    {
        pi_prev = pi_curr;
        n *= 2;
        pi_curr = compute_pi(n);
        
        std::cout << "Current pi - " << pi_curr << "\n";
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Known pi = 3.14159265358979323846\n";
    std::cout << "Calculated pi = " << pi_curr << "\n\n";
    std::cout << "";
    std::cout << "Number of squares n = " << n << "\n";
    std::cout << "OpenMP Time:" << elapsed.count() << "s\n";
    long double pi_true = 3.141592653589793238462643383279502884L;
    std::cout << "Abs diff with known pi = " << fabsl(pi_curr - pi_true);
}


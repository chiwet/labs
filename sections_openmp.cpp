#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <cmath>

void print_sum_squares(int n)
{
    if (n <= 0)
    {
        std::cout << "undefined for n <= 0";
        return;
    }
    int rem = n;
    int first = 1;
    while (rem > 0)
    {
        int root = (int)sqrt((double)rem);
        int sq = root * root;
        if (!first)
        {
            std::cout << " + ";
        }
        std::cout << root << "^2";
        rem -= sq;
        first = 0;
    }
}

unsigned long long fib(unsigned int n)
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    unsigned long long a = 0, b = 1, c;
    for (unsigned int i = 2; i <= n; ++i)
    {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}
int is_prime(int x)
{
    if (x < 2) return 0;
    if (x == 2) return 1;
    if (x % 2 == 0) return 0;
    int lim = (int)sqrt((double)x);
    for (int i = 3; i <= lim; i += 2)
    {
        if (x % i == 0) return 0;
    }
    return 1;
}
int nth_prime(int n)
{
    if (n <= 0) return -1;
    int count = 0;
    int candidate = 1;
    while (count < n)
    {
        ++candidate;
        if (is_prime(candidate))
        {
            ++count;
        }
    }
    return candidate;
}
long long sum_divisors(int n)
{
    if (n == 0) return 0;
    if (n < 0) {
        n = -n;
    }
    long long sum = 0;
    int lim = (int)sqrt((double)n);
    for (int i = 1; i <= lim; ++i)
    {
        if (n % i == 0)
        {
            sum += i;
            int j = n / i;
            if (j != i) sum += j;
        }
    }
    return sum;
}
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << "input.txt\n";
        return 1;
    }
    std::ifstream in;
    in.open(argv[1]);
    if (!in) {
        std::cerr << "Error opening file or file does not exist." << std::endl;
        return -1;
    }

    std::vector<int> a;
    int x;
    while (in >> x) {
        a.push_back(x);
    }
    in.close();

    int next_idx = 0;

#pragma omp parallel sections shared(a, next_idx)
    {
        //1: Squaeres sum
        #pragma omp section
        {
            for (;;) {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < static_cast<int>(a.size())) {
                        my_idx = next_idx;
                        ++next_idx;
                    }
                    else {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                #pragma omp critical(print)
                {
                    std::cout << "1st section> a[" << my_idx << "] = " << n << ": ";
                    print_sum_squares(n);
                    std::cout << "\n";
                }
            }
        }

        //2: Fib number
        #pragma omp section
        {
            for (;;) {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < static_cast<int>(a.size())) {
                        my_idx = next_idx;
                        ++next_idx;
                    }
                    else {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                #pragma omp critical(print)
                {
                    if (n < 0) {
                        std::cout << "2nd section> a[" << my_idx << "] = " << n
                            << ": Fibonacci undefined for n < 0\n";
                    }
                    else {
                        unsigned long long f_val = fib(static_cast<unsigned int>(n));
                        std::cout << "2nd section> a[" << my_idx << "] = " << n
                            << ": F(" << n << ") = " << f_val << "\n";
                    }
                }
            }
        }

        //3: n-th prime number
        #pragma omp section
        {
            for (;;) {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < static_cast<int>(a.size())) {
                        my_idx = next_idx;
                        ++next_idx;
                    }
                    else {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                #pragma omp critical(print)
                {
                    if (n <= 0) {
                        std::cout << "3rd section> a[" << my_idx << "] = " << n
                            << ": nth prime undefined for n <= 0" << "\n";
                    }
                    else {
                        int p = nth_prime(n);
                        std::cout << "3rd section> a[" << my_idx << "] = " << n
                            << ": " << n << "-th prime = " << p << "\n";
                    }
                }
            }
        }

        //4: Sum of dividers
        #pragma omp section
        {
            for (;;) {
                int my_idx;
                #pragma omp critical(queue)
                {
                    if (next_idx < static_cast<int>(a.size())) {
                        my_idx = next_idx;
                        ++next_idx;
                    }
                    else {
                        my_idx = -1;
                    }
                }

                if (my_idx == -1)
                    break;

                int n = a[my_idx];
                #pragma omp critical(print)
                {
                    long long s = sum_divisors(n);

                    std::cout << "4th section> a[" << my_idx << "] = " << n
                        << ": sum of divisors = " << s << "\n";
                }
            }
        }
    }

    return 0;
}


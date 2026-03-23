#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define N 1000  // Matrix size N x N
#define PRINT_PART_SIZE 5  // Size of the matrix part to print

void print_matrix_part(const char* name, const std::vector<float>& M) {
    std::cout << name << ":\n";
    for (int i = 0; i < PRINT_PART_SIZE; i++) {
        for (int j = 0; j < PRINT_PART_SIZE; j++) {
            std::cout << M[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matmul_gpu(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    size_t bytes = N * N * sizeof(float);

    // Memory allocation and initialization using vectors
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_cpu(N * N);
    std::vector<float> C_gpu(N * N);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand() % 10);
        B[i] = static_cast<float>(rand() % 10);
    }

    // Print parts of matrices A and B
    print_matrix_part("A (part)", A);
    print_matrix_part("B (part)", B);

    // Memory allocation on GPU
    float* dA, * dB, * dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    // Copy data to GPU
    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    std::cout << "Launching GPU kernel...\n";

    auto gstart = std::chrono::high_resolution_clock::now();
    matmul_gpu << <grid, block >> > (dA, dB, dC);
    cudaDeviceSynchronize();
    auto gend = std::chrono::high_resolution_clock::now();

    double gpu_time = std::chrono::duration<double>(gend - gstart).count();
    std::cout << "GPU time: " << gpu_time << " seconds\n";

    // Copy result back to CPU
    cudaMemcpy(C_gpu.data(), dC, bytes, cudaMemcpyDeviceToHost);
    print_matrix_part("C_gpu (part)", C_gpu);

    std::cout << "Computing on CPU...\n";

    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end - start).count();

    std::cout << "CPU time: " << cpu_time << " seconds\n";
    print_matrix_part("C_cpu (part)", C_cpu);

    std::cout << "GPU speedup: " << (cpu_time / gpu_time) << "x\n";

    // Verify results
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (std::abs(C_cpu[i] - C_gpu[i]) > 1e-3) {
            success = false;
            break;
        }
    }
    std::cout << "Verification: " << (success ? "SUCCESS" : "FAILURE") << "\n";

    // Clean up GPU memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
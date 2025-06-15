#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include "../555/555.cuh"
#include "../555/555_reg_bound.cuh"

#define N 5
#define NUM_MULS 10000000  // Number of independent 5x5 multiplications
#define REPEAT 10         // Number of times to repeat the benchmark for averaging

__global__ void kernel_func1(const float* A, const float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_MULS) return;
    matmul5x5_opt(A + idx * N * N, B + idx * N * N, C + idx * N * N);
}

__global__ void kernel_func2(const float* A, const float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_MULS) return;
    matmul5x5_opt_reg_bound(A + idx * N * N, B + idx * N * N, C + idx * N * N);
}

__global__ void kernel_naive(const float* A, const float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_MULS) return;
    const float* a = A + idx * N * N;
    const float* b = B + idx * N * N;
    float* c = C + idx * N * N;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += a[i * N + k] * b[k * N + j];
            c[i * N + j] = sum;
        }
}

void benchmark_kernel(void (*kernel)(const float*, const float*, float*), const char* name, const float* d_A, const float* d_B, float* d_C) {
    int threads = 256;
    int blocks = (NUM_MULS + threads - 1) / threads;
    cudaDeviceSynchronize();
    double total_ms = 0.0;
    for (int r = 0; r < REPEAT; ++r) {
        cudaMemset((void*)d_C, 0, NUM_MULS * N * N * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<blocks, threads>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
    }
    printf("%s: avg %f ms for %d runs (%d multiplications per run)\n", name, total_ms / REPEAT, REPEAT, NUM_MULS);
}

int main() {
    size_t total = NUM_MULS * N * N;
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_C(total);
    for (size_t i = 0; i < total; ++i) {
        h_A[i] = (i % 13) * 0.5f + 1.0f;
        h_B[i] = (i % 7) * 1.1f + 2.0f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, total * sizeof(float));

    printf("Benchmarking %d x 5x5 matrix multiplications (%d runs):\n", NUM_MULS, REPEAT);
    benchmark_kernel(kernel_func1, "matmul5x5_opt", d_A, d_B, d_C);
    benchmark_kernel(kernel_func2, "matmul5x5_opt_reg_bound", d_A, d_B, d_C);
    benchmark_kernel(kernel_naive, "naive", d_A, d_B, d_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

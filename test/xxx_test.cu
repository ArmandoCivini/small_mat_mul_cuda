#include <iostream>
#include <cmath>    // for fabs()
#include <cstdio>   // for printf()
#include <vector>
#include "../555/555.cuh"  // Update this header as needed for your generic matmul function
#include "../666/666.cuh"  // Update this header as needed for your generic matmul function

__global__ void test_matmul_kernel(const float* A, const float* B, float* C) {
    // Replace with your generic matrix multiplication function
    // matmul_generic<N>(A, B, C);
    matmul6x6_opt(A, B, C); // Placeholder - update this
}

void test_matrix_multiplication(int N) {
    printf("Testing %dx%d matrix multiplication:\n", N, N);
    
    // Allocate host matrices
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);
    std::vector<float> h_expected(N * N, 0.0f);

    // Initialize matrices with test data
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i + 1;
        h_B[i] = (i % N) + 1;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    cudaMemset(d_C, 0, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    test_matmul_kernel<<<1, 1>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute expected result on CPU
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                h_expected[i * N + j] += h_A[i * N + k] * h_B[k * N + j];

    // Compare results
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C[i] - h_expected[i]) > 1e-5) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], h_expected[i]);
            correct = false;
        }
    }

    if (correct) printf("Test passed for %dx%d!\n", N, N);
    else printf("Test failed for %dx%d!\n", N, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // Test different matrix sizes
    test_matrix_multiplication(6);
    
    return 0;
}

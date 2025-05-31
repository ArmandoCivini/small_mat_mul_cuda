#include <iostream>
#include <cmath>    // for fabs()
#include <cstdio>   // for printf()
#include "../555/555.cuh"  // Assuming this header contains the declaration of matmul5x5_opt

__global__ void test_matmul5x5_kernel(const float* A, const float* B, float* C) {
    matmul5x5_opt(A, B, C);
}

int main() {
    float h_A[25] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    float h_B[25] = {
        25, 24, 23, 22, 21,
        20, 19, 18, 17, 16,
        15, 14, 13, 12, 11,
        10, 9, 8, 7, 6,
        5, 4, 3, 2, 1
    };

    float h_C[25];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 25 * sizeof(float));
    cudaMalloc(&d_B, 25 * sizeof(float));
    cudaMalloc(&d_C, 25 * sizeof(float));

    cudaMemcpy(d_A, h_A, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 25 * sizeof(float), cudaMemcpyHostToDevice);


    test_matmul5x5_kernel<<<1, 1>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();


    cudaMemcpy(h_C, d_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute expected result on CPU
    float h_expected[25] = {0};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k)
                h_expected[i * 5 + j] += h_A[i * 5 + k] * h_B[k * 5 + j];

    // Compare
    bool correct = true;
    for (int i = 0; i < 25; ++i) {
        if (fabs(h_C[i] - h_expected[i]) > 1e-5) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], h_expected[i]);
            correct = false;
        }
    }

    if (correct) printf("Test passed!\n");
    else printf("Test failed!\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
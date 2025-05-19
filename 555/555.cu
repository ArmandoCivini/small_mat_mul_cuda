// matrix_5x5_opt.h
#pragma once

// A, B, and C are all length‐25 arrays in row‐major order (row*5 + col).
// This function must be inlined for best performance.

__device__ __forceinline__ void matmul5x5_opt(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C) {
    // For readability, give each element a short alias. For instance:
    // A00 = A[0], A01 = A[1], … A04 = A[4]
    // A10 = A[5], …  A44 = A[24], and similarly for B.
    //
    // Then write each of your 25 output formulas exactly in terms of these aliases.
    //
    float A00 = A[0],  A01 = A[1],  A02 = A[2],  A03 = A[3],  A04 = A[4];
    float A10 = A[5],  A11 = A[6],  A12 = A[7],  A13 = A[8],  A14 = A[9];
    float A20 = A[10], A21 = A[11], A22 = A[12], A23 = A[13], A24 = A[14];
    float A30 = A[15], A31 = A[16], A32 = A[17], A33 = A[18], A34 = A[19];
    float A40 = A[20], A41 = A[21], A42 = A[22], A43 = A[23], A44 = A[24];

    float B00 = B[0],  B01 = B[1],  B02 = B[2],  B03 = B[3],  B04 = B[4];
    float B10 = B[5],  B11 = B[6],  B12 = B[7],  B13 = B[8],  B14 = B[9];
    float B20 = B[10], B21 = B[11], B22 = B[12], B23 = B[13], B24 = B[14];
    float B30 = B[15], B31 = B[16], B32 = B[17], B33 = B[18], B34 = B[19];
    float B40 = B[20], B41 = B[21], B42 = B[22], B43 = B[23], B44 = B[24];

}
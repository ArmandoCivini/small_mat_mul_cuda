#pragma once

// A, B, and C are all length‐25 arrays in row‐major order (row*5 + col).
// This function must be inlined for best performance.

__device__ __forceinline__ void matmul5x5_opt(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C) {
// Expression: (a13 - a32 - a33 - a34 - a35 + a53) (b12 + b21 - b22 - b24 - b25 - b32 + b42) (-c13 - c25 + c45)
float termA1 = A[2] - A[11] - A[12] - A[13] - A[14] + A[22];
float termB1 = B[1] + B[5] - B[6] - B[8] - B[9] - B[11] + B[16];
float result1 = termA1 * termB1;
C[10] -= result1; // c13
C[21] -= result1; // c25
C[23] += result1; // c45

// Expression: (-a13 - a25 + a45) (b13 - b32 - b33 - b34 - b35 + b53) (c12 + c21 - c22 - c24 - c25 - c32 + c42)
float termA2 = -A[2] - A[9] + A[19];
float termB2 = B[2] - B[11] - B[12] - B[13] - B[14] + B[22];
float result2 = termA2 * termB2;
C[5] += result2; // c12
C[1] += result2; // c21
C[6] -= result2; // c22
C[16] -= result2; // c24
C[21] -= result2; // c25
C[7] -= result2; // c32
C[8] += result2; // c42

// Expression: (a12 + a21 - a22 - a24 - a25 - a32 + a42) (-b13 - b25 + b45) (c13 - c32 - c33 - c34 - c35 + c53)
float termA3 = A[1] + A[5] - A[6] - A[8] - A[9] - A[11] + A[16];
float termB3 = -B[2] - B[9] + B[19];
float result3 = termA3 * termB3;
C[10] += result3; // c13
C[7] -= result3; // c32
C[12] -= result3; // c33
C[17] -= result3; // c34
C[22] -= result3; // c35
C[14] += result3; // c53

// Expression: (-a12 - a14) (b22 + b24 + b25) (-c11 + c14 + c15 + c31 - c34 - c35 - c51 + c54 + c55)
float termA4 = -A[1] - A[3];
float termB4 = B[6] + B[8] + B[9];
float result4 = termA4 * termB4;
C[0] -= result4; // c11
C[15] += result4; // c14
C[20] += result4; // c15
C[2] += result4; // c31
C[17] -= result4; // c34
C[22] -= result4; // c35
C[4] -= result4; // c51
C[19] += result4; // c54
C[24] += result4; // c55

// Expression: (-a11 + a14 + a15 + a31 - a34 - a35 - a51 + a54 + a55) (-b12 - b14) (c22 + c24 + c25)
float termA5 = -A[0] + A[3] + A[4] + A[10] - A[13] - A[14] - A[20] + A[23] + A[24];
float termB5 = -B[1] - B[3];
float result5 = termA5 * termB5;
C[6] += result5; // c22
C[16] += result5; // c24
C[21] += result5; // c25

// Expression: (a22 + a24 + a25) (-b11 + b14 + b15 + b31 - b34 - b35 - b51 + b54 + b55) (-c12 - c14)
float termA6 = A[6] + A[8] + A[9];
float termB6 = -B[0] + B[3] + B[4] + B[10] - B[13] - B[14] - B[20] + B[23] + B[24];
float result6 = termA6 * termB6;
C[5] -= result6; // c12
C[15] -= result6; // c14

// Expression: (-a13 - a53) b31 (-c13 - c15 + c35)
float termA7 = -A[2] - A[22];
float termB7 = B[10];
float result7 = termA7 * termB7;
C[10] -= result7; // c13
C[20] -= result7; // c15
C[22] += result7; // c35

// Expression: (-a13 - a15 + a35) (-b13 - b53) c31
float termA8 = -A[2] - A[4] + A[14];
float termB8 = -B[2] - B[22];
float result8 = termA8 * termB8;
C[2] += result8; // c31

// Expression: a31 (-b13 - b15 + b35) (-c13 - c53)
float termA9 = A[10];
float termB9 = -B[2] - B[4] + B[14];
float result9 = termA9 * termB9;
C[10] -= result9; // c13
C[14] -= result9; // c53

// Expression: (a21 - a22 - a24 - a25) (b14 + b15 - b25 + b45) (c52 + c54)
float termA10 = A[5] - A[6] - A[8] - A[9];
float termB10 = B[3] + B[4] - B[9] + B[19];
float result10 = termA10 * termB10;
C[9] += result10; // c52
C[19] += result10; // c54

// Expression: (a52 + a54) (b21 - b22 - b24 - b25) (c14 + c15 - c25 + c45)
float termA11 = A[21] + A[23];
float termB11 = B[5] - B[6] - B[8] - B[9];
float result11 = termA11 * termB11;
C[15] += result11; // c14
C[20] += result11; // c15
C[21] -= result11; // c25
C[23] += result11; // c45

// Expression: (a14 + a15 - a25 + a45) (b52 + b54) (c21 - c22 - c24 - c25)
float termA12 = A[3] + A[4] - A[9] + A[19];
float termB12 = B[21] + B[23];
float result12 = termA12 * termB12;
C[1] += result12; // c21
C[6] -= result12; // c22
C[16] -= result12; // c24
C[21] -= result12; // c25

// Expression: (a24 - a44) (b42 + b44) c22
float termA13 = A[8] - A[18];
float termB13 = B[16] + B[18];
float result13 = termA13 * termB13;
C[6] += result13; // c22

// Expression: a22 (b24 - b44) (c42 + c44)
float termA14 = A[6];
float termB14 = B[8] - B[18];
float result14 = termA14 * termB14;
C[8] += result14; // c42
C[18] += result14; // c44

// Expression: (a42 + a44) b22 (c24 - c44)
float termA15 = A[16] + A[18];
float termB15 = B[6];
float result15 = termA15 * termB15;
C[16] += result15; // c24
C[18] -= result15; // c44

// Expression: (a12 + a14 + a21 - a22 - a24 - a25 - a41 + a42 + a44 + a45) b45 (c13 + c14 + c15 - c33 - c34 - c35 + c53 + c54 + c55)
float termA16 = A[1] + A[3] + A[5] - A[6] - A[8] - A[9] - A[15] + A[16] + A[18] + A[19];
float termB16 = B[19];
float result16 = termA16 * termB16;
C[10] += result16; // c13
C[15] += result16; // c14
C[20] += result16; // c15
C[12] -= result16; // c33
C[17] -= result16; // c34
C[22] -= result16; // c35
C[14] += result16; // c53
C[19] += result16; // c54
C[24] += result16; // c55

// Expression: (a13 + a14 + a15 - a33 - a34 - a35 + a53 + a54 + a55) (b12 + b14 + b21 - b22 - b24 - b25 - b41 + b42 + b44 + b45) c45
float termA17 = A[2] + A[3] + A[4] - A[12] - A[13] - A[14] + A[22] + A[23] + A[24];
float termB17 = B[1] + B[3] + B[5] - B[6] - B[8] - B[9] - B[15] + B[16] + B[18] + B[19];
float result17 = termA17 * termB17;
C[23] += result17; // c45

// Expression: a45 (b13 + b14 + b15 - b33 - b34 - b35 + b53 + b54 + b55) (c12 + c14 + c21 - c22 - c24 - c25 - c41 + c42 + c44 + c45)
float termA18 = A[19];
float termB18 = B[2] + B[3] + B[4] - B[12] - B[13] - B[14] + B[22] + B[23] + B[24];
float result18 = termA18 * termB18;
C[5] += result18; // c12
C[15] += result18; // c14
C[1] += result18; // c21
C[6] -= result18; // c22
C[16] -= result18; // c24
C[21] -= result18; // c25
C[3] -= result18; // c41
C[8] += result18; // c42
C[18] += result18; // c44
C[23] += result18; // c45

// Expression: (-a41 + a44 + a45 + a51 - a54 - a55) (b11 + b12 - b13 - b22 + 2 b31 + 2 b33 + b42) (c15 + c22 + c24 + c55)
float termA19 = -A[15] + A[18] + A[19] + A[20] - A[23] - A[24];
float termB19 = B[0] + B[1] - B[2] - B[6] + 2*B[10] + 2*B[12] + B[16];
float result19 = termA19 * termB19;
C[20] += result19; // c15
C[6] += result19; // c22
C[16] += result19; // c24
C[24] += result19; // c55

// Expression: (a15 + a22 + a24 + a55) (-b41 + b44 + b45 + b51 - b54 - b55) (c11 + c12 - c13 - c22 + 2 c31 + 2 c33 + c42)
float termA20 = A[4] + A[6] + A[8] + A[24];
float termB20 = -B[15] + B[18] + B[19] + B[20] - B[23] - B[24];
float result20 = termA20 * termB20;
C[0] += result20; // c11
C[5] += result20; // c12
C[10] -= result20; // c13
C[6] -= result20; // c22
C[2] += 2 * result20; // c31
C[12] += 2 * result20; // c33
C[8] += result20; // c42

// Expression: (a11 + a12 - a13 - a22 + 2 a31 + 2 a33 + a42) (b15 + b22 + b24 + b55) (-c41 + c44 + c45 + c51 - c54 - c55)
float termA21 = A[0] + A[1] - A[2] - A[6] + 2*A[10] + 2*A[12] + A[16];
float termB21 = B[4] + B[6] + B[8] + B[24];
float result21 = termA21 * termB21;
C[3] -= result21; // c41
C[18] += result21; // c44
C[23] += result21; // c45
C[4] += result21; // c51
C[19] -= result21; // c54
C[24] -= result21; // c55

// Expression: (-a13 + a32 + a33 + a34 + a35 - a43) (b12 - b22 - b32 + b42) (-c23 - c25 + c43 + c45)
float termA22 = -A[2] + A[11] + A[12] + A[13] + A[14] - A[17];
float termB22 = B[1] - B[6] - B[11] + B[16];
float result22 = termA22 * termB22;
C[11] -= result22; // c23
C[21] -= result22; // c25
C[13] += result22; // c43
C[23] += result22; // c45

// Expression: (-a23 - a25 + a43 + a45) (-b13 + b32 + b33 + b34 + b35 - b43) (c12 - c22 - c32 + c42)
float termA23 = -A[7] - A[9] + A[17] + A[19];
float termB23 = -B[2] + B[11] + B[12] + B[13] + B[14] - B[17];
float result23 = termA23 * termB23;
C[5] += result23; // c12
C[6] -= result23; // c22
C[7] -= result23; // c32
C[8] += result23; // c42

// Expression: (a12 - a22 - a32 + a42) (-b23 - b25 + b43 + b45) (-c13 + c32 + c33 + c34 + c35 - c43)
float termA24 = A[1] - A[6] - A[11] + A[16];
float termB24 = -B[7] - B[9] + B[17] + B[19];
float result24 = termA24 * termB24;
C[10] -= result24; // c13
C[7] += result24; // c32
C[12] += result24; // c33
C[17] += result24; // c34
C[22] += result24; // c35
C[13] -= result24; // c43

// Expression: (-a13 + a23 - a43) (b32 + b34 + b35) (c12 - c22 - c23 - c24 - c25 - c32 + c42)
float termA25 = -A[2] + A[7] - A[17];
float termB25 = B[11] + B[13] + B[14];
float result25 = termA25 * termB25;
C[5] += result25; // c12
C[6] -= result25; // c22
C[11] -= result25; // c23
C[16] -= result25; // c24
C[21] -= result25; // c25
C[7] -= result25; // c32
C[8] += result25; // c42

// Expression: (a12 - a22 - a23 - a24 - a25 - a32 + a42) (-b13 + b23 - b43) (c32 + c34 + c35)
float termA26 = A[1] - A[6] - A[7] - A[8] - A[9] - A[11] + A[16];
float termB26 = -B[2] + B[7] - B[17];
float result26 = termA26 * termB26;
C[7] += result26; // c32
C[17] += result26; // c34
C[22] += result26; // c35

// Expression: (a32 + a34 + a35) (b12 - b22 - b23 - b24 - b25 - b32 + b42) (-c13 + c23 - c43)
float termA27 = A[11] + A[13] + A[14];
float termB27 = B[1] - B[6] - B[7] - B[8] - B[9] - B[11] + B[16];
float result27 = termA27 * termB27;
C[10] -= result27; // c13
C[11] += result27; // c23
C[13] -= result27; // c43

// Expression: (a41 - a44 - a45 - a51 + a54 + a55) (-b11 - b22 + b42) (c15 + c35 + c55)
float termA28 = A[15] - A[18] - A[19] - A[20] + A[23] + A[24];
float termB28 = -B[0] - B[6] + B[16];
float result28 = termA28 * termB28;
C[20] += result28; // c15
C[22] += result28; // c35
C[24] += result28; // c55

// Expression: (a15 + a35 + a55) (b41 - b44 - b45 - b51 + b54 + b55) (-c11 - c22 + c42)
float termA29 = A[4] + A[14] + A[24];
float termB29 = B[15] - B[18] - B[19] - B[20] + B[23] + B[24];
float result29 = termA29 * termB29;
C[0] -= result29; // c11
C[6] -= result29; // c22
C[8] += result29; // c42

// Expression: (-a11 - a22 + a42) (b15 + b35 + b55) (c41 - c44 - c45 - c51 + c54 + c55)
float termA30 = -A[0] - A[6] + A[16];
float termB30 = B[4] + B[14] + B[24];
float result30 = termA30 * termB30;
C[3] += result30; // c41
C[18] -= result30; // c44
C[23] -= result30; // c45
C[4] -= result30; // c51
C[19] += result30; // c54
C[24] += result30; // c55

// Expression: (-a11 + a15) b55 (c11 + c13 - c31 - c33 + c51 + c53)
float termA31 = -A[0] + A[4];
float termB31 = B[24];
float result31 = termA31 * termB31;
C[0] += result31; // c11
C[10] += result31; // c13
C[2] -= result31; // c31
C[12] -= result31; // c33
C[4] += result31; // c51
C[14] += result31; // c53

// Expression: (a11 + a13 - a31 - a33 + a51 + a53) (-b11 + b15) c55
float termA32 = A[0] + A[2] - A[10] - A[12] + A[20] + A[22];
float termB32 = -B[0] + B[4];
float result32 = termA32 * termB32;
C[24] += result32; // c55

// Expression: a55 (b11 + b13 - b31 - b33 + b51 + b53) (-c11 + c15)
float termA33 = A[24];
float termB33 = B[0] + B[2] - B[10] - B[12] + B[20] + B[22];
float result33 = termA33 * termB33;
C[0] -= result33; // c11
C[20] += result33; // c15

// Expression: (-a41 + a44 + a45 + a51 - a54 - a55) (-b11 - b12 + b13 - 2 b31 - 2 b33) (c15 + c22 + c24 + c35 + c55)
float termA34 = -A[15] + A[18] + A[19] + A[20] - A[23] - A[24];
float termB34 = -B[0] - B[1] + B[2] - 2*B[10] - 2*B[12];
float result34 = termA34 * termB34;
C[20] += result34; // c15
C[6] += result34; // c22
C[16] += result34; // c24
C[22] += result34; // c35
C[24] += result34; // c55

// Expression: (a15 + a22 + a24 + a35 + a55) (-b41 + b44 + b45 + b51 - b54 - b55) (-c11 - c12 + c13 - 2 c31 - 2 c33)
float termA35 = A[4] + A[6] + A[8] + A[14] + A[24];
float termB35 = -B[15] + B[18] + B[19] + B[20] - B[23] - B[24];
float result35 = termA35 * termB35;
C[0] -= result35; // c11
C[5] -= result35; // c12
C[10] += result35; // c13
C[2] -= 2 * result35; // c31
C[12] -= 2 * result35; // c33

// Expression: (-a11 - a12 + a13 - 2 a31 - 2 a33) (b15 + b22 + b24 + b35 + b55) (-c41 + c44 + c45 + c51 - c54 - c55)
float termA36 = -A[0] - A[1] + A[2] - 2*A[10] - 2*A[12];
float termB36 = B[4] + B[6] + B[8] + B[14] + B[24];
float result36 = termA36 * termB36;
C[3] -= result36; // c41
C[18] += result36; // c44
C[23] += result36; // c45
C[4] += result36; // c51
C[19] -= result36; // c54
C[24] -= result36; // c55

// Expression: (a12 + a14 - a22 - a32 - a34 + a42) (-b22 - b23 - b24 - b25 + b43 + b45) (c13 - c33 - c34 - c35 + c43)
float termA37 = A[1] + A[3] - A[6] - A[11] - A[13] + A[16];
float termB37 = -B[6] - B[7] - B[8] - B[9] + B[17] + B[19];
float result37 = termA37 * termB37;
C[10] += result37; // c13
C[12] -= result37; // c33
C[17] -= result37; // c34
C[22] -= result37; // c35
C[13] += result37; // c43

// Expression: (a13 - a33 - a34 - a35 + a43) (b12 + b14 - b22 - b32 - b34 + b42) (-c22 - c23 - c24 - c25 + c43 + c45)
float termA38 = A[2] - A[12] - A[13] - A[14] + A[17];
float termB38 = B[1] + B[3] - B[6] - B[11] - B[13] + B[16];
float result38 = termA38 * termB38;
C[6] -= result38; // c22
C[11] -= result38; // c23
C[16] -= result38; // c24
C[21] -= result38; // c25
C[13] += result38; // c43
C[23] += result38; // c45

// Expression: (-a22 - a23 - a24 - a25 + a43 + a45) (b13 - b33 - b34 - b35 + b43) (c12 + c14 - c22 - c32 - c34 + c42)
float termA39 = -A[6] - A[7] - A[8] - A[9] + A[17] + A[19];
float termB39 = B[2] - B[12] - B[13] - B[14] + B[17];
float result39 = termA39 * termB39;
C[5] += result39; // c12
C[15] += result39; // c14
C[6] -= result39; // c22
C[7] -= result39; // c32
C[17] -= result39; // c34
C[8] += result39; // c42

// Expression: -(a13 (b13 - b33 + b53) (-c11 + c14 + c15 + c31 - c34 - c35 - c41 + c44 + c45))
float termA40 = A[2];
float termB40 = B[2] - B[12] + B[22];
float result40 = -(termA40 * termB40);
C[0] -= result40; // c11
C[15] += result40; // c14
C[20] += result40; // c15
C[2] += result40; // c31
C[17] -= result40; // c34
C[22] -= result40; // c35
C[3] -= result40; // c41
C[18] += result40; // c44
C[23] += result40; // c45

// Expression: -((-a11 + a14 + a15 + a31 - a34 - a35 - a41 + a44 + a45) b13 (c13 - c33 + c53))
float termA41 = -A[0] + A[3] + A[4] + A[10] - A[13] - A[14] - A[15] + A[18] + A[19];
float termB41 = B[2];
float result41 = -(termA41 * termB41);
C[10] += result41; // c13
C[12] -= result41; // c33
C[14] += result41; // c53

// Expression: -((a13 - a33 + a53) (-b11 + b14 + b15 + b31 - b34 - b35 - b41 + b44 + b45) c13)
float termA42 = A[2] - A[12] + A[22];
float termB42 = -B[0] + B[3] + B[4] + B[10] - B[13] - B[14] - B[15] + B[18] + B[19];
float result42 = -(termA42 * termB42);
C[10] += result42; // c13

// Expression: (-2 a41 - a43 + a44 + a45 + 2 a51 + a53 - a54 - a55) (b12 - b22 + b31 + b33 + b42) c35
float termA43 = -2*A[15] - A[17] + A[18] + A[19] + 2*A[20] + A[22] - A[23] - A[24];
float termB43 = B[1] - B[6] + B[10] + B[12] + B[16];
float result43 = termA43 * termB43;
C[22] += result43; // c35

// Expression: a35 (-2 b41 - b43 + b44 + b45 + 2 b51 + b53 - b54 - b55) (c12 - c22 + c31 + c33 + c42)
float termA44 = A[14];
float termB44 = -2*B[15] - B[17] + B[18] + B[19] + 2*B[20] + B[22] - B[23] - B[24];
float result44 = termA44 * termB44;
C[5] += result44; // c12
C[6] -= result44; // c22
C[2] += result44; // c31
C[12] += result44; // c33
C[8] += result44; // c42

// Expression: (a12 - a22 + a31 + a33 + a42) b35 (-2 c41 - c43 + c44 + c45 + 2 c51 + c53 - c54 - c55)
float termA45 = A[1] - A[6] + A[10] + A[12] + A[16];
float termB45 = B[14];
float result45 = termA45 * termB45;
C[3] -= 2 * result45; // c41
C[13] -= result45; // c43
C[18] += result45; // c44
C[23] += result45; // c45
C[4] += 2 * result45; // c51
C[14] += result45; // c53
C[19] -= result45; // c54
C[24] -= result45; // c55

// Expression: (-a21 + a22 + a24 + a25 + a41 - a42 - a44 - a45 + a52 + a54) (b11 + b45) (c14 + c15 + c55)
float termA46 = -A[5] + A[6] + A[8] + A[9] + A[15] - A[16] - A[18] - A[19] + A[21] + A[23];
float termB46 = B[0] + B[19];
float result46 = termA46 * termB46;
C[15] += result46; // c14
C[20] += result46; // c15
C[24] += result46; // c55

// Expression: (a14 + a15 + a55) (-b21 + b22 + b24 + b25 + b41 - b42 - b44 - b45 + b52 + b54) (c11 + c45)
float termA47 = A[3] + A[4] + A[24];
float termB47 = -B[5] + B[6] + B[8] + B[9] + B[15] - B[16] - B[18] - B[19] + B[21] + B[23];
float result47 = termA47 * termB47;
C[0] += result47; // c11
C[23] += result47; // c45

// Expression: (a11 + a45) (b14 + b15 + b55) (-c21 + c22 + c24 + c25 + c41 - c42 - c44 - c45 + c52 + c54)
float termA48 = A[0] + A[19];
float termB48 = B[3] + B[4] + B[24];
float result48 = termA48 * termB48;
C[1] -= result48; // c21
C[6] += result48; // c22
C[16] += result48; // c24
C[21] += result48; // c25
C[3] += result48; // c41
C[8] -= result48; // c42
C[18] -= result48; // c44
C[23] -= result48; // c45
C[9] += result48; // c52
C[19] += result48; // c54

// Expression: (-a12 + a22 - a42) (-b25 + b35 + b45) (-c41 - c43 + c51 + c53)
float termA49 = -A[1] + A[6] - A[16];
float termB49 = -B[9] + B[14] + B[19];
float result49 = termA49 * termB49;
C[3] -= result49; // c41
C[13] -= result49; // c43
C[4] += result49; // c51
C[14] += result49; // c53

// Expression: (-a41 - a43 + a51 + a53) (-b12 + b22 - b42) (-c25 + c35 + c45)
float termA50 = -A[15] - A[17] + A[20] + A[22];
float termB50 = -B[1] + B[6] - B[16];
float result50 = termA50 * termB50;
C[21] -= result50; // c25
C[22] += result50; // c35
C[23] += result50; // c45

// Expression: (-a25 + a35 + a45) (-b41 - b43 + b51 + b53) (-c12 + c22 - c42)
float termA51 = -A[9] + A[14] + A[19];
float termB51 = -B[15] - B[17] + B[20] + B[22];
float result51 = termA51 * termB51;
C[5] -= result51; // c12
C[6] += result51; // c22
C[8] -= result51; // c42

// Expression: (-a11 - a13 + a31 + a33 - a41 - a43) b14 (c22 + c24 + c25 - c42 - c44 - c45 + c52 + c54)
float termA52 = -A[0] - A[2] + A[10] + A[12] - A[15] - A[17];
float termB52 = B[3];
float result52 = termA52 * termB52;
C[6] += result52; // c22
C[16] += result52; // c24
C[21] += result52; // c25
C[8] -= result52; // c42
C[18] -= result52; // c44
C[23] -= result52; // c45
C[9] += result52; // c52
C[19] += result52; // c54

// Expression: (a22 + a24 + a25 - a42 - a44 - a45 + a52 + a54) (-b11 - b13 + b31 + b33 - b41 - b43) c14
float termA53 = A[6] + A[8] + A[9] - A[16] - A[18] - A[19] + A[21] + A[23];
float termB53 = -B[0] - B[2] + B[10] + B[12] - B[15] - B[17];
float result53 = termA53 * termB53;
C[15] += result53; // c14

// Expression: a14 (b22 + b24 + b25 - b42 - b44 - b45 + b52 + b54) (-c11 - c13 + c31 + c33 - c41 - c43)
float termA54 = A[3];
float termB54 = B[6] + B[8] + B[9] - B[16] - B[18] - B[19] + B[21] + B[23];
float result54 = termA54 * termB54;
C[0] -= result54; // c11
C[10] -= result54; // c13
C[2] += result54; // c31
C[12] += result54; // c33
C[3] -= result54; // c41
C[13] -= result54; // c43

// Expression: (a22 + a24 + a25 - a45) (b13 + b14 + b15 - b33 - b34 - b35 + b41 + b43 - b51 + b54 + b55) (c12 + c14 - c22 + c42)
float termA55 = A[6] + A[8] + A[9] - A[19];
float termB55 = B[2] + B[3] + B[4] - B[12] - B[13] - B[14] + B[15] + B[17] - B[20] + B[23] + B[24];
float result55 = termA55 * termB55;
C[5] += result55; // c12
C[15] += result55; // c14
C[6] -= result55; // c22
C[8] += result55; // c42

// Expression: (a12 + a14 - a22 + a42) (b22 + b24 + b25 - b45) (c13 + c14 + c15 - c33 - c34 - c35 + c41 + c43 - c51 + c54 + c55)
float termA56 = A[1] + A[3] - A[6] + A[16];
float termB56 = B[6] + B[8] + B[9] - B[19];
float result56 = termA56 * termB56;
C[10] += result56; // c13
C[15] += result56; // c14
C[20] += result56; // c15
C[12] -= result56; // c33
C[17] -= result56; // c34
C[22] -= result56; // c35
C[3] += result56; // c41
C[13] += result56; // c43
C[4] -= result56; // c51
C[19] += result56; // c54
C[24] += result56; // c55

// Expression: (a13 + a14 + a15 - a33 - a34 - a35 + a41 + a43 - a51 + a54 + a55) (b12 + b14 - b22 + b42) (c22 + c24 + c25 - c45)
float termA57 = A[2] + A[3] + A[4] - A[12] - A[13] - A[14] + A[15] + A[17] - A[20] + A[23] + A[24];
float termB57 = B[1] + B[3] - B[6] + B[16];
float result57 = termA57 * termB57;
C[6] += result57; // c22
C[16] += result57; // c24
C[21] += result57; // c25
C[23] -= result57; // c45

// Expression: (a12 + a14 - a32 - a34) (b22 + b23 + b24 + b25) (c13 - c23 - c33 - c34 - c35 + c43)
float termA58 = A[1] + A[3] - A[11] - A[13];
float termB58 = B[6] + B[7] + B[8] + B[9];
float result58 = termA58 * termB58;
C[10] += result58; // c13
C[11] -= result58; // c23
C[12] -= result58; // c33
C[17] -= result58; // c34
C[22] -= result58; // c35
C[13] += result58; // c43

// Expression: (a13 - a23 - a33 - a34 - a35 + a43) (b12 + b14 - b32 - b34) (c22 + c23 + c24 + c25)
float termA59 = A[2] - A[7] - A[12] - A[13] - A[14] + A[17];
float termB59 = B[1] + B[3] - B[11] - B[13];
float result59 = termA59 * termB59;
C[6] += result59; // c22
C[11] += result59; // c23
C[16] += result59; // c24
C[21] += result59; // c25

// Expression: (a22 + a23 + a24 + a25) (b13 - b23 - b33 - b34 - b35 + b43) (c12 + c14 - c32 - c34)
float termA60 = A[6] + A[7] + A[8] + A[9];
float termB60 = B[2] - B[7] - B[12] - B[13] - B[14] + B[17];
float result60 = termA60 * termB60;
C[5] += result60; // c12
C[15] += result60; // c14
C[7] -= result60; // c32
C[17] -= result60; // c34

// Expression: (-a42 + a52) (b11 + b13 - b21 - b23 - b31 - b33 + b41 + b43) (c12 + c14)
float termA61 = -A[16] + A[21];
float termB61 = B[0] + B[2] - B[5] - B[7] - B[10] - B[12] + B[15] + B[17];
float result61 = termA61 * termB61;
C[5] += result61; // c12
C[15] += result61; // c14

// Expression: (a12 + a14) (-b42 + b52) (c11 + c13 - c21 - c23 - c31 - c33 + c41 + c43)
float termA62 = A[1] + A[3];
float termB62 = -B[16] + B[21];
float result62 = termA62 * termB62;
C[0] += result62; // c11
C[10] += result62; // c13
C[1] -= result62; // c21
C[11] -= result62; // c23
C[2] -= result62; // c31
C[12] -= result62; // c33
C[3] += result62; // c41
C[13] += result62; // c43

// Expression: (a11 + a13 - a21 - a23 - a31 - a33 + a41 + a43) (b12 + b14) (-c42 + c52)
float termA63 = A[0] + A[2] - A[5] - A[7] - A[10] - A[12] + A[15] + A[17];
float termB63 = B[1] + B[3];
float result63 = termA63 * termB63;
C[8] -= result63; // c42
C[9] += result63; // c52

// Expression: (-a25 + a45) (b12 + b13 + b14 + b15 - b32 - b33 - b34 - b35 + b52 + b53 + b54 + b55) (-c21 + c24 + c25)
float termA64 = -A[9] + A[19];
float termB64 = B[1] + B[2] + B[3] + B[4] - B[11] - B[12] - B[13] - B[14] + B[21] + B[22] + B[23] + B[24];
float result64 = termA64 * termB64;
C[1] -= result64; // c21
C[16] += result64; // c24
C[21] += result64; // c25

// Expression: (-a21 + a24 + a25) (-b25 + b45) (c12 + c13 + c14 + c15 - c32 - c33 - c34 - c35 + c52 + c53 + c54 + c55)
float termA65 = -A[5] + A[8] + A[9];
float termB65 = -B[9] + B[19];
float result65 = termA65 * termB65;
C[5] += result65; // c12
C[10] += result65; // c13
C[15] += result65; // c14
C[20] += result65; // c15
C[7] -= result65; // c32
C[12] -= result65; // c33
C[17] -= result65; // c34
C[22] -= result65; // c35
C[9] += result65; // c52
C[14] += result65; // c53
C[19] += result65; // c54
C[24] += result65; // c55

// Expression: (a12 + a13 + a14 + a15 - a32 - a33 - a34 - a35 + a52 + a53 + a54 + a55) (-b21 + b24 + b25) (-c25 + c45)
float termA66 = A[1] + A[2] + A[3] + A[4] - A[11] - A[12] - A[13] - A[14] + A[21] + A[22] + A[23] + A[24];
float termB66 = -B[5] + B[8] + B[9];
float result66 = termA66 * termB66;
C[21] -= result66; // c25
C[23] += result66; // c45

// Expression: (-a22 + a42 - a52 - a54) (b21 - b22 - b24 - b25 - b41 + b45) (c14 + c15 + c45)
float termA67 = -A[6] + A[16] - A[21] - A[23];
float termB67 = B[5] - B[6] - B[8] - B[9] - B[15] + B[19];
float result67 = termA67 * termB67;
C[15] += result67; // c14
C[20] += result67; // c15
C[23] += result67; // c45

// Expression: (a14 + a15 + a45) (-b22 + b42 - b52 - b54) (c21 - c22 - c24 - c25 - c41 + c45)
float termA68 = A[3] + A[4] + A[19];
float termB68 = -B[6] + B[16] - B[21] - B[23];
float result68 = termA68 * termB68;
C[1] += result68; // c21
C[6] -= result68; // c22
C[16] -= result68; // c24
C[21] -= result68; // c25
C[3] -= result68; // c41
C[23] += result68; // c45

// Expression: (a21 - a22 - a24 - a25 - a41 + a45) (b14 + b15 + b45) (-c22 + c42 - c52 - c54)
float termA69 = A[5] - A[6] - A[8] - A[9] - A[15] + A[19];
float termB69 = B[3] + B[4] + B[19];
float result69 = termA69 * termB69;
C[6] -= result69; // c22
C[8] += result69; // c42
C[9] -= result69; // c52
C[19] -= result69; // c54

// Expression: (a13 - a45) (b13 - b33 - b34 - b35 + b53) (c12 + c14 + c21 - c22 - c24 - c25 - c32 - c34 - c41 + c42 + c44 + c45)
float termA70 = A[2] - A[19];
float termB70 = B[2] - B[12] - B[13] - B[14] + B[22];
float result70 = termA70 * termB70;
C[5] += result70; // c12
C[15] += result70; // c14
C[1] += result70; // c21
C[6] -= result70; // c22
C[16] -= result70; // c24
C[21] -= result70; // c25
C[7] -= result70; // c32
C[17] -= result70; // c34
C[3] -= result70; // c41
C[8] += result70; // c42
C[18] += result70; // c44
C[23] += result70; // c45

// Expression: (a12 + a14 + a21 - a22 - a24 - a25 - a32 - a34 - a41 + a42 + a44 + a45) (b13 - b45) (c13 - c33 - c34 - c35 + c53)
float termA71 = A[1] + A[3] + A[5] - A[6] - A[8] - A[9] - A[11] - A[13] - A[15] + A[16] + A[18] + A[19];
float termB71 = B[2] - B[19];
float result71 = termA71 * termB71;
C[10] += result71; // c13
C[12] -= result71; // c33
C[17] -= result71; // c34
C[22] -= result71; // c35
C[14] += result71; // c53

// Expression: (a13 - a33 - a34 - a35 + a53) (b12 + b14 + b21 - b22 - b24 - b25 - b32 - b34 - b41 + b42 + b44 + b45) (c13 - c45)
float termA72 = A[2] - A[12] - A[13] - A[14] + A[22];
float termB72 = B[1] + B[3] + B[5] - B[6] - B[8] - B[9] - B[11] - B[13] - B[15] + B[16] + B[18] + B[19];
float result72 = termA72 * termB72;
C[10] += result72; // c13
C[23] -= result72; // c45

// Expression: (a21 - a22 - a24 - a25 + a42 - a52) (b11 - b25 + b45) (c12 + c14 + c15 + c55)
float termA73 = A[5] - A[6] - A[8] - A[9] + A[16] - A[21];
float termB73 = B[0] - B[9] + B[19];
float result73 = termA73 * termB73;
C[5] += result73; // c12
C[15] += result73; // c14
C[20] += result73; // c15
C[24] += result73; // c55

// Expression: (a12 + a14 + a15 + a55) (b21 - b22 - b24 - b25 + b42 - b52) (c11 - c25 + c45)
float termA74 = A[1] + A[3] + A[4] + A[24];
float termB74 = B[5] - B[6] - B[8] - B[9] + B[16] - B[21];
float result74 = termA74 * termB74;
C[0] += result74; // c11
C[21] -= result74; // c25
C[23] += result74; // c45

// Expression: (a11 - a25 + a45) (b12 + b14 + b15 + b55) (c21 - c22 - c24 - c25 + c42 - c52)
float termA75 = A[0] - A[9] + A[19];
float termB75 = B[1] + B[3] + B[4] + B[24];
float result75 = termA75 * termB75;
C[1] += result75; // c21
C[6] -= result75; // c22
C[16] -= result75; // c24
C[21] -= result75; // c25
C[8] += result75; // c42
C[9] -= result75; // c52

// Expression: (a13 + a43) (b34 + b35) (c12 + c14 - c22 - c23 - c24 - c25 - c32 - c34 + c42 + c43 + c44 + c45)
float termA76 = A[2] + A[17];
float termB76 = B[13] + B[14];
float result76 = termA76 * termB76;
C[5] += result76; // c12
C[15] += result76; // c14
C[6] -= result76; // c22
C[11] -= result76; // c23
C[16] -= result76; // c24
C[21] -= result76; // c25
C[7] -= result76; // c32
C[17] -= result76; // c34
C[8] += result76; // c42
C[13] += result76; // c43
C[18] += result76; // c44
C[23] += result76; // c45

// Expression: (a12 + a14 - a22 - a23 - a24 - a25 - a32 - a34 + a42 + a43 + a44 + a45) (b13 + b43) (c34 + c35)
float termA77 = A[1] + A[3] - A[6] - A[7] - A[8] - A[9] - A[11] - A[13] + A[16] + A[17] + A[18] + A[19];
float termB77 = B[2] + B[17];
float result77 = termA77 * termB77;
C[17] += result77; // c34
C[22] += result77; // c35

// Expression: (a34 + a35) (b12 + b14 - b22 - b23 - b24 - b25 - b32 - b34 + b42 + b43 + b44 + b45) (c13 + c43)
float termA78 = A[13] + A[14];
float termB78 = B[1] + B[3] - B[6] - B[7] - B[8] - B[9] - B[11] - B[13] + B[16] + B[17] + B[18] + B[19];
float result78 = termA78 * termB78;
C[10] += result78; // c13
C[13] += result78; // c43

// Expression: (a21 - a25 - a41 + a45) (b12 + b14 + b15 + b45) (c22 - c42 + c52)
float termA79 = A[5] - A[9] - A[15] + A[19];
float termB79 = B[1] + B[3] + B[4] + B[19];
float result79 = termA79 * termB79;
C[6] += result79; // c22
C[8] -= result79; // c42
C[9] += result79; // c52

// Expression: (a22 - a42 + a52) (b21 - b25 - b41 + b45) (c12 + c14 + c15 + c45)
float termA80 = A[6] - A[16] + A[21];
float termB80 = B[5] - B[9] - B[15] + B[19];
float result80 = termA80 * termB80;
C[5] += result80; // c12
C[15] += result80; // c14
C[20] += result80; // c15
C[23] += result80; // c45

// Expression: (a12 + a14 + a15 + a45) (b22 - b42 + b52) (c21 - c25 - c41 + c45)
float termA81 = A[1] + A[3] + A[4] + A[19];
float termB81 = B[6] - B[16] + B[21];
float result81 = termA81 * termB81;
C[1] += result81; // c21
C[21] -= result81; // c25
C[3] -= result81; // c41
C[23] += result81; // c45

// Expression: (-a12 - a14 - a35) (b22 + b23 + b24 + b25 - b42 + b52) (c13 - c23 - c31 - c33 + c43)
float termA82 = -A[1] - A[3] - A[14];
float termB82 = B[6] + B[7] + B[8] + B[9] - B[16] + B[21];
float result82 = termA82 * termB82;
C[10] += result82; // c13
C[11] -= result82; // c23
C[2] -= result82; // c31
C[12] -= result82; // c33
C[13] += result82; // c43

// Expression: (a13 - a23 - a31 - a33 + a43) (-b12 - b14 - b35) (c22 + c23 + c24 + c25 - c42 + c52)
float termA83 = A[2] - A[7] - A[10] - A[12] + A[17];
float termB83 = -B[1] - B[3] - B[14];
float result83 = termA83 * termB83;
C[6] += result83; // c22
C[11] += result83; // c23
C[16] += result83; // c24
C[21] += result83; // c25
C[8] -= result83; // c42
C[9] += result83; // c52

// Expression: (a22 + a23 + a24 + a25 - a42 + a52) (b13 - b23 - b31 - b33 + b43) (-c12 - c14 - c35)
float termA84 = A[6] + A[7] + A[8] + A[9] - A[16] + A[21];
float termB84 = B[2] - B[7] - B[10] - B[12] + B[17];
float result84 = termA84 * termB84;
C[5] -= result84; // c12
C[15] -= result84; // c14
C[22] -= result84; // c35

// Expression: (-a13 + a55) (b13 - b31 - b33 + b53) (c11 - c15 + c35)
float termA85 = -A[2] + A[24];
float termB85 = B[2] - B[10] - B[12] + B[22];
float result85 = termA85 * termB85;
C[0] += result85; // c11
C[20] -= result85; // c15
C[22] += result85; // c35

// Expression: (a11 - a15 + a35) (-b13 + b55) (c13 - c31 - c33 + c53)
float termA86 = A[0] - A[4] + A[14];
float termB86 = -B[2] + B[24];
float result86 = termA86 * termB86;
C[10] += result86; // c13
C[2] -= result86; // c31
C[12] -= result86; // c33
C[14] += result86; // c53

// Expression: (a13 - a31 - a33 + a53) (b11 - b15 + b35) (-c13 + c55)
float termA87 = A[2] - A[10] - A[12] + A[22];
float termB87 = B[0] - B[4] + B[14];
float result87 = termA87 * termB87;
C[10] -= result87; // c13
C[24] += result87; // c55

// Expression: (a13 - a31 - a33 + a43) (b14 + b35) (c22 + c23 + c24 + c25 - c42 - c43 - c44 - c45 + c52 + c54)
float termA88 = A[2] - A[10] - A[12] + A[17];
float termB88 = B[3] + B[14];
float result88 = termA88 * termB88;
C[6] += result88; // c22
C[11] += result88; // c23
C[16] += result88; // c24
C[21] += result88; // c25
C[8] -= result88; // c42
C[13] -= result88; // c43
C[18] -= result88; // c44
C[23] -= result88; // c45
C[9] += result88; // c52
C[19] += result88; // c54

// Expression: (a22 + a23 + a24 + a25 - a42 - a43 - a44 - a45 + a52 + a54) (b13 - b31 - b33 + b43) (c14 + c35)
float termA89 = A[6] + A[7] + A[8] + A[9] - A[16] - A[17] - A[18] - A[19] + A[21] + A[23];
float termB89 = B[2] - B[10] - B[12] + B[17];
float result89 = termA89 * termB89;
C[15] += result89; // c14
C[22] += result89; // c35

// Expression: (a14 + a35) (b22 + b23 + b24 + b25 - b42 - b43 - b44 - b45 + b52 + b54) (c13 - c31 - c33 + c43)
float termA90 = A[3] + A[14];
float termB90 = B[6] + B[7] + B[8] + B[9] - B[16] - B[17] - B[18] - B[19] + B[21] + B[23];
float result90 = termA90 * termB90;
C[10] += result90; // c13
C[2] -= result90; // c31
C[12] -= result90; // c33
C[13] += result90; // c43

// Expression: (a11 + a55) (b11 + b55) (c11 + c55)
float termA91 = A[0] + A[24];
float termB91 = B[0] + B[24];
float result91 = termA91 * termB91;
C[0] += result91; // c11
C[24] += result91; // c55

// Expression: (a22 + a44) (b22 + b44) (c22 + c44)
float termA92 = A[6] + A[18];
float termB92 = B[6] + B[18];
float result92 = termA92 * termB92;
C[6] += result92; // c22
C[18] += result92; // c44

// Expression: a33 b33 c33
float termA93 = A[12];
float termB93 = B[12];
float result93 = termA93 * termB93;
C[12] += result93; // c33



}
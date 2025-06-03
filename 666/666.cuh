#pragma once
// A, B, and C are all length‐36 arrays in row‐major order (row*6 + col).
// This function must be inlined for best performance.

__device__ __forceinline__ void matmul6x6_opt(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C) {
    // Expression: (a12+a13-a22-a23-a32-a33+a42+a43-a52-a53)(-b36+b46)(c45+c46-c55-c56+c65+c66)
    float termA1 = A[1] + A[2] - A[7] - A[8] - A[13] - A[14] + A[19] + A[20] - A[25] - A[26];
    float termB1 = -B[17] + B[23];
    float result1 = termA1 * termB1;
    C[27] += result1; // c45
    C[33] += result1; // c46
    C[28] -= result1; // c55
    C[34] -= result1; // c56
    C[29] += result1; // c65
    C[35] += result1; // c66

    // Expression: (-a45-a46+a55+a56-a65-a66)(b12+b13+2*b14+2*b15-b22-b23-2*b24-2*b25+b32+b33+2*b34+2*b35-b42-b43+b52+b53)(-c36+c46)
    float termA2 = -A[22] - A[23] + A[28] + A[29] - A[34] - A[35];
    float termB2 = B[1] + B[2] - B[7] - B[8] + B[13] + B[14] - B[19] - B[20] + B[25] + B[26];
    float result2 = termA2 * termB2;
    C[32] -= result2; // c36
    C[33] += result2; // c46

    // Expression: (a36-a46)(b45+b46-b55-b56+b65+b66)(c12-c13+c22-c23-c32+c33+c42-c43-c52+c53)
    float termA3 = A[17] - A[23];
    float termB3 = B[22] + B[23] - B[28] - B[29] + B[34] + B[35];
    float result3 = termA3 * termB3;
    C[6] += result3; // c12
    C[12] -= result3; // c13
    C[7] += result3; // c22
    C[13] -= result3; // c23
    C[8] -= result3; // c32
    C[14] += result3; // c33
    C[9] += result3; // c42
    C[15] -= result3; // c43
    C[10] -= result3; // c52
    C[16] += result3; // c53

    // Expression: (a24+a25+a34+a35-a44-a45+a54+a55-a64-a65)(-b31+b41)(c11+c12-c21-c22+c31+c32)
    float termA4 = A[9] + A[10] + A[15] + A[16] - A[21] - A[22] + A[27] + A[28] - A[33] - A[34];
    float termB4 = -B[12] + B[18];
    float result4 = termA4 * termB4;
    C[0] += result4; // c11
    C[6] += result4; // c12
    C[1] -= result4; // c21
    C[7] -= result4; // c22
    C[2] += result4; // c31
    C[8] += result4; // c32

    // Expression: (-a11-a12+a21+a22+a31+a32)(-b24-b25+b34+b35-b44-b45+b54+b55+b64+b65)(-c31+c41)
    float termA5 = -A[0] - A[1] + A[6] + A[7] + A[12] + A[13];
    float termB5 = -B[9] - B[10] + B[15] + B[16] - B[21] - B[22] + B[27] + B[28] + B[33] + B[34];
    float result5 = termA5 * termB5;
    C[2] -= result5; // c31
    C[3] += result5; // c41

    // Expression: (a31-a41)(b11+b12-b21-b22+b31+b32)(-c24-c25+c34+c35-c44-c45+c54+c55+c64+c65)
    float termA6 = A[12] - A[18];
    float termB6 = B[0] + B[1] - B[6] - B[7] + B[12] + B[13];
    float result6 = termA6 * termB6;
    C[19] -= result6; // c24
    C[25] -= result6; // c25
    C[20] += result6; // c34
    C[26] += result6; // c35
    C[21] -= result6; // c44
    C[27] -= result6; // c45
    C[22] += result6; // c54
    C[28] += result6; // c55
    C[23] += result6; // c64
    C[29] += result6; // c65

    // Expression: (-a64-a65)(b53+b54)(-c12+c13+c14-c16+c22-c23-c24+c26-c32+c33+c34-c36-c42+c43+c52-c53-c62+c63)
    float termA7 = -A[33] - A[34];
    float termB7 = B[26] + B[27];
    float result7 = termA7 * termB7;
    C[6] -= result7; // c12
    C[12] += result7; // c13
    C[18] += result7; // c14
    C[30] -= result7; // c16
    C[7] += result7; // c22
    C[13] -= result7; // c23
    C[19] -= result7; // c24
    C[31] += result7; // c26
    C[8] -= result7; // c32
    C[14] += result7; // c33
    C[20] += result7; // c34
    C[32] -= result7; // c36
    C[9] -= result7; // c42
    C[15] += result7; // c43
    C[10] += result7; // c52
    C[16] -= result7; // c53
    C[11] -= result7; // c62
    C[17] += result7; // c63

    // Expression: (-2*a11-a12+a13+a14-a16+2*a21+a22-a23-a24+a26+2*a31+a32-a33-a34+a36-a42-a43+a52+a53-a62-a63)(b64+b65)(c53+c54)
    float termA8 = -A[1] + A[2] + A[3] - A[5] + A[7] - A[8] - A[9] + A[11] + A[13] - A[14] - A[15] + A[17] - A[19] - A[20] + A[25] + A[26] - A[31] - A[32];
    float termB8 = B[33] + B[34];
    float result8 = termA8 * termB8;
    C[16] += result8; // c53
    C[22] += result8; // c54

    // Expression: (-a53-a54)(-2*b11-b12+b13+b14-b16+2*b21+b22-b23-b24+b26-2*b31-b32+b33+b34-b36+b42+b43-b52-b53+b62+b63)(c64+c65)
    float termA9 = -A[26] - A[27];
    float termB9 = -B[1] + B[2] + B[3] - B[5] + B[7] - B[8] - B[9] + B[11] - B[13] + B[14] + B[15] - B[17] + B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result9 = termA9 * termB9;
    C[23] += result9; // c64
    C[29] += result9; // c65

    // Expression: (a12+a13)(b23+b24)(-c14-c15+c24+c25-c34-c35+c41+c43+c44+c45-c51-c53-c54-c55+c61+c63+c64+c65)
    float termA10 = A[1] + A[2];
    float termB10 = B[8] + B[9];
    float result10 = termA10 * termB10;
    C[18] -= result10; // c14
    C[24] -= result10; // c15
    C[19] += result10; // c24
    C[25] += result10; // c25
    C[20] -= result10; // c34
    C[26] -= result10; // c35
    C[3] += result10; // c41
    C[15] += result10; // c43
    C[21] += result10; // c44
    C[27] += result10; // c45
    C[4] -= result10; // c51
    C[16] -= result10; // c53
    C[22] -= result10; // c54
    C[28] -= result10; // c55
    C[5] += result10; // c61
    C[17] += result10; // c63
    C[23] += result10; // c64
    C[29] += result10; // c65

    // Expression: (a14+a15-a24-a25-a34-a35+a41-a43-a44-a45-a51+a53+a54+a55+a61-a63-a64-a65)(b12+b13)(c23+c24)
    float termA11 = A[3] + A[4] - A[9] - A[10] - A[15] - A[16] + A[18] - A[20] - A[21] - A[22] - A[24] + A[26] + A[27] + A[28] + A[30] - A[32] - A[33] - A[34];
    float termB11 = B[1] + B[2];
    float result11 = termA11 * termB11;
    C[13] += result11; // c23
    C[19] += result11; // c24

    // Expression: (-a23-a24)(b14+b15-b24-b25+b34+b35+b41-b43-b44+b45+2*b46-b51+b53+b54-b55-2*b56+b61-b63-b64+b65+2*b66)(-c12+c13)
    float termA12 = -A[8] - A[9];
    float termB12 = B[3] + B[4] - B[9] - B[10] + B[15] + B[16] + B[18] - B[20] - B[21] + B[22] - B[24] + B[26] + B[27] - B[28] + B[30] - B[32] - B[33] + B[34];
    float result12 = termA12 * termB12;
    C[6] -= result12; // c12
    C[12] += result12; // c13

    // Expression: (a15+a16-a25-a26-a35-a36+a45+a46-a55-a56+a64+a65)(-b32+b42+b53+b54)(-c21-c23-c24+c26+c31)
    float termA13 = A[4] + A[5] - A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29] + A[33] + A[34];
    float termB13 = -B[13] + B[19] + B[26] + B[27];
    float result13 = termA13 * termB13;
    C[1] -= result13; // c21
    C[13] -= result13; // c23
    C[19] -= result13; // c24
    C[31] += result13; // c26
    C[2] += result13; // c31

    // Expression: (-a21+a23+a24-a26-a31)(-b15-b16+b25+b26-b35-b36+b45+b46-b55-b56+b64+b65)(c32-c42+c53+c54)
    float termA14 = -A[6] + A[8] + A[9] - A[11] - A[12];
    float termB14 = -B[4] - B[5] + B[10] + B[11] - B[16] - B[17] + B[22] + B[23] - B[28] - B[29] + B[33] + B[34];
    float result14 = termA14 * termB14;
    C[8] += result14; // c32
    C[9] -= result14; // c42
    C[16] += result14; // c53
    C[22] += result14; // c54

    // Expression: (a32-a42-a53-a54)(-b21+b23+b24-b26+b31)(-c15-c16+c25+c26-c35-c36+c45+c46-c55-c56+c64+c65)
    float termA15 = A[13] - A[19] - A[26] - A[27];
    float termB15 = -B[6] + B[8] + B[9] - B[11] + B[12];
    float result15 = termA15 * termB15;
    C[24] -= result15; // c15
    C[30] -= result15; // c16
    C[25] += result15; // c25
    C[31] += result15; // c26
    C[26] -= result15; // c35
    C[32] -= result15; // c36
    C[27] += result15; // c45
    C[33] += result15; // c46
    C[28] -= result15; // c55
    C[34] -= result15; // c56
    C[23] += result15; // c64
    C[29] += result15; // c65

    // Expression: (a12+a13-a21-a22-a31-a32+a41+a42-a51-a52+a61+a62)(-b23-b24-b35+b45)(-c46-c51-c53-c54+c56)
    float termA16 = A[1] + A[2] - A[6] - A[7] - A[12] - A[13] + A[18] + A[19] - A[24] - A[25] + A[30] + A[31];
    float termB16 = -B[8] - B[9] - B[16] + B[22];
    float result16 = termA16 * termB16;
    C[33] -= result16; // c46
    C[4] -= result16; // c51
    C[16] -= result16; // c53
    C[22] -= result16; // c54
    C[34] += result16; // c56

    // Expression: (-a46+a51-a53-a54+a56)(-b12-b13+b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(-c23-c24-c35+c45)
    float termA17 = -A[23] + A[24] - A[26] - A[27] + A[29];
    float termB17 = -B[1] - B[2] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result17 = termA17 * termB17;
    C[13] -= result17; // c23
    C[19] -= result17; // c24
    C[26] -= result17; // c35
    C[27] += result17; // c45

    // Expression: (-a23-a24+a35-a45)(-b46+b51-b53-b54+b56)(-c12+c13+c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA18 = -A[8] - A[9] + A[16] - A[22];
    float termB18 = -B[23] + B[24] - B[26] - B[27] + B[29];
    float result18 = termA18 * termB18;
    C[6] -= result18; // c12
    C[12] += result18; // c13
    C[1] += result18; // c21
    C[7] += result18; // c22
    C[2] -= result18; // c31
    C[8] -= result18; // c32
    C[3] += result18; // c41
    C[9] += result18; // c42
    C[4] -= result18; // c51
    C[10] -= result18; // c52
    C[5] += result18; // c61
    C[11] += result18; // c62

    // Expression: (-a33-a34-a35+a45)(-b46-b54+b56)(-c33+c43-c53-c54+c63+c64)
    float termA19 = -A[14] - A[15] - A[16] + A[22];
    float termB19 = -B[23] - B[27] + B[29];
    float result19 = termA19 * termB19;
    C[14] -= result19; // c33
    C[15] += result19; // c43
    C[16] -= result19; // c53
    C[22] -= result19; // c54
    C[17] += result19; // c63
    C[23] += result19; // c64

    // Expression: (-a33+a43-a53-a54+a63+a64)(-b33-b34-b35+b45)(-c46-c54+c56)
    float termA20 = -A[14] + A[20] - A[26] - A[27] + A[32] + A[33];
    float termB20 = -B[14] - B[15] - B[16] + B[22];
    float result20 = termA20 * termB20;
    C[33] -= result20; // c46
    C[22] -= result20; // c54
    C[34] += result20; // c56

    // Expression: (-a46-a54+a56)(-b33+b43-b53-b54+b63+b64)(-c33-c34-c35+c45)
    float termA21 = -A[23] - A[27] + A[29];
    float termB21 = -B[14] + B[20] - B[26] - B[27] + B[32] + B[33];
    float result21 = termA21 * termB21;
    C[14] -= result21; // c33
    C[20] -= result21; // c34
    C[26] -= result21; // c35
    C[27] += result21; // c45

    // Expression: (a32-a42-a43-a44)(-b21+b23+b31)(-c13-c14+c23+c24-c34+c44)
    float termA22 = A[13] - A[19] - A[20] - A[21];
    float termB22 = -B[6] + B[8] + B[12];
    float result22 = termA22 * termB22;
    C[12] -= result22; // c13
    C[18] -= result22; // c14
    C[13] += result22; // c23
    C[19] += result22; // c24
    C[20] -= result22; // c34
    C[21] += result22; // c44

    // Expression: (a13+a14-a23-a24-a34+a44)(-b32+b42+b43+b44)(-c21-c23+c31)
    float termA23 = A[2] + A[3] - A[8] - A[9] - A[15] + A[21];
    float termB23 = -B[13] + B[19] + B[20] + B[21];
    float result23 = termA23 * termB23;
    C[1] -= result23; // c21
    C[13] -= result23; // c23
    C[2] += result23; // c31

    // Expression: (-a21+a23-a31)(-b13-b14+b23+b24-b34+b44)(c32-c42+c43+c44)
    float termA24 = -A[6] + A[8] - A[12];
    float termB24 = -B[2] - B[3] + B[8] + B[9] - B[15] + B[21];
    float result24 = termA24 * termB24;
    C[8] += result24; // c32
    C[9] -= result24; // c42
    C[15] += result24; // c43
    C[21] += result24; // c44

    // Expression: (-a11+a23)(-b13-b14+b23+b24-b32-b33-b34+b42+b43+b44)(-c21-c22+c31)
    float termA25 = -A[0] + A[8];
    float termB25 = -B[2] - B[3] + B[8] + B[9] - B[13] - B[14] - B[15] + B[19] + B[20] + B[21];
    float result25 = termA25 * termB25;
    C[1] -= result25; // c21
    C[7] -= result25; // c22
    C[2] += result25; // c31

    // Expression: (a21+a22+a31)(-b11+b23)(-c13-c14+c23+c24+c32-c33-c34-c42+c43+c44)
    float termA26 = A[6] + A[7] + A[12];
    float termB26 = -B[0] + B[8];
    float result26 = termA26 * termB26;
    C[12] -= result26; // c13
    C[18] -= result26; // c14
    C[13] += result26; // c23
    C[19] += result26; // c24
    C[8] += result26; // c32
    C[14] -= result26; // c33
    C[20] -= result26; // c34
    C[9] -= result26; // c42
    C[15] += result26; // c43
    C[21] += result26; // c44

    // Expression: (a13+a14-a23-a24-a32-a33-a34+a42+a43+a44)(-b21-b22+b31)(c11+c23)
    float termA27 = A[2] + A[3] - A[8] - A[9] - A[13] - A[14] - A[15] + A[19] + A[20] + A[21];
    float termB27 = -B[6] - B[7] + B[12];
    float result27 = termA27 * termB27;
    C[0] += result27; // c11
    C[13] += result27; // c23

    // Expression: (a54-a66)(-b33-b34-b35+b43+b44+b45-b53-b54+b63+b64)(-c46+c55+c56)
    float termA28 = A[27] - A[35];
    float termB28 = -B[14] - B[15] - B[16] + B[20] + B[21] + B[22] - B[26] - B[27] + B[32] + B[33];
    float result28 = termA28 * termB28;
    C[33] -= result28; // c46
    C[28] += result28; // c55
    C[34] += result28; // c56

    // Expression: (a46-a55-a56)(-b54+b66)(-c33-c34-c35+c43+c44+c45-c53-c54+c63+c64)
    float termA29 = A[23] - A[28] - A[29];
    float termB29 = -B[27] + B[35];
    float result29 = termA29 * termB29;
    C[14] -= result29; // c33
    C[20] -= result29; // c34
    C[26] -= result29; // c35
    C[15] += result29; // c43
    C[21] += result29; // c44
    C[27] += result29; // c45
    C[16] -= result29; // c53
    C[22] -= result29; // c54
    C[17] += result29; // c63
    C[23] += result29; // c64

    // Expression: (a33+a34+a35-a43-a44-a45+a53+a54-a63-a64)(-b46+b55+b56)(-c54+c66)
    float termA30 = A[14] + A[15] + A[16] - A[20] - A[21] - A[22] + A[26] + A[27] - A[32] - A[33];
    float termB30 = -B[23] + B[28] + B[29];
    float result30 = termA30 * termB30;
    C[22] -= result30; // c54
    C[35] += result30; // c66

    // Expression: (-a21-a22)(b11+b43+b44)(-c12+c13+c22-c23-c32+c33)
    float termA31 = -A[6] - A[7];
    float termB31 = B[0] + B[20] + B[21];
    float result31 = termA31 * termB31;
    C[6] -= result31; // c12
    C[12] += result31; // c13
    C[7] += result31; // c22
    C[13] -= result31; // c23
    C[8] -= result31; // c32
    C[14] += result31; // c33

    // Expression: (-a12-a13+a22+a23+a32+a33)(b21+b22)(-c11+c43+c44)
    float termA32 = -A[1] - A[2] + A[7] + A[8] + A[13] + A[14];
    float termB32 = B[6] + B[7];
    float result32 = termA32 * termB32;
    C[0] -= result32; // c11
    C[15] += result32; // c43
    C[21] += result32; // c44

    // Expression: (-a11-a43-a44)(b12+b13-b22-b23+b32+b33)(c21+c22-2*c34+2*c44)
    float termA33 = -A[0] - A[20] - A[21];
    float termB33 = B[1] + B[2] - B[7] - B[8] + B[13] + B[14];
    float result33 = termA33 * termB33;
    C[1] += result33; // c21
    C[7] += result33; // c22

    // Expression: (a55+a56)(-b33-b34+b66)(c44+c45-c54-c55+c64+c65)
    float termA34 = A[28] + A[29];
    float termB34 = -B[14] - B[15] + B[35];
    float result34 = termA34 * termB34;
    C[21] += result34; // c44
    C[27] += result34; // c45
    C[22] -= result34; // c54
    C[28] -= result34; // c55
    C[23] += result34; // c64
    C[29] += result34; // c65

    // Expression: (a44+a45-a54-a55+a64+a65)(b55+b56)(-c33-c34+c66)
    float termA35 = A[21] + A[22] - A[27] - A[28] + A[33] + A[34];
    float termB35 = B[28] + B[29];
    float result35 = termA35 * termB35;
    C[14] -= result35; // c33
    C[20] -= result35; // c34
    C[35] += result35; // c66

    // Expression: (-a33-a34+a66)(b44+b45-b54-b55+b64+b65)(c55+c56)
    float termA36 = -A[14] - A[15] + A[35];
    float termB36 = B[21] + B[22] - B[27] - B[28] + B[33] + B[34];
    float result36 = termA36 * termB36;
    C[28] += result36; // c55
    C[34] += result36; // c56

    // Expression: (a56-a66)(b65+b66)c55
    float termA37 = A[29] - A[35];
    float termB37 = B[34] + B[35];
    float result37 = termA37 * termB37;
    C[28] += result37; // c55

    // Expression: -(a55(-b56+b66)(c65+c66))
    float termA38 = A[28];
    float termB38 = -B[29] + B[35];
    float result38 = -(termA38 * termB38);
    C[29] += result38; // c65
    C[35] += result38; // c66

    // Expression: (-a65-a66)b55(-c56+c66)
    float termA39 = -A[34] - A[35];
    float termB39 = B[28];
    float result39 = termA39 * termB39;
    C[34] -= result39; // c56
    C[35] += result39; // c66

    // Expression: (-a11+a21)(b11+b12)c22
    float termA40 = -A[0] + A[6];
    float termB40 = B[0] + B[1];
    float result40 = termA40 * termB40;
    C[7] += result40; // c22

    // Expression: a22(-b11+b21)(c11+c12)
    float termA41 = A[7];
    float termB41 = -B[0] + B[6];
    float result41 = termA41 * termB41;
    C[0] += result41; // c11
    C[6] += result41; // c12

    // Expression: (a11+a12)b22(-c11+c21)
    float termA42 = A[0] + A[1];
    float termB42 = B[7];
    float result42 = termA42 * termB42;
    C[0] -= result42; // c11
    C[1] += result42; // c21

    // Expression: (-a12-a13-a14-a15+a21+a22+a31+a32-a41-a42+a51+a52-a61-a62)(-b26+b36-b46+b55+b56)(-c51-c53-c54+c66)
    float termA43 = -A[1] - A[2] - A[3] - A[4] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25] - A[30] - A[31];
    float termB43 = -B[11] + B[17] - B[23] + B[28] + B[29];
    float result43 = termA43 * termB43;
    C[4] -= result43; // c51
    C[16] -= result43; // c53
    C[22] -= result43; // c54
    C[35] += result43; // c66

    // Expression: (-a51+a53+a54-a66)(-b12-b13-b14-b15+b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(-c26+c36-c46+c55+c56)
    float termA44 = -A[24] + A[26] + A[27] - A[35];
    float termB44 = -B[1] - B[2] - B[3] - B[4] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result44 = termA44 * termB44;
    C[31] -= result44; // c26
    C[32] += result44; // c36
    C[33] -= result44; // c46
    C[28] += result44; // c55
    C[34] += result44; // c56

    // Expression: (a26+a36-a46+a55+a56)(b51-b53-b54+b66)(-c12+c13+c14+c15+c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA45 = A[11] + A[17] - A[23] + A[28] + A[29];
    float termB45 = B[24] - B[26] - B[27] + B[35];
    float result45 = termA45 * termB45;
    C[6] -= result45; // c12
    C[12] += result45; // c13
    C[18] += result45; // c14
    C[24] += result45; // c15
    C[1] += result45; // c21
    C[7] += result45; // c22
    C[2] -= result45; // c31
    C[8] -= result45; // c32
    C[3] += result45; // c41
    C[9] += result45; // c42
    C[4] -= result45; // c51
    C[10] -= result45; // c52
    C[5] += result45; // c61
    C[11] += result45; // c62

    // Expression: (-a15-a16+a25+a26+a35+a36-a45-a46+a55+a56-a62-a63-a64-a65)(-b21-b22+b31-b41+b51)(-c11-c23-c24+c26)
    float termA46 = -A[4] - A[5] + A[10] + A[11] + A[16] + A[17] - A[22] - A[23] + A[28] + A[29] - A[31] - A[32] - A[33] - A[34];
    float termB46 = -B[6] - B[7] + B[12] - B[18] + B[24];
    float result46 = termA46 * termB46;
    C[0] -= result46; // c11
    C[13] -= result46; // c23
    C[19] -= result46; // c24
    C[31] += result46; // c26

    // Expression: (-a11+a23+a24-a26)(-b15-b16+b25+b26-b35-b36+b45+b46-b55-b56+b62+b63+b64+b65)(-c21-c22+c31-c41+c51)
    float termA47 = -A[0] + A[8] + A[9] - A[11];
    float termB47 = -B[4] - B[5] + B[10] + B[11] - B[16] - B[17] + B[22] + B[23] - B[28] - B[29] + B[31] + B[32] + B[33] + B[34];
    float result47 = termA47 * termB47;
    C[1] -= result47; // c21
    C[7] -= result47; // c22
    C[2] += result47; // c31
    C[3] -= result47; // c41
    C[4] += result47; // c51

    // Expression: (-a21-a22-a31+a41-a51)(b11-b23-b24+b26)(-c15-c16+c25+c26-c35-c36+c45+c46-c55-c56-c62+c63+c64+c65)
    float termA48 = -A[6] - A[7] - A[12] + A[18] - A[24];
    float termB48 = B[0] - B[8] - B[9] + B[11];
    float result48 = termA48 * termB48;
    C[24] -= result48; // c15
    C[30] -= result48; // c16
    C[25] += result48; // c25
    C[31] += result48; // c26
    C[26] -= result48; // c35
    C[32] -= result48; // c36
    C[27] += result48; // c45
    C[33] += result48; // c46
    C[28] -= result48; // c55
    C[34] -= result48; // c56
    C[11] -= result48; // c62
    C[17] += result48; // c63
    C[23] += result48; // c64
    C[29] += result48; // c65

    // Expression: (a31-a41+a53+a54-a63-a64)(b12+b13+b14+b15-b21-b22+b31+b32)(-c26+c36-c46-c54+c56)
    float termA49 = A[12] - A[18] + A[26] + A[27] - A[32] - A[33];
    float termB49 = B[1] + B[2] + B[3] + B[4] - B[6] - B[7] + B[12] + B[13];
    float result49 = termA49 * termB49;
    C[31] -= result49; // c26
    C[32] += result49; // c36
    C[33] -= result49; // c46
    C[22] -= result49; // c54
    C[34] += result49; // c56

    // Expression: (a26+a36-a46-a54+a56)(-b31+b41-b53-b54+b63+b64)(c12-c13-c14-c15-c21-c22+c31+c32)
    float termA50 = A[11] + A[17] - A[23] - A[27] + A[29];
    float termB50 = -B[12] + B[18] - B[26] - B[27] + B[32] + B[33];
    float result50 = termA50 * termB50;
    C[6] += result50; // c12
    C[12] -= result50; // c13
    C[18] -= result50; // c14
    C[24] -= result50; // c15
    C[1] -= result50; // c21
    C[7] -= result50; // c22
    C[2] += result50; // c31
    C[8] += result50; // c32

    // Expression: (a12+a13+a14+a15-a21-a22-a31-a32)(-b26+b36-b46-b54+b56)(c31-c41-c53-c54+c63+c64)
    float termA51 = A[1] + A[2] + A[3] + A[4] - A[6] - A[7] - A[12] - A[13];
    float termB51 = -B[11] + B[17] - B[23] - B[27] + B[29];
    float result51 = termA51 * termB51;
    C[2] += result51; // c31
    C[3] -= result51; // c41
    C[16] -= result51; // c53
    C[22] -= result51; // c54
    C[17] += result51; // c63
    C[23] += result51; // c64

    // Expression: (a13+a14-a23-a24-a36+a46)(b45+b46-b55-b56+b62+b63+b64+b65)(-c21-c23+c31-c41+c51)
    float termA52 = A[2] + A[3] - A[8] - A[9] - A[17] + A[23];
    float termB52 = B[22] + B[23] - B[28] - B[29] + B[31] + B[32] + B[33] + B[34];
    float result52 = termA52 * termB52;
    C[1] -= result52; // c21
    C[13] -= result52; // c23
    C[2] += result52; // c31
    C[3] -= result52; // c41
    C[4] += result52; // c51

    // Expression: (-a21+a23-a31+a41-a51)(-b13-b14+b23+b24-b36+b46)(c45+c46-c55-c56-c62+c63+c64+c65)
    float termA53 = -A[6] + A[8] - A[12] + A[18] - A[24];
    float termB53 = -B[2] - B[3] + B[8] + B[9] - B[17] + B[23];
    float result53 = termA53 * termB53;
    C[27] += result53; // c45
    C[33] += result53; // c46
    C[28] -= result53; // c55
    C[34] -= result53; // c56
    C[11] -= result53; // c62
    C[17] += result53; // c63
    C[23] += result53; // c64
    C[29] += result53; // c65

    // Expression: (-a45-a46+a55+a56-a62-a63-a64-a65)(-b21+b23+b31-b41+b51)(-c13-c14+c23+c24-c36+c46)
    float termA54 = -A[22] - A[23] + A[28] + A[29] - A[31] - A[32] - A[33] - A[34];
    float termB54 = -B[6] + B[8] + B[12] - B[18] + B[24];
    float result54 = termA54 * termB54;
    C[12] -= result54; // c13
    C[18] -= result54; // c14
    C[13] += result54; // c23
    C[19] += result54; // c24
    C[32] -= result54; // c36
    C[33] += result54; // c46

    // Expression: (a21-a23+a31-a41)(-b13-b14+b23+b24)(-c42+c43+c44+c45+c52-c53-c54-c55-c62+c63+c64+c65)
    float termA55 = A[6] - A[8] + A[12] - A[18];
    float termB55 = -B[2] - B[3] + B[8] + B[9];
    float result55 = termA55 * termB55;
    C[9] -= result55; // c42
    C[15] += result55; // c43
    C[21] += result55; // c44
    C[27] += result55; // c45
    C[10] += result55; // c52
    C[16] -= result55; // c53
    C[22] -= result55; // c54
    C[28] -= result55; // c55
    C[11] -= result55; // c62
    C[17] += result55; // c63
    C[23] += result55; // c64
    C[29] += result55; // c65

    // Expression: (-a42-a43-a44-a45+a52+a53+a54+a55-a62-a63-a64-a65)(b21-b23-b31+b41)(-c13-c14+c23+c24)
    float termA56 = -A[19] - A[20] - A[21] - A[22] + A[25] + A[26] + A[27] + A[28] - A[31] - A[32] - A[33] - A[34];
    float termB56 = B[6] - B[8] - B[12] + B[18];
    float result56 = termA56 * termB56;
    C[12] -= result56; // c13
    C[18] -= result56; // c14
    C[13] += result56; // c23
    C[19] += result56; // c24

    // Expression: (a13+a14-a23-a24)(b42+b43+b44+b45-b52-b53-b54-b55+b62+b63+b64+b65)(c21+c23-c31+c41)
    float termA57 = A[2] + A[3] - A[8] - A[9];
    float termB57 = B[19] + B[20] + B[21] + B[22] - B[25] - B[26] - B[27] - B[28] + B[31] + B[32] + B[33] + B[34];
    float result57 = termA57 * termB57;
    C[1] += result57; // c21
    C[13] += result57; // c23
    C[2] -= result57; // c31
    C[3] += result57; // c41

    // Expression: (a36-a46-a54+a56)(-b53-b54+b63+b64)(-c12+c13+c14+c15+c22-c23-c24-c25-c32+c33+c34+c35)
    float termA58 = A[17] - A[23] - A[27] + A[29];
    float termB58 = -B[26] - B[27] + B[32] + B[33];
    float result58 = termA58 * termB58;
    C[6] -= result58; // c12
    C[12] += result58; // c13
    C[18] += result58; // c14
    C[24] += result58; // c15
    C[7] += result58; // c22
    C[13] -= result58; // c23
    C[19] -= result58; // c24
    C[25] -= result58; // c25
    C[8] -= result58; // c32
    C[14] += result58; // c33
    C[20] += result58; // c34
    C[26] += result58; // c35

    // Expression: (-a12-a13-a14-a15+a22+a23+a24+a25+a32+a33+a34+a35)(b36-b46-b54+b56)(-c53-c54+c63+c64)
    float termA59 = -A[1] - A[2] - A[3] - A[4] + A[7] + A[8] + A[9] + A[10] + A[13] + A[14] + A[15] + A[16];
    float termB59 = B[17] - B[23] - B[27] + B[29];
    float result59 = termA59 * termB59;
    C[16] -= result59; // c53
    C[22] -= result59; // c54
    C[17] += result59; // c63
    C[23] += result59; // c64

    // Expression: (-a53-a54+a63+a64)(b12+b13+b14+b15-b22-b23-b24-b25+b32+b33+b34+b35)(c36-c46-c54+c56)
    float termA60 = -A[26] - A[27] + A[32] + A[33];
    float termB60 = B[1] + B[2] + B[3] + B[4] - B[7] - B[8] - B[9] - B[10] + B[13] + B[14] + B[15] + B[16];
    float result60 = termA60 * termB60;
    C[32] += result60; // c36
    C[33] -= result60; // c46
    C[22] -= result60; // c54
    C[34] += result60; // c56

    // Expression: (a26+a36-a46-a52-a53-a54+a56)(-b31+b41-b51+b61)(c11+c13+c14+c15)
    float termA61 = A[11] + A[17] - A[23] - A[25] - A[26] - A[27] + A[29];
    float termB61 = -B[12] + B[18] - B[24] + B[30];
    float result61 = termA61 * termB61;
    C[0] += result61; // c11
    C[12] += result61; // c13
    C[18] += result61; // c14
    C[24] += result61; // c15

    // Expression: (-a11+a13+a14+a15)(-b26+b36-b46-b52-b53-b54+b56)(-c31+c41-c51+c61)
    float termA62 = -A[0] + A[2] + A[3] + A[4];
    float termB62 = -B[11] + B[17] - B[23] - B[25] - B[26] - B[27] + B[29];
    float result62 = termA62 * termB62;
    C[2] -= result62; // c31
    C[3] += result62; // c41
    C[4] -= result62; // c51
    C[5] += result62; // c61

    // Expression: (-a31+a41-a51+a61)(-b11+b13+b14+b15)(-c26+c36-c46+c52-c53-c54+c56)
    float termA63 = -A[12] + A[18] - A[24] + A[30];
    float termB63 = -B[0] + B[2] + B[3] + B[4];
    float result63 = termA63 * termB63;
    C[31] -= result63; // c26
    C[32] += result63; // c36
    C[33] -= result63; // c46
    C[10] += result63; // c52
    C[16] -= result63; // c53
    C[22] -= result63; // c54
    C[34] += result63; // c56

    // Expression: (-a21+a23+a24+a25-a31+a41-a51)(-b16+b26-b36+b46)(c62-c63-c64+c66)
    float termA64 = -A[6] + A[8] + A[9] + A[10] - A[12] + A[18] - A[24];
    float termB64 = -B[5] + B[11] - B[17] + B[23];
    float result64 = termA64 * termB64;
    C[11] += result64; // c62
    C[17] -= result64; // c63
    C[23] -= result64; // c64
    C[35] += result64; // c66

    // Expression: (a62+a63+a64-a66)(-b21+b23+b24+b25+b31-b41+b51)(-c16+c26-c36+c46)
    float termA65 = A[31] + A[32] + A[33] - A[35];
    float termB65 = -B[6] + B[8] + B[9] + B[10] + B[12] - B[18] + B[24];
    float result65 = termA65 * termB65;
    C[30] -= result65; // c16
    C[31] += result65; // c26
    C[32] -= result65; // c36
    C[33] += result65; // c46

    // Expression: (a16-a26-a36+a46)(-b62-b63-b64+b66)(-c21-c23-c24-c25+c31-c41+c51)
    float termA66 = A[5] - A[11] - A[17] + A[23];
    float termB66 = -B[31] - B[32] - B[33] + B[35];
    float result66 = termA66 * termB66;
    C[1] -= result66; // c21
    C[13] -= result66; // c23
    C[19] -= result66; // c24
    C[25] -= result66; // c25
    C[2] += result66; // c31
    C[3] -= result66; // c41
    C[4] += result66; // c51

    // Expression: (-a21-a22-a31+a41)(-b11+b23+b24)(-c14-c15+c24+c25-c34-c35-c42+c43+c44+c45+c52-c53-c54-c55-c62+c63+c64+c65)
    float termA67 = -A[6] - A[7] - A[12] + A[18];
    float termB67 = -B[0] + B[8] + B[9];
    float result67 = termA67 * termB67;
    C[18] -= result67; // c14
    C[24] -= result67; // c15
    C[19] += result67; // c24
    C[25] += result67; // c25
    C[20] -= result67; // c34
    C[26] -= result67; // c35
    C[9] -= result67; // c42
    C[15] += result67; // c43
    C[21] += result67; // c44
    C[27] += result67; // c45
    C[10] += result67; // c52
    C[16] -= result67; // c53
    C[22] -= result67; // c54
    C[28] -= result67; // c55
    C[11] -= result67; // c62
    C[17] += result67; // c63
    C[23] += result67; // c64
    C[29] += result67; // c65

    // Expression: (a14+a15-a24-a25-a34-a35+a42+a43+a44+a45-a52-a53-a54-a55+a62+a63+a64+a65)(b21+b22-b31+b41)(c11+c23+c24)
    float termA68 = A[3] + A[4] - A[9] - A[10] - A[15] - A[16] + A[19] + A[20] + A[21] + A[22] - A[25] - A[26] - A[27] - A[28] + A[31] + A[32] + A[33] + A[34];
    float termB68 = B[6] + B[7] - B[12] + B[18];
    float result68 = termA68 * termB68;
    C[0] += result68; // c11
    C[13] += result68; // c23
    C[19] += result68; // c24

    // Expression: (-a11+a23+a24)(-b14-b15+b24+b25-b34-b35+b42+b43+b44+b45-b52-b53-b54-b55+b62+b63+b64+b65)(c21+c22-c31+c41)
    float termA69 = -A[0] + A[8] + A[9];
    float termB69 = -B[3] - B[4] + B[9] + B[10] - B[15] - B[16] + B[19] + B[20] + B[21] + B[22] - B[25] - B[26] - B[27] - B[28] + B[31] + B[32] + B[33] + B[34];
    float result69 = termA69 * termB69;
    C[1] += result69; // c21
    C[7] += result69; // c22
    C[2] -= result69; // c31
    C[3] += result69; // c41

    // Expression: (a36-a46+a55+a56)(-b53-b54+b66)(c12-c13-c14-c15-c22+c23+c24+c25+c32-c33-c34-c35-c42+c43+c52-c53-c62+c63)
    float termA70 = A[17] - A[23] + A[28] + A[29];
    float termB70 = -B[26] - B[27] + B[35];
    float result70 = termA70 * termB70;
    C[6] += result70; // c12
    C[12] -= result70; // c13
    C[18] -= result70; // c14
    C[24] -= result70; // c15
    C[7] -= result70; // c22
    C[13] += result70; // c23
    C[19] += result70; // c24
    C[25] += result70; // c25
    C[8] += result70; // c32
    C[14] -= result70; // c33
    C[20] -= result70; // c34
    C[26] -= result70; // c35
    C[9] -= result70; // c42
    C[15] += result70; // c43
    C[10] += result70; // c52
    C[16] -= result70; // c53
    C[11] -= result70; // c62
    C[17] += result70; // c63

    // Expression: (a12+a13+a14+a15-a22-a23-a24-a25-a32-a33-a34-a35+a42+a43-a52-a53+a62+a63)(b36-b46+b55+b56)(-c53-c54+c66)
    float termA71 = A[1] + A[2] + A[3] + A[4] - A[7] - A[8] - A[9] - A[10] - A[13] - A[14] - A[15] - A[16] + A[19] + A[20] - A[25] - A[26] + A[31] + A[32];
    float termB71 = B[17] - B[23] + B[28] + B[29];
    float result71 = termA71 * termB71;
    C[16] -= result71; // c53
    C[22] -= result71; // c54
    C[35] += result71; // c66

    // Expression: (-a53-a54+a66)(-b12-b13-b14-b15+b22+b23+b24+b25-b32-b33-b34-b35+b42+b43-b52-b53+b62+b63)(c36-c46+c55+c56)
    float termA72 = -A[26] - A[27] + A[35];
    float termB72 = -B[1] - B[2] - B[3] - B[4] + B[7] + B[8] + B[9] + B[10] - B[13] - B[14] - B[15] - B[16] + B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result72 = termA72 * termB72;
    C[32] += result72; // c36
    C[33] -= result72; // c46
    C[28] += result72; // c55
    C[34] += result72; // c56

    // Expression: -(a31(-b11+b13)(c23+c24+c32-c33-c34-c42+c43+c44))
    float termA73 = A[12];
    float termB73 = -B[0] + B[2];
    float result73 = -(termA73 * termB73);
    C[13] += result73; // c23
    C[19] += result73; // c24
    C[8] += result73; // c32
    C[14] -= result73; // c33
    C[20] -= result73; // c34
    C[9] -= result73; // c42
    C[15] += result73; // c43
    C[21] += result73; // c44

    // Expression: (a23+a24+a32+a33+a34-a42-a43-a44)b31(c11+c13)
    float termA74 = A[8] + A[9] + A[13] + A[14] + A[15] - A[19] - A[20] - A[21];
    float termB74 = B[12];
    float result74 = termA74 * termB74;
    C[0] += result74; // c11
    C[12] += result74; // c13

    // Expression: (a11-a13)(b23+b24-b32-b33-b34+b42+b43+b44)c31
    float termA75 = A[0] - A[2];
    float termB75 = B[8] + B[9] - B[13] - B[14] - B[15] + B[19] + B[20] + B[21];
    float result75 = termA75 * termB75;
    C[2] += result75; // c31

    // Expression: a46(-b64+b66)(c33+c34+c35-c43-c44-c45+c53+c54)
    float termA76 = A[23];
    float termB76 = -B[33] + B[35];
    float result76 = termA76 * termB76;
    C[14] += result76; // c33
    C[20] += result76; // c34
    C[26] += result76; // c35
    C[15] -= result76; // c43
    C[21] -= result76; // c44
    C[27] -= result76; // c45
    C[16] += result76; // c53
    C[22] += result76; // c54

    // Expression: (a33+a34+a35-a43-a44-a45+a53+a54)b46(-c64+c66)
    float termA77 = A[14] + A[15] + A[16] - A[20] - A[21] - A[22] + A[26] + A[27];
    float termB77 = B[23];
    float result77 = termA77 * termB77;
    C[23] -= result77; // c64
    C[35] += result77; // c66

    // Expression: (-a64+a66)(b33+b34+b35-b43-b44-b45+b53+b54)c46
    float termA78 = -A[33] + A[35];
    float termB78 = B[14] + B[15] + B[16] - B[20] - B[21] - B[22] + B[26] + B[27];
    float result78 = termA78 * termB78;
    C[33] += result78; // c46

    // Expression: (a21+a22+a31+a32-a41-a42+a51+a52)(-b23-b24+b26)(-c15-c16+c25+c26-c35-c36+c45+c46-c55-c56+c61+c63+c64+c65)
    float termA79 = A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25];
    float termB79 = -B[8] - B[9] + B[11];
    float result79 = termA79 * termB79;
    C[24] -= result79; // c15
    C[30] -= result79; // c16
    C[25] += result79; // c25
    C[31] += result79; // c26
    C[26] -= result79; // c35
    C[32] -= result79; // c36
    C[27] += result79; // c45
    C[33] += result79; // c46
    C[28] -= result79; // c55
    C[34] -= result79; // c56
    C[5] += result79; // c61
    C[17] += result79; // c63
    C[23] += result79; // c64
    C[29] += result79; // c65

    // Expression: (a15+a16-a25-a26-a35-a36+a45+a46-a55-a56-a61+a63+a64+a65)(-b21-b22+b31+b32-b41-b42+b51+b52)(-c23-c24+c26)
    float termA80 = A[4] + A[5] - A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29] - A[30] + A[32] + A[33] + A[34];
    float termB80 = -B[6] - B[7] + B[12] + B[13] - B[18] - B[19] + B[24] + B[25];
    float result80 = termA80 * termB80;
    C[13] -= result80; // c23
    C[19] -= result80; // c24
    C[31] += result80; // c26

    // Expression: (-a23-a24+a26)(-b15-b16+b25+b26-b35-b36+b45+b46-b55-b56+b61-b63-b64+b65+2*b66)(-c21-c22+c31+c32-c41-c42+c51+c52)
    float termA81 = -A[8] - A[9] + A[11];
    float termB81 = -B[4] - B[5] + B[10] + B[11] - B[16] - B[17] + B[22] + B[23] - B[28] - B[29] + B[30] - B[32] - B[33] + B[34];
    float result81 = termA81 * termB81;
    C[1] -= result81; // c21
    C[7] -= result81; // c22
    C[2] += result81; // c31
    C[8] += result81; // c32
    C[3] -= result81; // c41
    C[9] -= result81; // c42
    C[4] += result81; // c51
    C[10] += result81; // c52

    // Expression: (a25+a26+a35+a36-a45-a46+a55+a56)(-b51+b53+b54)(-c12+c13+c14-c16+c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA82 = A[10] + A[11] + A[16] + A[17] - A[22] - A[23] + A[28] + A[29];
    float termB82 = -B[24] + B[26] + B[27];
    float result82 = termA82 * termB82;
    C[6] -= result82; // c12
    C[12] += result82; // c13
    C[18] += result82; // c14
    C[30] -= result82; // c16
    C[1] += result82; // c21
    C[7] += result82; // c22
    C[2] -= result82; // c31
    C[8] -= result82; // c32
    C[3] += result82; // c41
    C[9] += result82; // c42
    C[4] -= result82; // c51
    C[10] -= result82; // c52
    C[5] += result82; // c61
    C[11] += result82; // c62

    // Expression: (-2*a11-a12+a13+a14-a16+a21+a22+a31+a32-a41-a42+a51+a52-a61-a62)(-b25-b26+b35+b36-b45-b46+b55+b56)(c51+c53+c54)
    float termA83 = -A[1] + A[2] + A[3] - A[5] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25] - A[30] - A[31];
    float termB83 = -B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29];
    float result83 = termA83 * termB83;
    C[4] += result83; // c51
    C[16] += result83; // c53
    C[22] += result83; // c54

    // Expression: (a51-a53-a54)(-2*b11-b12+b13+b14-b16+b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(-c25-c26+c35+c36-c45-c46+c55+c56)
    float termA84 = A[24] - A[26] - A[27];
    float termB84 = -B[1] + B[2] + B[3] - B[5] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result84 = termA84 * termB84;
    C[25] -= result84; // c25
    C[31] -= result84; // c26
    C[26] += result84; // c35
    C[32] += result84; // c36
    C[27] -= result84; // c45
    C[33] -= result84; // c46
    C[28] += result84; // c55
    C[34] += result84; // c56

    // Expression: (-a21+a23+a24+a25-a31+a41-a51+a61)(b16-b26+b36-b46+b55+b56)(c52-c53-c54+c66)
    float termA85 = -A[6] + A[8] + A[9] + A[10] - A[12] + A[18] - A[24] + A[30];
    float termB85 = B[5] - B[11] + B[17] - B[23] + B[28] + B[29];
    float result85 = termA85 * termB85;
    C[10] += result85; // c52
    C[16] -= result85; // c53
    C[22] -= result85; // c54
    C[35] += result85; // c66

    // Expression: (-a52-a53-a54+a66)(b21-b23-b24-b25-b31+b41-b51+b61)(c16-c26+c36-c46+c55+c56)
    float termA86 = -A[25] - A[26] - A[27] + A[35];
    float termB86 = B[6] - B[8] - B[9] - B[10] - B[12] + B[18] - B[24] + B[30];
    float result86 = termA86 * termB86;
    C[30] += result86; // c16
    C[31] -= result86; // c26
    C[32] += result86; // c36
    C[33] -= result86; // c46
    C[28] += result86; // c55
    C[34] += result86; // c56

    // Expression: (a16-a26-a36+a46-a55-a56)(-b52-b53-b54+b66)(c21+c23+c24+c25-c31+c41-c51+c61)
    float termA87 = A[5] - A[11] - A[17] + A[23] - A[28] - A[29];
    float termB87 = -B[25] - B[26] - B[27] + B[35];
    float result87 = termA87 * termB87;
    C[1] += result87; // c21
    C[13] += result87; // c23
    C[19] += result87; // c24
    C[25] += result87; // c25
    C[2] -= result87; // c31
    C[3] += result87; // c41
    C[4] -= result87; // c51
    C[5] += result87; // c61

    // Expression: (a16-a26-a36+a46+a52+a53+a54-a56)(b21+b22-b31+b41-b51+b61)(c11+c23+c24+c25)
    float termA88 = A[5] - A[11] - A[17] + A[23] + A[25] + A[26] + A[27] - A[29];
    float termB88 = B[6] + B[7] - B[12] + B[18] - B[24] + B[30];
    float result88 = termA88 * termB88;
    C[0] += result88; // c11
    C[13] += result88; // c23
    C[19] += result88; // c24
    C[25] += result88; // c25

    // Expression: (a11-a23-a24-a25)(b16-b26+b36-b46-b52-b53-b54+b56)(c21+c22-c31+c41-c51+c61)
    float termA89 = A[0] - A[8] - A[9] - A[10];
    float termB89 = B[5] - B[11] + B[17] - B[23] - B[25] - B[26] - B[27] + B[29];
    float result89 = termA89 * termB89;
    C[1] += result89; // c21
    C[7] += result89; // c22
    C[2] -= result89; // c31
    C[3] += result89; // c41
    C[4] -= result89; // c51
    C[5] += result89; // c61

    // Expression: (a21+a22+a31-a41+a51-a61)(-b11+b23+b24+b25)(c16-c26+c36-c46+c52-c53-c54+c56)
    float termA90 = A[6] + A[7] + A[12] - A[18] + A[24] - A[30];
    float termB90 = -B[0] + B[8] + B[9] + B[10];
    float result90 = termA90 * termB90;
    C[30] += result90; // c16
    C[31] -= result90; // c26
    C[32] += result90; // c36
    C[33] -= result90; // c46
    C[10] += result90; // c52
    C[16] -= result90; // c53
    C[22] -= result90; // c54
    C[34] += result90; // c56

    // Expression: (a46+a53+a54-a56)(-b12-b13+b22+b23-b32-b33+b42+b43-b52-b53+b62+b63)(-c35+c45)
    float termA91 = A[23] + A[26] + A[27] - A[29];
    float termB91 = -B[1] - B[2] + B[7] + B[8] - B[13] - B[14] + B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result91 = termA91 * termB91;
    C[26] -= result91; // c35
    C[27] += result91; // c45

    // Expression: (a35-a45)(-b46-b53-b54+b56)(c12-c13-c22+c23+c32-c33-c42+c43+c52-c53-c62+c63)
    float termA92 = A[16] - A[22];
    float termB92 = -B[23] - B[26] - B[27] + B[29];
    float result92 = termA92 * termB92;
    C[6] += result92; // c12
    C[12] -= result92; // c13
    C[7] -= result92; // c22
    C[13] += result92; // c23
    C[8] += result92; // c32
    C[14] -= result92; // c33
    C[9] -= result92; // c42
    C[15] += result92; // c43
    C[10] += result92; // c52
    C[16] -= result92; // c53
    C[11] -= result92; // c62
    C[17] += result92; // c63

    // Expression: (-a12-a13+a22+a23+a32+a33-a42-a43+a52+a53-a62-a63)(-b35+b45)(-c46-c53-c54+c56)
    float termA93 = -A[1] - A[2] + A[7] + A[8] + A[13] + A[14] - A[19] - A[20] + A[25] + A[26] - A[31] - A[32];
    float termB93 = -B[16] + B[22];
    float result93 = termA93 * termB93;
    C[33] -= result93; // c46
    C[16] -= result93; // c53
    C[22] -= result93; // c54
    C[34] += result93; // c56

    // Expression: (-a21+a23+a24-a31)(-b14-b15+b24+b25-b34-b35+b44+b45-b54-b55+b64+b65)(-c32+c42)
    float termA94 = -A[6] + A[8] + A[9] - A[12];
    float termB94 = -B[3] - B[4] + B[9] + B[10] - B[15] - B[16] + B[21] + B[22] - B[27] - B[28] + B[33] + B[34];
    float result94 = termA94 * termB94;
    C[8] -= result94; // c32
    C[9] += result94; // c42

    // Expression: (-a32+a42)(-b21+b23+b24+b31)(-c14-c15+c24+c25-c34-c35+c44+c45-c54-c55+c64+c65)
    float termA95 = -A[13] + A[19];
    float termB95 = -B[6] + B[8] + B[9] + B[12];
    float result95 = termA95 * termB95;
    C[18] -= result95; // c14
    C[24] -= result95; // c15
    C[19] += result95; // c24
    C[25] += result95; // c25
    C[20] -= result95; // c34
    C[26] -= result95; // c35
    C[21] += result95; // c44
    C[27] += result95; // c45
    C[22] -= result95; // c54
    C[28] -= result95; // c55
    C[23] += result95; // c64
    C[29] += result95; // c65

    // Expression: (-a14-a15+a24+a25+a34+a35-a44-a45+a54+a55-a64-a65)(-b32+b42)(-c21-c23-c24+c31)
    float termA96 = -A[3] - A[4] + A[9] + A[10] + A[15] + A[16] - A[21] - A[22] + A[27] + A[28] - A[33] - A[34];
    float termB96 = -B[13] + B[19];
    float result96 = termA96 * termB96;
    C[1] -= result96; // c21
    C[13] -= result96; // c23
    C[19] -= result96; // c24
    C[2] += result96; // c31

    // Expression: -(a54(-b33-b34+b36+b43+b44-b46-b53-b54+b63+b64)(-c45-c46+c55+c56))
    float termA97 = A[27];
    float termB97 = -B[14] - B[15] + B[17] + B[20] + B[21] - B[23] - B[26] - B[27] + B[32] + B[33];
    float result97 = -(termA97 * termB97);
    C[27] -= result97; // c45
    C[33] -= result97; // c46
    C[28] += result97; // c55
    C[34] += result97; // c56

    // Expression: (a45+a46-a55-a56)b54(-c33-c34+c36+c43+c44-c46-c53-c54+c63+c64)
    float termA98 = A[22] + A[23] - A[28] - A[29];
    float termB98 = B[27];
    float result98 = termA98 * termB98;
    C[14] -= result98; // c33
    C[20] -= result98; // c34
    C[32] += result98; // c36
    C[15] += result98; // c43
    C[21] += result98; // c44
    C[33] -= result98; // c46
    C[16] -= result98; // c53
    C[22] -= result98; // c54
    C[17] += result98; // c63
    C[23] += result98; // c64

    // Expression: (a33+a34-a36-a43-a44+a46+a53+a54-a63-a64)(-b45-b46+b55+b56)c54
    float termA99 = A[14] + A[15] - A[17] - A[20] - A[21] + A[23] + A[26] + A[27] - A[32] - A[33];
    float termB99 = -B[22] - B[23] + B[28] + B[29];
    float result99 = termA99 * termB99;
    C[22] += result99; // c54

    // Expression: -(a23(-b13-b14+b23+b24+b31-b33-b34-b41+b43+b44)(-c21-c22+c31+c32))
    float termA100 = A[8];
    float termB100 = -B[2] - B[3] + B[8] + B[9] + B[12] - B[14] - B[15] - B[18] + B[20] + B[21];
    float result100 = -(termA100 * termB100);
    C[1] -= result100; // c21
    C[7] -= result100; // c22
    C[2] += result100; // c31
    C[8] += result100; // c32

    // Expression: (-a21-a22-a31-a32)b23(-c13-c14+c23+c24-c31-c33-c34+c41+c43+c44)
    float termA101 = -A[6] - A[7] - A[12] - A[13];
    float termB101 = B[8];
    float result101 = termA101 * termB101;
    C[12] -= result101; // c13
    C[18] -= result101; // c14
    C[13] += result101; // c23
    C[19] += result101; // c24
    C[2] -= result101; // c31
    C[14] -= result101; // c33
    C[20] -= result101; // c34
    C[3] += result101; // c41
    C[15] += result101; // c43
    C[21] += result101; // c44

    // Expression: (-a13-a14+a23+a24-a31+a33+a34+a41-a43-a44)(-b21-b22+b31+b32)c23
    float termA102 = -A[2] - A[3] + A[8] + A[9] - A[12] + A[14] + A[15] + A[18] - A[20] - A[21];
    float termB102 = -B[6] - B[7] + B[12] + B[13];
    float result102 = termA102 * termB102;
    C[13] += result102; // c23

    // Expression: (-a42-a43+a52+a53-a62-a63)(-b31+b41)(c11+c13+c14)
    float termA103 = -A[19] - A[20] + A[25] + A[26] - A[31] - A[32];
    float termB103 = -B[12] + B[18];
    float result103 = termA103 * termB103;
    C[0] += result103; // c11
    C[12] += result103; // c13
    C[18] += result103; // c14

    // Expression: (a11-a13-a14)(b42+b43-b52-b53+b62+b63)(-c31+c41)
    float termA104 = A[0] - A[2] - A[3];
    float termB104 = B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result104 = termA104 * termB104;
    C[2] -= result104; // c31
    C[3] += result104; // c41

    // Expression: (a31-a41)(-b11+b13+b14)(-c42+c43+c52-c53-c62+c63)
    float termA105 = A[12] - A[18];
    float termB105 = -B[0] + B[2] + B[3];
    float result105 = termA105 * termB105;
    C[9] -= result105; // c42
    C[15] += result105; // c43
    C[10] += result105; // c52
    C[16] -= result105; // c53
    C[11] -= result105; // c62
    C[17] += result105; // c63

    // Expression: (a14+a15-a24-a25-a34-a35)(-b36+b46)(-c63-c64+c66)
    float termA106 = A[3] + A[4] - A[9] - A[10] - A[15] - A[16];
    float termB106 = -B[17] + B[23];
    float result106 = termA106 * termB106;
    C[17] -= result106; // c63
    C[23] -= result106; // c64
    C[35] += result106; // c66

    // Expression: (2*a45+2*a46-2*a55-2*a56+a63+a64+2*a65+a66)(b14+b15-b24-b25+b34+b35)(-c36+c46)
    float termA107 = A[32] + A[33] + A[35];
    float termB107 = B[3] + B[4] - B[9] - B[10] + B[15] + B[16];
    float result107 = termA107 * termB107;
    C[32] -= result107; // c36
    C[33] += result107; // c46

    // Expression: (a36-a46)(-b63-b64+b66)(c14+c15-c24-c25+c34+c35)
    float termA108 = A[17] - A[23];
    float termB108 = -B[32] - B[33] + B[35];
    float result108 = termA108 * termB108;
    C[18] += result108; // c14
    C[24] += result108; // c15
    C[19] -= result108; // c24
    C[25] -= result108; // c25
    C[20] += result108; // c34
    C[26] += result108; // c35

    // Expression: (a31-a41+a51)(b11-b13-b14+b16)(-c25-c26+c35+c36-c45-c46+c55+c56-c62+c63+c64+c65)
    float termA109 = A[12] - A[18] + A[24];
    float termB109 = B[0] - B[2] - B[3] + B[5];
    float result109 = termA109 * termB109;
    C[25] -= result109; // c25
    C[31] -= result109; // c26
    C[26] += result109; // c35
    C[32] += result109; // c36
    C[27] -= result109; // c45
    C[33] -= result109; // c46
    C[28] += result109; // c55
    C[34] += result109; // c56
    C[11] -= result109; // c62
    C[17] += result109; // c63
    C[23] += result109; // c64
    C[29] += result109; // c65

    // Expression: (-a25-a26-a35-a36+a45+a46-a55-a56+a62+a63+a64+a65)(b31-b41+b51)(-c11-c13-c14+c16)
    float termA110 = -A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29] + A[31] + A[32] + A[33] + A[34];
    float termB110 = B[12] - B[18] + B[24];
    float result110 = termA110 * termB110;
    C[0] -= result110; // c11
    C[12] -= result110; // c13
    C[18] -= result110; // c14
    C[30] += result110; // c16

    // Expression: (a11-a13-a14+a16)(-b25-b26+b35+b36-b45-b46+b55+b56+b62+b63+b64+b65)(c31-c41+c51)
    float termA111 = A[0] - A[2] - A[3] + A[5];
    float termB111 = -B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29] + B[31] + B[32] + B[33] + B[34];
    float result111 = termA111 * termB111;
    C[2] += result111; // c31
    C[3] -= result111; // c41
    C[4] += result111; // c51

    // Expression: (-a26-a36+a46)(b61-b63-b64+b66)(-c12+c13+c14+c15-c21-c22+c31+c32-c41-c42+c51+c52)
    float termA112 = -A[11] - A[17] + A[23];
    float termB112 = B[30] - B[32] - B[33] + B[35];
    float result112 = termA112 * termB112;
    C[6] -= result112; // c12
    C[12] += result112; // c13
    C[18] += result112; // c14
    C[24] += result112; // c15
    C[1] -= result112; // c21
    C[7] -= result112; // c22
    C[2] += result112; // c31
    C[8] += result112; // c32
    C[3] -= result112; // c41
    C[9] -= result112; // c42
    C[4] += result112; // c51
    C[10] += result112; // c52

    // Expression: (-a12-a13-a14-a15+a21+a22+a31+a32-a41-a42+a51+a52)(b26-b36+b46)(-c61-c63-c64+c66)
    float termA113 = -A[1] - A[2] - A[3] - A[4] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25];
    float termB113 = B[11] - B[17] + B[23];
    float result113 = termA113 * termB113;
    C[5] -= result113; // c61
    C[17] -= result113; // c63
    C[23] -= result113; // c64
    C[35] += result113; // c66

    // Expression: (a61-a63-a64+a66)(b12+b13+b14+b15-b21-b22+b31+b32-b41-b42+b51+b52)(c26-c36+c46)
    float termA114 = A[30] - A[32] - A[33] + A[35];
    float termB114 = B[1] + B[2] + B[3] + B[4] - B[6] - B[7] + B[12] + B[13] - B[18] - B[19] + B[24] + B[25];
    float result114 = termA114 * termB114;
    C[31] += result114; // c26
    C[32] -= result114; // c36
    C[33] += result114; // c46

    // Expression: (a25+a26+a35+a36-a45-a46+a55+a56-a64-a65)(b31-b41+b53+b54)(c12-c13-c14+c16-c21-c22+c31+c32)
    float termA115 = A[10] + A[11] + A[16] + A[17] - A[22] - A[23] + A[28] + A[29] - A[33] - A[34];
    float termB115 = B[12] - B[18] + B[26] + B[27];
    float result115 = termA115 * termB115;
    C[6] += result115; // c12
    C[12] -= result115; // c13
    C[18] -= result115; // c14
    C[30] += result115; // c16
    C[1] -= result115; // c21
    C[7] -= result115; // c22
    C[2] += result115; // c31
    C[8] += result115; // c32

    // Expression: (2*a11+a12-a13-a14+a16-a21-a22-a31-a32)(-b25-b26+b35+b36-b45-b46+b55+b56+b64+b65)(-c31+c41+c53+c54)
    float termA116 = A[1] - A[2] - A[3] + A[5] - A[6] - A[7] - A[12] - A[13];
    float termB116 = -B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29] + B[33] + B[34];
    float result116 = termA116 * termB116;
    C[2] -= result116; // c31
    C[3] += result116; // c41
    C[16] += result116; // c53
    C[22] += result116; // c54

    // Expression: (-a31+a41-a53-a54)(2*b11+b12-b13-b14+b16-b21-b22+b31+b32)(-c25-c26+c35+c36-c45-c46+c55+c56+c64+c65)
    float termA117 = -A[12] + A[18] - A[26] - A[27];
    float termB117 = B[1] - B[2] - B[3] + B[5] - B[6] - B[7] + B[12] + B[13];
    float result117 = termA117 * termB117;
    C[25] -= result117; // c25
    C[31] -= result117; // c26
    C[26] += result117; // c35
    C[32] += result117; // c36
    C[27] -= result117; // c45
    C[33] -= result117; // c46
    C[28] += result117; // c55
    C[34] += result117; // c56
    C[23] += result117; // c64
    C[29] += result117; // c65

    // Expression: (-a12-a13+a21+a22+a31+a32-a41-a42+a51+a52)(b23+b24-b36+b46)(c45+c46-c55-c56+c61+c63+c64+c65)
    float termA118 = -A[1] - A[2] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25];
    float termB118 = B[8] + B[9] - B[17] + B[23];
    float result118 = termA118 * termB118;
    C[27] += result118; // c45
    C[33] += result118; // c46
    C[28] -= result118; // c55
    C[34] -= result118; // c56
    C[5] += result118; // c61
    C[17] += result118; // c63
    C[23] += result118; // c64
    C[29] += result118; // c65

    // Expression: (a45+a46-a55-a56-a61+a63+a64+a65)(b12+b13-b21-b22+b31+b32-b41-b42+b51+b52)(c23+c24-c36+c46)
    float termA119 = A[22] + A[23] - A[28] - A[29] - A[30] + A[32] + A[33] + A[34];
    float termB119 = B[1] + B[2] - B[6] - B[7] + B[12] + B[13] - B[18] - B[19] + B[24] + B[25];
    float result119 = termA119 * termB119;
    C[13] += result119; // c23
    C[19] += result119; // c24
    C[32] -= result119; // c36
    C[33] += result119; // c46

    // Expression: (a23+a24+a36-a46)(b45+b46-b55-b56+b61-b63-b64+b65+2*b66)(-c12+c13-c21-c22+c31+c32-c41-c42+c51+c52)
    float termA120 = A[8] + A[9] + A[17] - A[23];
    float termB120 = B[22] + B[23] - B[28] - B[29] + B[30] - B[32] - B[33] + B[34];
    float result120 = termA120 * termB120;
    C[6] -= result120; // c12
    C[12] += result120; // c13
    C[1] -= result120; // c21
    C[7] -= result120; // c22
    C[2] += result120; // c31
    C[8] += result120; // c32
    C[3] -= result120; // c41
    C[9] -= result120; // c42
    C[4] += result120; // c51
    C[10] += result120; // c52

    // Expression: (a45+a46-a55-a56+a64+a65)(b33-b43+b53+b54)(c33+c34-c36+c46)
    float termA121 = A[22] + A[23] - A[28] - A[29] + A[33] + A[34];
    float termB121 = B[14] - B[20] + B[26] + B[27];
    float result121 = termA121 * termB121;
    C[14] += result121; // c33
    C[20] += result121; // c34
    C[32] -= result121; // c36
    C[33] += result121; // c46

    // Expression: (a33+a34-a36+a46)(b45+b46-b55-b56+b64+b65)(c33-c43+c53+c54)
    float termA122 = A[14] + A[15] - A[17] + A[23];
    float termB122 = B[22] + B[23] - B[28] - B[29] + B[33] + B[34];
    float result122 = termA122 * termB122;
    C[14] += result122; // c33
    C[15] -= result122; // c43
    C[16] += result122; // c53
    C[22] += result122; // c54

    // Expression: (a33-a43+a53+a54)(b33+b34-b36+b46)(c45+c46-c55-c56+c64+c65)
    float termA123 = A[14] - A[20] + A[26] + A[27];
    float termB123 = B[14] + B[15] - B[17] + B[23];
    float result123 = termA123 * termB123;
    C[27] += result123; // c45
    C[33] += result123; // c46
    C[28] -= result123; // c55
    C[34] -= result123; // c56
    C[23] += result123; // c64
    C[29] += result123; // c65

    // Expression: (-a12-a13+a21+a22+a31+a32)(b23+b24-b34+b44)(-c31+c41+c43+c44)
    float termA124 = -A[1] - A[2] + A[6] + A[7] + A[12] + A[13];
    float termB124 = B[8] + B[9] - B[15] + B[21];
    float result124 = termA124 * termB124;
    C[2] -= result124; // c31
    C[3] += result124; // c41
    C[15] += result124; // c43
    C[21] += result124; // c44

    // Expression: (a31-a41+a43+a44)(b12+b13-b21-b22+b31+b32)(c23+c24-c34+c44)
    float termA125 = A[12] - A[18] + A[20] + A[21];
    float termB125 = B[1] + B[2] - B[6] - B[7] + B[12] + B[13];
    float result125 = termA125 * termB125;
    C[13] += result125; // c23
    C[19] += result125; // c24
    C[20] -= result125; // c34
    C[21] += result125; // c44

    // Expression: (a23+a24+a34-a44)(b31-b41+b43+b44)(c12-c13-c21-c22+c31+c32)
    float termA126 = A[8] + A[9] + A[15] - A[21];
    float termB126 = B[12] - B[18] + B[20] + B[21];
    float result126 = termA126 * termB126;
    C[6] += result126; // c12
    C[12] -= result126; // c13
    C[1] -= result126; // c21
    C[7] -= result126; // c22
    C[2] += result126; // c31
    C[8] += result126; // c32

    // Expression: (2*a11+a43+a44)(b12+b13-b22-b23+b32+b33)(c21+c22-c34+c44)
    float termA127 = A[20] + A[21];
    float termB127 = B[1] + B[2] - B[7] - B[8] + B[13] + B[14];
    float result127 = termA127 * termB127;
    C[1] += result127; // c21
    C[7] += result127; // c22
    C[20] -= result127; // c34
    C[21] += result127; // c44

    // Expression: (a21+a22+a34-a44)(b43+b44)(-c12+c13+c22-c23-c32+c33)
    float termA128 = A[6] + A[7] + A[15] - A[21];
    float termB128 = B[20] + B[21];
    float result128 = termA128 * termB128;
    C[6] -= result128; // c12
    C[12] += result128; // c13
    C[7] += result128; // c22
    C[13] -= result128; // c23
    C[8] -= result128; // c32
    C[14] += result128; // c33

    // Expression: (a12+a13-a22-a23-a32-a33)(b21+b22-b34+b44)(c43+c44)
    float termA129 = A[1] + A[2] - A[7] - A[8] - A[13] - A[14];
    float termB129 = B[6] + B[7] - B[15] + B[21];
    float result129 = termA129 * termB129;
    C[15] += result129; // c43
    C[21] += result129; // c44

    // Expression: (a33+a34)(b44+b45-b54-b55+b64+b65)(-c33+c43+c55+c56)
    float termA130 = A[14] + A[15];
    float termB130 = B[21] + B[22] - B[27] - B[28] + B[33] + B[34];
    float result130 = termA130 * termB130;
    C[14] -= result130; // c33
    C[15] += result130; // c43
    C[28] += result130; // c55
    C[34] += result130; // c56

    // Expression: (-a33+a43+a55+a56)(b33+b34)(c44+c45-c54-c55+c64+c65)
    float termA131 = -A[14] + A[20] + A[28] + A[29];
    float termB131 = B[14] + B[15];
    float result131 = termA131 * termB131;
    C[21] += result131; // c44
    C[27] += result131; // c45
    C[22] -= result131; // c54
    C[28] -= result131; // c55
    C[23] += result131; // c64
    C[29] += result131; // c65

    // Expression: (a44+a45-a54-a55+a64+a65)(-b33+b43+b55+b56)(c33+c34)
    float termA132 = A[21] + A[22] - A[27] - A[28] + A[33] + A[34];
    float termB132 = -B[14] + B[20] + B[28] + B[29];
    float result132 = termA132 * termB132;
    C[14] += result132; // c33
    C[20] += result132; // c34

    // Expression: (a32-a42-a53-a54+a63+a64)(-b21+b23+b24+b25+b31)(c16-c26+c36-c46-c54+c56)
    float termA133 = A[13] - A[19] - A[26] - A[27] + A[32] + A[33];
    float termB133 = -B[6] + B[8] + B[9] + B[10] + B[12];
    float result133 = termA133 * termB133;
    C[30] += result133; // c16
    C[31] -= result133; // c26
    C[32] += result133; // c36
    C[33] -= result133; // c46
    C[22] -= result133; // c54
    C[34] += result133; // c56

    // Expression: (a16-a26-a36+a46+a54-a56)(b32-b42-b53-b54+b63+b64)(-c21-c23-c24-c25+c31)
    float termA134 = A[5] - A[11] - A[17] + A[23] + A[27] - A[29];
    float termB134 = B[13] - B[19] - B[26] - B[27] + B[32] + B[33];
    float result134 = termA134 * termB134;
    C[1] -= result134; // c21
    C[13] -= result134; // c23
    C[19] -= result134; // c24
    C[25] -= result134; // c25
    C[2] += result134; // c31

    // Expression: (a21-a23-a24-a25+a31)(b16-b26+b36-b46-b54+b56)(-c32+c42-c53-c54+c63+c64)
    float termA135 = A[6] - A[8] - A[9] - A[10] + A[12];
    float termB135 = B[5] - B[11] + B[17] - B[23] - B[27] + B[29];
    float result135 = termA135 * termB135;
    C[8] -= result135; // c32
    C[9] += result135; // c42
    C[16] -= result135; // c53
    C[22] -= result135; // c54
    C[17] += result135; // c63
    C[23] += result135; // c64

    // Expression: (-a13-a14+a23+a24-a35+a45)(-b46-b52-b53-b54+b56)(c21+c23-c31+c41-c51+c61)
    float termA136 = -A[2] - A[3] + A[8] + A[9] - A[16] + A[22];
    float termB136 = -B[23] - B[25] - B[26] - B[27] + B[29];
    float result136 = termA136 * termB136;
    C[1] += result136; // c21
    C[13] += result136; // c23
    C[2] -= result136; // c31
    C[3] += result136; // c41
    C[4] -= result136; // c51
    C[5] += result136; // c61

    // Expression: (a21-a23+a31-a41+a51-a61)(b13+b14-b23-b24-b35+b45)(-c46+c52-c53-c54+c56)
    float termA137 = A[6] - A[8] + A[12] - A[18] + A[24] - A[30];
    float termB137 = B[2] + B[3] - B[8] - B[9] - B[16] + B[22];
    float result137 = termA137 * termB137;
    C[33] -= result137; // c46
    C[10] += result137; // c52
    C[16] -= result137; // c53
    C[22] -= result137; // c54
    C[34] += result137; // c56

    // Expression: (a46+a52+a53+a54-a56)(b21-b23-b31+b41-b51+b61)(c13+c14-c23-c24-c35+c45)
    float termA138 = A[23] + A[25] + A[26] + A[27] - A[29];
    float termB138 = B[6] - B[8] - B[12] + B[18] - B[24] + B[30];
    float result138 = termA138 * termB138;
    C[12] += result138; // c13
    C[18] += result138; // c14
    C[13] -= result138; // c23
    C[19] -= result138; // c24
    C[26] -= result138; // c35
    C[27] += result138; // c45

    // Expression: (a23+a24+a25)(b16-b26+b36-b46+b51-b53-b54+b56)(c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA139 = A[8] + A[9] + A[10];
    float termB139 = B[5] - B[11] + B[17] - B[23] + B[24] - B[26] - B[27] + B[29];
    float result139 = termA139 * termB139;
    C[1] += result139; // c21
    C[7] += result139; // c22
    C[2] -= result139; // c31
    C[8] -= result139; // c32
    C[3] += result139; // c41
    C[9] += result139; // c42
    C[4] -= result139; // c51
    C[10] -= result139; // c52
    C[5] += result139; // c61
    C[11] += result139; // c62

    // Expression: (-a21-a22-a31-a32+a41+a42-a51-a52+a61+a62)(b23+b24+b25)(c16-c26+c36-c46-c51-c53-c54+c56)
    float termA140 = -A[6] - A[7] - A[12] - A[13] + A[18] + A[19] - A[24] - A[25] + A[30] + A[31];
    float termB140 = B[8] + B[9] + B[10];
    float result140 = termA140 * termB140;
    C[30] += result140; // c16
    C[31] -= result140; // c26
    C[32] += result140; // c36
    C[33] -= result140; // c46
    C[4] -= result140; // c51
    C[16] -= result140; // c53
    C[22] -= result140; // c54
    C[34] += result140; // c56

    // Expression: (-a16+a26+a36-a46+a51-a53-a54+a56)(b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(c23+c24+c25)
    float termA141 = -A[5] + A[11] + A[17] - A[23] + A[24] - A[26] - A[27] + A[29];
    float termB141 = B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result141 = termA141 * termB141;
    C[13] += result141; // c23
    C[19] += result141; // c24
    C[25] += result141; // c25

    // Expression: (a52+a53+a54)(b21-b23-b24+b26-b31+b41-b51+b61)(c15+c16-c25-c26+c35+c36-c45-c46+c55+c56)
    float termA142 = A[25] + A[26] + A[27];
    float termB142 = B[6] - B[8] - B[9] + B[11] - B[12] + B[18] - B[24] + B[30];
    float result142 = termA142 * termB142;
    C[24] += result142; // c15
    C[30] += result142; // c16
    C[25] -= result142; // c25
    C[31] -= result142; // c26
    C[26] += result142; // c35
    C[32] += result142; // c36
    C[27] -= result142; // c45
    C[33] -= result142; // c46
    C[28] += result142; // c55
    C[34] += result142; // c56

    // Expression: (a15+a16-a25-a26-a35-a36+a45+a46-a55-a56)(b52+b53+b54)(c21+c23+c24-c26-c31+c41-c51+c61)
    float termA143 = A[4] + A[5] - A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29];
    float termB143 = B[25] + B[26] + B[27];
    float result143 = termA143 * termB143;
    C[1] += result143; // c21
    C[13] += result143; // c23
    C[19] += result143; // c24
    C[31] -= result143; // c26
    C[2] -= result143; // c31
    C[3] += result143; // c41
    C[4] -= result143; // c51
    C[5] += result143; // c61

    // Expression: (-a21+a23+a24-a26-a31+a41-a51+a61)(b15+b16-b25-b26+b35+b36-b45-b46+b55+b56)(-c52+c53+c54)
    float termA144 = -A[6] + A[8] + A[9] - A[11] - A[12] + A[18] - A[24] + A[30];
    float termB144 = B[4] + B[5] - B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29];
    float result144 = termA144 * termB144;
    C[10] -= result144; // c52
    C[16] += result144; // c53
    C[22] += result144; // c54

    // Expression: (a53+a54)(b42+b43-b52-b53+b62+b63)(c35+c36-c45-c46+c55+c56+c64+c65)
    float termA145 = A[26] + A[27];
    float termB145 = B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result145 = termA145 * termB145;
    C[26] += result145; // c35
    C[32] += result145; // c36
    C[27] -= result145; // c45
    C[33] -= result145; // c46
    C[28] += result145; // c55
    C[34] += result145; // c56
    C[23] += result145; // c64
    C[29] += result145; // c65

    // Expression: (a35+a36-a45-a46+a55+a56+a64+a65)(b53+b54)(-c42+c43+c52-c53-c62+c63)
    float termA146 = A[16] + A[17] - A[22] - A[23] + A[28] + A[29] + A[33] + A[34];
    float termB146 = B[26] + B[27];
    float result146 = termA146 * termB146;
    C[9] -= result146; // c42
    C[15] += result146; // c43
    C[10] += result146; // c52
    C[16] -= result146; // c53
    C[11] -= result146; // c62
    C[17] += result146; // c63

    // Expression: (a42+a43-a52-a53+a62+a63)(b35+b36-b45-b46+b55+b56+b64+b65)(c53+c54)
    float termA147 = A[19] + A[20] - A[25] - A[26] + A[31] + A[32];
    float termB147 = B[16] + B[17] - B[22] - B[23] + B[28] + B[29] + B[33] + B[34];
    float result147 = termA147 * termB147;
    C[16] += result147; // c53
    C[22] += result147; // c54

    // Expression: (a23+a24)(b14+b15-b24-b25+b34+b35)(-c12+c13+c21+c22-c31-c32+c41+c42)
    float termA148 = A[8] + A[9];
    float termB148 = B[3] + B[4] - B[9] - B[10] + B[15] + B[16];
    float result148 = termA148 * termB148;
    C[6] -= result148; // c12
    C[12] += result148; // c13
    C[1] += result148; // c21
    C[7] += result148; // c22
    C[2] -= result148; // c31
    C[8] -= result148; // c32
    C[3] += result148; // c41
    C[9] += result148; // c42

    // Expression: (a12+a13-a21-a22-a31-a32+a41+a42)(b23+b24)(c14+c15-c24-c25+c34+c35)
    float termA149 = A[1] + A[2] - A[6] - A[7] - A[12] - A[13] + A[18] + A[19];
    float termB149 = B[8] + B[9];
    float result149 = termA149 * termB149;
    C[18] += result149; // c14
    C[24] += result149; // c15
    C[19] -= result149; // c24
    C[25] -= result149; // c25
    C[20] += result149; // c34
    C[26] += result149; // c35

    // Expression: (-a14-a15+a24+a25+a34+a35)(b12+b13+b21+b22-b31-b32+b41+b42)(c23+c24)
    float termA150 = -A[3] - A[4] + A[9] + A[10] + A[15] + A[16];
    float termB150 = B[1] + B[2] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19];
    float result150 = termA150 * termB150;
    C[13] += result150; // c23
    C[19] += result150; // c24

    // Expression: (a11+a22)(b11+b22)(c11+c22)
    float termA151 = A[0] + A[7];
    float termB151 = B[0] + B[7];
    float result151 = termA151 * termB151;
    C[0] += result151; // c11
    C[7] += result151; // c22

    // Expression: (a33+a44)(b33+b44)(c33+c44)
    float termA152 = A[14] + A[21];
    float termB152 = B[14] + B[21];
    float result152 = termA152 * termB152;
    C[14] += result152; // c33
    C[21] += result152; // c44

    // Expression: (a55+a66)(b55+b66)(c55+c66)
    float termA153 = A[28] + A[35];
    float termB153 = B[28] + B[35];
    float result153 = termA153 * termB153;
    C[28] += result153; // c55
    C[35] += result153; // c66
}
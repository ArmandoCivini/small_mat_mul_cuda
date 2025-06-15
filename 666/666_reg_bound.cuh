#pragma once
// A, B, and C are all length‐36 arrays in row‐major order (row*6 + col).
// This function must be inlined for best performance.

__device__ __forceinline__ void matmul6x6_opt_reg_bound(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C) {
    // Expression: (a12+a13-a22-a23-a32-a33+a42+a43-a52-a53)(-b36+b46)(c45+c46-c55-c56+c65+c66)
    float termA1 = A[1] + A[2] - A[7] - A[8] - A[13] - A[14] + A[19] + A[20] - A[25] - A[26];
    float termB1 = -B[17] + B[23];
    float result1 = termA1 * termB1;

    // Expression: (-a45-a46+a55+a56-a65-a66)(b12+b13+2*b14+2*b15-b22-b23-2*b24-2*b25+b32+b33+2*b34+2*b35-b42-b43+b52+b53)(-c36+c46)
    float termA2 = -A[22] - A[23] + A[28] + A[29] - A[34] - A[35];
    float termB2 = B[1] + B[2] + 2*B[3] + 2*B[4] - B[7] - B[8] - 2*B[9] - 2*B[10] + B[13] + B[14] + 2*B[15] + 2*B[16] - B[19] - B[20] + B[25] + B[26];
    float result2 = termA2 * termB2;

    // Expression: (a36-a46)(b45+b46-b55-b56+b65+b66)(c12-c13+c22-c23-c32+c33+c42-c43-c52+c53)
    float termA3 = A[17] - A[23];
    float termB3 = B[22] + B[23] - B[28] - B[29] + B[34] + B[35];
    float result3 = termA3 * termB3;

    // Expression: (a24+a25+a34+a35-a44-a45+a54+a55-a64-a65)(-b31+b41)(c11+c12-c21-c22+c31+c32)
    float termA4 = A[9] + A[10] + A[15] + A[16] - A[21] - A[22] + A[27] + A[28] - A[33] - A[34];
    float termB4 = -B[12] + B[18];
    float result4 = termA4 * termB4;

    // Expression: (-a11-a12+a21+a22+a31+a32)(-b24-b25+b34+b35-b44-b45+b54+b55+b64+b65)(-c31+c41)
    float termA5 = -A[0] - A[1] + A[6] + A[7] + A[12] + A[13];
    float termB5 = -B[9] - B[10] + B[15] + B[16] - B[21] - B[22] + B[27] + B[28] + B[33] + B[34];
    float result5 = termA5 * termB5;

    // Expression: (a31-a41)(b11+b12-b21-b22+b31+b32)(-c24-c25+c34+c35-c44-c45+c54+c55+c64+c65)
    float termA6 = A[12] - A[18];
    float termB6 = B[0] + B[1] - B[6] - B[7] + B[12] + B[13];
    float result6 = termA6 * termB6;

    // Expression: (-a64-a65)(b53+b54)(-c12+c13+c14-c16+c22-c23-c24+c26-c32+c33+c34-c36-c42+c43+c52-c53-c62+c63)
    float termA7 = -A[33] - A[34];
    float termB7 = B[26] + B[27];
    float result7 = termA7 * termB7;

    // Expression: (-2*a11-a12+a13+a14-a16+2*a21+a22-a23-a24+a26+2*a31+a32-a33-a34+a36-a42-a43+a52+a53-a62-a63)(b64+b65)(c53+c54)
    float termA8 = -2*A[0] - A[1] + A[2] + A[3] - A[5] + 2*A[6] + A[7] - A[8] - A[9] + A[11] + 2*A[12] + A[13] - A[14] - A[15] + A[17] - A[19] - A[20] + A[25] + A[26] - A[31] - A[32];
    float termB8 = B[33] + B[34];
    float result8 = termA8 * termB8;

    // Expression: (-a53-a54)(-2*b11-b12+b13+b14-b16+2*b21+b22-b23-b24+b26-2*b31-b32+b33+b34-b36+b42+b43-b52-b53+b62+b63)(c64+c65)
    float termA9 = -A[26] - A[27];
    float termB9 = -2*B[0] - B[1] + B[2] + B[3] - B[5] + 2*B[6] + B[7] - B[8] - B[9] + B[11] - 2*B[12] - B[13] + B[14] + B[15] - B[17] + B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result9 = termA9 * termB9;

    // Expression: (a12+a13)(b23+b24)(-c14-c15+c24+c25-c34-c35+c41+c43+c44+c45-c51-c53-c54-c55+c61+c63+c64+c65)
    float termA10 = A[1] + A[2];
    float termB10 = B[8] + B[9];
    float result10 = termA10 * termB10;

    // Expression: (a14+a15-a24-a25-a34-a35+a41-a43-a44-a45-a51+a53+a54+a55+a61-a63-a64-a65)(b12+b13)(c23+c24)
    float termA11 = A[3] + A[4] - A[9] - A[10] - A[15] - A[16] + A[18] - A[20] - A[21] - A[22] - A[24] + A[26] + A[27] + A[28] + A[30] - A[32] - A[33] - A[34];
    float termB11 = B[1] + B[2];
    float result11 = termA11 * termB11;

    // Expression: (-a23-a24)(b14+b15-b24-b25+b34+b35+b41-b43-b44+b45+2*b46-b51+b53+b54-b55-2*b56+b61-b63-b64+b65+2*b66)(-c12+c13)
    float termA12 = -A[8] - A[9];
    float termB12 = B[3] + B[4] - B[9] - B[10] + B[15] + B[16] + B[18] - B[20] - B[21] + B[22] + 2*B[23] - B[24] + B[26] + B[27] - B[28] - 2*B[29] + B[30] - B[32] - B[33] + B[34] + 2*B[35];
    float result12 = termA12 * termB12;

    // Expression: (a15+a16-a25-a26-a35-a36+a45+a46-a55-a56+a64+a65)(-b32+b42+b53+b54)(-c21-c23-c24+c26+c31)
    float termA13 = A[4] + A[5] - A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29] + A[33] + A[34];
    float termB13 = -B[13] + B[19] + B[26] + B[27];
    float result13 = termA13 * termB13;

    // Expression: (-a21+a23+a24-a26-a31)(-b15-b16+b25+b26-b35-b36+b45+b46-b55-b56+b64+b65)(c32-c42+c53+c54)
    float termA14 = -A[6] + A[8] + A[9] - A[11] - A[12];
    float termB14 = -B[4] - B[5] + B[10] + B[11] - B[16] - B[17] + B[22] + B[23] - B[28] - B[29] + B[33] + B[34];
    float result14 = termA14 * termB14;

    // Expression: (a32-a42-a53-a54)(-b21+b23+b24-b26+b31)(-c15-c16+c25+c26-c35-c36+c45+c46-c55-c56+c64+c65)
    float termA15 = A[13] - A[19] - A[26] - A[27];
    float termB15 = -B[6] + B[8] + B[9] - B[11] + B[12];
    float result15 = termA15 * termB15;

    // Expression: (a12+a13-a21-a22-a31-a32+a41+a42-a51-a52+a61+a62)(-b23-b24-b35+b45)(-c46-c51-c53-c54+c56)
    float termA16 = A[1] + A[2] - A[6] - A[7] - A[12] - A[13] + A[18] + A[19] - A[24] - A[25] + A[30] + A[31];
    float termB16 = -B[8] - B[9] - B[16] + B[22];
    float result16 = termA16 * termB16;

    // Expression: (-a46+a51-a53-a54+a56)(-b12-b13+b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(-c23-c24-c35+c45)
    float termA17 = -A[23] + A[24] - A[26] - A[27] + A[29];
    float termB17 = -B[1] - B[2] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result17 = termA17 * termB17;

    // Expression: (-a23-a24+a35-a45)(-b46+b51-b53-b54+b56)(-c12+c13+c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA18 = -A[8] - A[9] + A[16] - A[22];
    float termB18 = -B[23] + B[24] - B[26] - B[27] + B[29];
    float result18 = termA18 * termB18;

    // Expression: (-a33-a34-a35+a45)(-b46-b54+b56)(-c33+c43-c53-c54+c63+c64)
    float termA19 = -A[14] - A[15] - A[16] + A[22];
    float termB19 = -B[23] - B[27] + B[29];
    float result19 = termA19 * termB19;

    // Expression: (-a33+a43-a53-a54+a63+a64)(-b33-b34-b35+b45)(-c46-c54+c56)
    float termA20 = -A[14] + A[20] - A[26] - A[27] + A[32] + A[33];
    float termB20 = -B[14] - B[15] - B[16] + B[22];
    float result20 = termA20 * termB20;

    // Expression: (-a46-a54+a56)(-b33+b43-b53-b54+b63+b64)(-c33-c34-c35+c45)
    float termA21 = -A[23] - A[27] + A[29];
    float termB21 = -B[14] + B[20] - B[26] - B[27] + B[32] + B[33];
    float result21 = termA21 * termB21;

    // Expression: (a32-a42-a43-a44)(-b21+b23+b31)(-c13-c14+c23+c24-c34+c44)
    float termA22 = A[13] - A[19] - A[20] - A[21];
    float termB22 = -B[6] + B[8] + B[12];
    float result22 = termA22 * termB22;

    // Expression: (a13+a14-a23-a24-a34+a44)(-b32+b42+b43+b44)(-c21-c23+c31)
    float termA23 = A[2] + A[3] - A[8] - A[9] - A[15] + A[21];
    float termB23 = -B[13] + B[19] + B[20] + B[21];
    float result23 = termA23 * termB23;

    // Expression: (-a21+a23-a31)(-b13-b14+b23+b24-b34+b44)(c32-c42+c43+c44)
    float termA24 = -A[6] + A[8] - A[12];
    float termB24 = -B[2] - B[3] + B[8] + B[9] - B[15] + B[21];
    float result24 = termA24 * termB24;

    // Expression: (-a11+a23)(-b13-b14+b23+b24-b32-b33-b34+b42+b43+b44)(-c21-c22+c31)
    float termA25 = -A[0] + A[8];
    float termB25 = -B[2] - B[3] + B[8] + B[9] - B[13] - B[14] - B[15] + B[19] + B[20] + B[21];
    float result25 = termA25 * termB25;

    // Expression: (a21+a22+a31)(-b11+b23)(-c13-c14+c23+c24+c32-c33-c34-c42+c43+c44)
    float termA26 = A[6] + A[7] + A[12];
    float termB26 = -B[0] + B[8];
    float result26 = termA26 * termB26;

    // Expression: (a13+a14-a23-a24-a32-a33-a34+a42+a43+a44)(-b21-b22+b31)(c11+c23)
    float termA27 = A[2] + A[3] - A[8] - A[9] - A[13] - A[14] - A[15] + A[19] + A[20] + A[21];
    float termB27 = -B[6] - B[7] + B[12];
    float result27 = termA27 * termB27;

    // Expression: (a54-a66)(-b33-b34-b35+b43+b44+b45-b53-b54+b63+b64)(-c46+c55+c56)
    float termA28 = A[27] - A[35];
    float termB28 = -B[14] - B[15] - B[16] + B[20] + B[21] + B[22] - B[26] - B[27] + B[32] + B[33];
    float result28 = termA28 * termB28;

    // Expression: (a46-a55-a56)(-b54+b66)(-c33-c34-c35+c43+c44+c45-c53-c54+c63+c64)
    float termA29 = A[23] - A[28] - A[29];
    float termB29 = -B[27] + B[35];
    float result29 = termA29 * termB29;

    // Expression: (a33+a34+a35-a43-a44-a45+a53+a54-a63-a64)(-b46+b55+b56)(-c54+c66)
    float termA30 = A[14] + A[15] + A[16] - A[20] - A[21] - A[22] + A[26] + A[27] - A[32] - A[33];
    float termB30 = -B[23] + B[28] + B[29];
    float result30 = termA30 * termB30;

    // Expression: (-a21-a22)(b11+b43+b44)(-c12+c13+c22-c23-c32+c33)
    float termA31 = -A[6] - A[7];
    float termB31 = B[0] + B[20] + B[21];
    float result31 = termA31 * termB31;

    // Expression: (-a12-a13+a22+a23+a32+a33)(b21+b22)(-c11+c43+c44)
    float termA32 = -A[1] - A[2] + A[7] + A[8] + A[13] + A[14];
    float termB32 = B[6] + B[7];
    float result32 = termA32 * termB32;

    // Expression: (-a11-a43-a44)(b12+b13-b22-b23+b32+b33)(c21+c22-2*c34+2*c44)
    float termA33 = -A[0] - A[20] - A[21];
    float termB33 = B[1] + B[2] - B[7] - B[8] + B[13] + B[14];
    float result33 = termA33 * termB33;

    // Expression: (a55+a56)(-b33-b34+b66)(c44+c45-c54-c55+c64+c65)
    float termA34 = A[28] + A[29];
    float termB34 = -B[14] - B[15] + B[35];
    float result34 = termA34 * termB34;

    // Expression: (a44+a45-a54-a55+a64+a65)(b55+b56)(-c33-c34+c66)
    float termA35 = A[21] + A[22] - A[27] - A[28] + A[33] + A[34];
    float termB35 = B[28] + B[29];
    float result35 = termA35 * termB35;

    // Expression: (-a33-a34+a66)(b44+b45-b54-b55+b64+b65)(c55+c56)
    float termA36 = -A[14] - A[15] + A[35];
    float termB36 = B[21] + B[22] - B[27] - B[28] + B[33] + B[34];
    float result36 = termA36 * termB36;

    // Expression: (a56-a66)(b65+b66)c55
    float termA37 = A[29] - A[35];
    float termB37 = B[34] + B[35];
    float result37 = termA37 * termB37;

    // Expression: -(a55(-b56+b66)(c65+c66))
    float termA38 = A[28];
    float termB38 = -B[29] + B[35];
    float result38 = -(termA38 * termB38);

    // Expression: (-a65-a66)b55(-c56+c66)
    float termA39 = -A[34] - A[35];
    float termB39 = B[28];
    float result39 = termA39 * termB39;

    // Expression: (-a11+a21)(b11+b12)c22
    float termA40 = -A[0] + A[6];
    float termB40 = B[0] + B[1];
    float result40 = termA40 * termB40;

    // Expression: a22(-b11+b21)(c11+c12)
    float termA41 = A[7];
    float termB41 = -B[0] + B[6];
    float result41 = termA41 * termB41;

    // Expression: (a11+a12)b22(-c11+c21)
    float termA42 = A[0] + A[1];
    float termB42 = B[7];
    float result42 = termA42 * termB42;

    // Expression: (-a12-a13-a14-a15+a21+a22+a31+a32-a41-a42+a51+a52-a61-a62)(-b26+b36-b46+b55+b56)(-c51-c53-c54+c66)
    float termA43 = -A[1] - A[2] - A[3] - A[4] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25] - A[30] - A[31];
    float termB43 = -B[11] + B[17] - B[23] + B[28] + B[29];
    float result43 = termA43 * termB43;

    // Expression: (-a51+a53+a54-a66)(-b12-b13-b14-b15+b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(-c26+c36-c46+c55+c56)
    float termA44 = -A[24] + A[26] + A[27] - A[35];
    float termB44 = -B[1] - B[2] - B[3] - B[4] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result44 = termA44 * termB44;

    // Expression: (a26+a36-a46+a55+a56)(b51-b53-b54+b66)(-c12+c13+c14+c15+c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA45 = A[11] + A[17] - A[23] + A[28] + A[29];
    float termB45 = B[24] - B[26] - B[27] + B[35];
    float result45 = termA45 * termB45;

    // Expression: (-a15-a16+a25+a26+a35+a36-a45-a46+a55+a56-a62-a63-a64-a65)(-b21-b22+b31-b41+b51)(-c11-c23-c24+c26)
    float termA46 = -A[4] - A[5] + A[10] + A[11] + A[16] + A[17] - A[22] - A[23] + A[28] + A[29] - A[31] - A[32] - A[33] - A[34];
    float termB46 = -B[6] - B[7] + B[12] - B[18] + B[24];
    float result46 = termA46 * termB46;

    // Expression: (-a11+a23+a24-a26)(-b15-b16+b25+b26-b35-b36+b45+b46-b55-b56+b62+b63+b64+b65)(-c21-c22+c31-c41+c51)
    float termA47 = -A[0] + A[8] + A[9] - A[11];
    float termB47 = -B[4] - B[5] + B[10] + B[11] - B[16] - B[17] + B[22] + B[23] - B[28] - B[29] + B[31] + B[32] + B[33] + B[34];
    float result47 = termA47 * termB47;

    // Expression: (-a21-a22-a31+a41-a51)(b11-b23-b24+b26)(-c15-c16+c25+c26-c35-c36+c45+c46-c55-c56-c62+c63+c64+c65)
    float termA48 = -A[6] - A[7] - A[12] + A[18] - A[24];
    float termB48 = B[0] - B[8] - B[9] + B[11];
    float result48 = termA48 * termB48;

    // Expression: (a31-a41+a53+a54-a63-a64)(b12+b13+b14+b15-b21-b22+b31+b32)(-c26+c36-c46-c54+c56)
    float termA49 = A[12] - A[18] + A[26] + A[27] - A[32] - A[33];
    float termB49 = B[1] + B[2] + B[3] + B[4] - B[6] - B[7] + B[12] + B[13];
    float result49 = termA49 * termB49;

    // Expression: (a26+a36-a46-a54+a56)(-b31+b41-b53-b54+b63+b64)(c12-c13-c14-c15-c21-c22+c31+c32)
    float termA50 = A[11] + A[17] - A[23] - A[27] + A[29];
    float termB50 = -B[12] + B[18] - B[26] - B[27] + B[32] + B[33];
    float result50 = termA50 * termB50;

    // Expression: (a12+a13+a14+a15-a21-a22-a31-a32)(-b26+b36-b46-b54+b56)(c31-c41-c53-c54+c63+c64)
    float termA51 = A[1] + A[2] + A[3] + A[4] - A[6] - A[7] - A[12] - A[13];
    float termB51 = -B[11] + B[17] - B[23] - B[27] + B[29];
    float result51 = termA51 * termB51;

    // Expression: (a13+a14-a23-a24-a36+a46)(b45+b46-b55-b56+b62+b63+b64+b65)(-c21-c23+c31-c41+c51)
    float termA52 = A[2] + A[3] - A[8] - A[9] - A[17] + A[23];
    float termB52 = B[22] + B[23] - B[28] - B[29] + B[31] + B[32] + B[33] + B[34];
    float result52 = termA52 * termB52;

    // Expression: (-a21+a23-a31+a41-a51)(-b13-b14+b23+b24-b36+b46)(c45+c46-c55-c56-c62+c63+c64+c65)
    float termA53 = -A[6] + A[8] - A[12] + A[18] - A[24];
    float termB53 = -B[2] - B[3] + B[8] + B[9] - B[17] + B[23];
    float result53 = termA53 * termB53;

    // Expression: (-a45-a46+a55+a56-a62-a63-a64-a65)(-b21+b23+b31-b41+b51)(-c13-c14+c23+c24-c36+c46)
    float termA54 = -A[22] - A[23] + A[28] + A[29] - A[31] - A[32] - A[33] - A[34];
    float termB54 = -B[6] + B[8] + B[12] - B[18] + B[24];
    float result54 = termA54 * termB54;

    // Expression: (a21-a23+a31-a41)(-b13-b14+b23+b24)(-c42+c43+c44+c45+c52-c53-c54-c55-c62+c63+c64+c65)
    float termA55 = A[6] - A[8] + A[12] - A[18];
    float termB55 = -B[2] - B[3] + B[8] + B[9];
    float result55 = termA55 * termB55;

    // Expression: (-a42-a43-a44-a45+a52+a53+a54+a55-a62-a63-a64-a65)(b21-b23-b31+b41)(-c13-c14+c23+c24)
    float termA56 = -A[19] - A[20] - A[21] - A[22] + A[25] + A[26] + A[27] + A[28] - A[31] - A[32] - A[33] - A[34];
    float termB56 = B[6] - B[8] - B[12] + B[18];
    float result56 = termA56 * termB56;

    // Expression: (a13+a14-a23-a24)(b42+b43+b44+b45-b52-b53-b54-b55+b62+b63+b64+b65)(c21+c23-c31+c41)
    float termA57 = A[2] + A[3] - A[8] - A[9];
    float termB57 = B[19] + B[20] + B[21] + B[22] - B[25] - B[26] - B[27] - B[28] + B[31] + B[32] + B[33] + B[34];
    float result57 = termA57 * termB57;

    // Expression: (a36-a46-a54+a56)(-b53-b54+b63+b64)(-c12+c13+c14+c15+c22-c23-c24-c25-c32+c33+c34+c35)
    float termA58 = A[17] - A[23] - A[27] + A[29];
    float termB58 = -B[26] - B[27] + B[32] + B[33];
    float result58 = termA58 * termB58;

    // Expression: (-a12-a13-a14-a15+a22+a23+a24+a25+a32+a33+a34+a35)(b36-b46-b54+b56)(-c53-c54+c63+c64)
    float termA59 = -A[1] - A[2] - A[3] - A[4] + A[7] + A[8] + A[9] + A[10] + A[13] + A[14] + A[15] + A[16];
    float termB59 = B[17] - B[23] - B[27] + B[29];
    float result59 = termA59 * termB59;

    // Expression: (-a53-a54+a63+a64)(b12+b13+b14+b15-b22-b23-b24-b25+b32+b33+b34+b35)(c36-c46-c54+c56)
    float termA60 = -A[26] - A[27] + A[32] + A[33];
    float termB60 = B[1] + B[2] + B[3] + B[4] - B[7] - B[8] - B[9] - B[10] + B[13] + B[14] + B[15] + B[16];
    float result60 = termA60 * termB60;

    // Expression: (a26+a36-a46-a52-a53-a54+a56)(-b31+b41-b51+b61)(c11+c13+c14+c15)
    float termA61 = A[11] + A[17] - A[23] - A[25] - A[26] - A[27] + A[29];
    float termB61 = -B[12] + B[18] - B[24] + B[30];
    float result61 = termA61 * termB61;

    // Expression: (-a11+a13+a14+a15)(-b26+b36-b46-b52-b53-b54+b56)(-c31+c41-c51+c61)
    float termA62 = -A[0] + A[2] + A[3] + A[4];
    float termB62 = -B[11] + B[17] - B[23] - B[25] - B[26] - B[27] + B[29];
    float result62 = termA62 * termB62;

    // Expression: (-a31+a41-a51+a61)(-b11+b13+b14+b15)(-c26+c36-c46+c52-c53-c54+c56)
    float termA63 = -A[12] + A[18] - A[24] + A[30];
    float termB63 = -B[0] + B[2] + B[3] + B[4];
    float result63 = termA63 * termB63;

    // Expression: (-a21+a23+a24+a25-a31+a41-a51)(-b16+b26-b36+b46)(c62-c63-c64+c66)
    float termA64 = -A[6] + A[8] + A[9] + A[10] - A[12] + A[18] - A[24];
    float termB64 = -B[5] + B[11] - B[17] + B[23];
    float result64 = termA64 * termB64;

    // Expression: (a62+a63+a64-a66)(-b21+b23+b24+b25+b31-b41+b51)(-c16+c26-c36+c46)
    float termA65 = A[31] + A[32] + A[33] - A[35];
    float termB65 = -B[6] + B[8] + B[9] + B[10] + B[12] - B[18] + B[24];
    float result65 = termA65 * termB65;

    // Expression: (a16-a26-a36+a46)(-b62-b63-b64+b66)(-c21-c23-c24-c25+c31-c41+c51)
    float termA66 = A[5] - A[11] - A[17] + A[23];
    float termB66 = -B[31] - B[32] - B[33] + B[35];
    float result66 = termA66 * termB66;

    // Expression: (-a21-a22-a31+a41)(-b11+b23+b24)(-c14-c15+c24+c25-c34-c35-c42+c43+c44+c45+c52-c53-c54-c55-c62+c63+c64+c65)
    float termA67 = -A[6] - A[7] - A[12] + A[18];
    float termB67 = -B[0] + B[8] + B[9];
    float result67 = termA67 * termB67;

    // Expression: (a14+a15-a24-a25-a34-a35+a42+a43+a44+a45-a52-a53-a54-a55+a62+a63+a64+a65)(b21+b22-b31+b41)(c11+c23+c24)
    float termA68 = A[3] + A[4] - A[9] - A[10] - A[15] - A[16] + A[19] + A[20] + A[21] + A[22] - A[25] - A[26] - A[27] - A[28] + A[31] + A[32] + A[33] + A[34];
    float termB68 = B[6] + B[7] - B[12] + B[18];
    float result68 = termA68 * termB68;

    // Expression: (-a11+a23+a24)(-b14-b15+b24+b25-b34-b35+b42+b43+b44+b45-b52-b53-b54-b55+b62+b63+b64+b65)(c21+c22-c31+c41)
    float termA69 = -A[0] + A[8] + A[9];
    float termB69 = -B[3] - B[4] + B[9] + B[10] - B[15] - B[16] + B[19] + B[20] + B[21] + B[22] - B[25] - B[26] - B[27] - B[28] + B[31] + B[32] + B[33] + B[34];
    float result69 = termA69 * termB69;

    // Expression: (a36-a46+a55+a56)(-b53-b54+b66)(c12-c13-c14-c15-c22+c23+c24+c25+c32-c33-c34-c35-c42+c43+c52-c53-c62+c63)
    float termA70 = A[17] - A[23] + A[28] + A[29];
    float termB70 = -B[26] - B[27] + B[35];
    float result70 = termA70 * termB70;

    // Expression: (a12+a13+a14+a15-a22-a23-a24-a25-a32-a33-a34-a35+a42+a43-a52-a53+a62+a63)(b36-b46+b55+b56)(-c53-c54+c66)
    float termA71 = A[1] + A[2] + A[3] + A[4] - A[7] - A[8] - A[9] - A[10] - A[13] - A[14] - A[15] - A[16] + A[19] + A[20] - A[25] - A[26] + A[31] + A[32];
    float termB71 = B[17] - B[23] + B[28] + B[29];
    float result71 = termA71 * termB71;

    // Expression: (-a53-a54+a66)(-b12-b13-b14-b15+b22+b23+b24+b25-b32-b33-b34-b35+b42+b43-b52-b53+b62+b63)(c36-c46+c55+c56)
    float termA72 = -A[26] - A[27] + A[35];
    float termB72 = -B[1] - B[2] - B[3] - B[4] + B[7] + B[8] + B[9] + B[10] - B[13] - B[14] - B[15] - B[16] + B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result72 = termA72 * termB72;

    // Expression: -(a31(-b11+b13)(c23+c24+c32-c33-c34-c42+c43+c44))
    float termA73 = A[12];
    float termB73 = -B[0] + B[2];
    float result73 = -(termA73 * termB73);

    // Expression: (a23+a24+a32+a33+a34-a42-a43-a44)b31(c11+c13)
    float termA74 = A[8] + A[9] + A[13] + A[14] + A[15] - A[19] - A[20] - A[21];
    float termB74 = B[12];
    float result74 = termA74 * termB74;

    // Expression: (a11-a13)(b23+b24-b32-b33-b34+b42+b43+b44)c31
    float termA75 = A[0] - A[2];
    float termB75 = B[8] + B[9] - B[13] - B[14] - B[15] + B[19] + B[20] + B[21];
    float result75 = termA75 * termB75;

    // Expression: a46(-b64+b66)(c33+c34+c35-c43-c44-c45+c53+c54)
    float termA76 = A[23];
    float termB76 = -B[33] + B[35];
    float result76 = termA76 * termB76;

    // Expression: (a33+a34+a35-a43-a44-a45+a53+a54)b46(-c64+c66)
    float termA77 = A[14] + A[15] + A[16] - A[20] - A[21] - A[22] + A[26] + A[27];
    float termB77 = B[23];
    float result77 = termA77 * termB77;

    // Expression: (-a64+a66)(b33+b34+b35-b43-b44-b45+b53+b54)c46
    float termA78 = -A[33] + A[35];
    float termB78 = B[14] + B[15] + B[16] - B[20] - B[21] - B[22] + B[26] + B[27];
    float result78 = termA78 * termB78;

    // Expression: (a21+a22+a31+a32-a41-a42+a51+a52)(-b23-b24+b26)(-c15-c16+c25+c26-c35-c36+c45+c46-c55-c56+c61+c63+c64+c65)
    float termA79 = A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25];
    float termB79 = -B[8] - B[9] + B[11];
    float result79 = termA79 * termB79;

    // Expression: (a15+a16-a25-a26-a35-a36+a45+a46-a55-a56-a61+a63+a64+a65)(-b21-b22+b31+b32-b41-b42+b51+b52)(-c23-c24+c26)
    float termA80 = A[4] + A[5] - A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29] - A[30] + A[32] + A[33] + A[34];
    float termB80 = -B[6] - B[7] + B[12] + B[13] - B[18] - B[19] + B[24] + B[25];
    float result80 = termA80 * termB80;

    // Expression: (-a23-a24+a26)(-b15-b16+b25+b26-b35-b36+b45+b46-b55-b56+b61-b63-b64+b65+2*b66)(-c21-c22+c31+c32-c41-c42+c51+c52)
    float termA81 = -A[8] - A[9] + A[11];
    float termB81 = -B[4] - B[5] + B[10] + B[11] - B[16] - B[17] + B[22] + B[23] - B[28] - B[29] + B[30] - B[32] - B[33] + B[34] + 2*B[35];
    float result81 = termA81 * termB81;

    // Expression: (a25+a26+a35+a36-a45-a46+a55+a56)(-b51+b53+b54)(-c12+c13+c14-c16+c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA82 = A[10] + A[11] + A[16] + A[17] - A[22] - A[23] + A[28] + A[29];
    float termB82 = -B[24] + B[26] + B[27];
    float result82 = termA82 * termB82;

    // Expression: (-2*a11-a12+a13+a14-a16+a21+a22+a31+a32-a41-a42+a51+a52-a61-a62)(-b25-b26+b35+b36-b45-b46+b55+b56)(c51+c53+c54)
    float termA83 = -2*A[0] - A[1] + A[2] + A[3] - A[5] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25] - A[30] - A[31];
    float termB83 = -B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29];
    float result83 = termA83 * termB83;

    // Expression: (a51-a53-a54)(-2*b11-b12+b13+b14-b16+b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(-c25-c26+c35+c36-c45-c46+c55+c56)
    float termA84 = A[24] - A[26] - A[27];
    float termB84 = -2*B[0] - B[1] + B[2] + B[3] - B[5] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result84 = termA84 * termB84;

    // Expression: (-a21+a23+a24+a25-a31+a41-a51+a61)(b16-b26+b36-b46+b55+b56)(c52-c53-c54+c66)
    float termA85 = -A[6] + A[8] + A[9] + A[10] - A[12] + A[18] - A[24] + A[30];
    float termB85 = B[5] - B[11] + B[17] - B[23] + B[28] + B[29];
    float result85 = termA85 * termB85;

    // Expression: (-a52-a53-a54+a66)(b21-b23-b24-b25-b31+b41-b51+b61)(c16-c26+c36-c46+c55+c56)
    float termA86 = -A[25] - A[26] - A[27] + A[35];
    float termB86 = B[6] - B[8] - B[9] - B[10] - B[12] + B[18] - B[24] + B[30];
    float result86 = termA86 * termB86;

    // Expression: (a16-a26-a36+a46-a55-a56)(-b52-b53-b54+b66)(c21+c23+c24+c25-c31+c41-c51+c61)
    float termA87 = A[5] - A[11] - A[17] + A[23] - A[28] - A[29];
    float termB87 = -B[25] - B[26] - B[27] + B[35];
    float result87 = termA87 * termB87;

    // Expression: (a16-a26-a36+a46+a52+a53+a54-a56)(b21+b22-b31+b41-b51+b61)(c11+c23+c24+c25)
    float termA88 = A[5] - A[11] - A[17] + A[23] + A[25] + A[26] + A[27] - A[29];
    float termB88 = B[6] + B[7] - B[12] + B[18] - B[24] + B[30];
    float result88 = termA88 * termB88;

    // Expression: (a11-a23-a24-a25)(b16-b26+b36-b46-b52-b53-b54+b56)(c21+c22-c31+c41-c51+c61)
    float termA89 = A[0] - A[8] - A[9] - A[10];
    float termB89 = B[5] - B[11] + B[17] - B[23] - B[25] - B[26] - B[27] + B[29];
    float result89 = termA89 * termB89;

    // Expression: (a21+a22+a31-a41+a51-a61)(-b11+b23+b24+b25)(c16-c26+c36-c46+c52-c53-c54+c56)
    float termA90 = A[6] + A[7] + A[12] - A[18] + A[24] - A[30];
    float termB90 = -B[0] + B[8] + B[9] + B[10];
    float result90 = termA90 * termB90;

    // Expression: (a46+a53+a54-a56)(-b12-b13+b22+b23-b32-b33+b42+b43-b52-b53+b62+b63)(-c35+c45)
    float termA91 = A[23] + A[26] + A[27] - A[29];
    float termB91 = -B[1] - B[2] + B[7] + B[8] - B[13] - B[14] + B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result91 = termA91 * termB91;

    // Expression: (a35-a45)(-b46-b53-b54+b56)(c12-c13-c22+c23+c32-c33-c42+c43+c52-c53-c62+c63)
    float termA92 = A[16] - A[22];
    float termB92 = -B[23] - B[26] - B[27] + B[29];
    float result92 = termA92 * termB92;

    // Expression: (-a12-a13+a22+a23+a32+a33-a42-a43+a52+a53-a62-a63)(-b35+b45)(-c46-c53-c54+c56)
    float termA93 = -A[1] - A[2] + A[7] + A[8] + A[13] + A[14] - A[19] - A[20] + A[25] + A[26] - A[31] - A[32];
    float termB93 = -B[16] + B[22];
    float result93 = termA93 * termB93;

    // Expression: (-a21+a23+a24-a31)(-b14-b15+b24+b25-b34-b35+b44+b45-b54-b55+b64+b65)(-c32+c42)
    float termA94 = -A[6] + A[8] + A[9] - A[12];
    float termB94 = -B[3] - B[4] + B[9] + B[10] - B[15] - B[16] + B[21] + B[22] - B[27] - B[28] + B[33] + B[34];
    float result94 = termA94 * termB94;

    // Expression: (-a32+a42)(-b21+b23+b24+b31)(-c14-c15+c24+c25-c34-c35+c44+c45-c54-c55+c64+c65)
    float termA95 = -A[13] + A[19];
    float termB95 = -B[6] + B[8] + B[9] + B[12];
    float result95 = termA95 * termB95;

    // Expression: (-a14-a15+a24+a25+a34+a35-a44-a45+a54+a55-a64-a65)(-b32+b42)(-c21-c23-c24+c31)
    float termA96 = -A[3] - A[4] + A[9] + A[10] + A[15] + A[16] - A[21] - A[22] + A[27] + A[28] - A[33] - A[34];
    float termB96 = -B[13] + B[19];
    float result96 = termA96 * termB96;

    // Expression: -(a54(-b33-b34+b36+b43+b44-b46-b53-b54+b63+b64)(-c45-c46+c55+c56))
    float termA97 = A[27];
    float termB97 = -B[14] - B[15] + B[17] + B[20] + B[21] - B[23] - B[26] - B[27] + B[32] + B[33];
    float result97 = -(termA97 * termB97);

    // Expression: (a45+a46-a55-a56)b54(-c33-c34+c36+c43+c44-c46-c53-c54+c63+c64)
    float termA98 = A[22] + A[23] - A[28] - A[29];
    float termB98 = B[27];
    float result98 = termA98 * termB98;

    // Expression: (a33+a34-a36-a43-a44+a46+a53+a54-a63-a64)(-b45-b46+b55+b56)c54
    float termA99 = A[14] + A[15] - A[17] - A[20] - A[21] + A[23] + A[26] + A[27] - A[32] - A[33];
    float termB99 = -B[22] - B[23] + B[28] + B[29];
    float result99 = termA99 * termB99;

    // Expression: -(a23(-b13-b14+b23+b24+b31-b33-b34-b41+b43+b44)(-c21-c22+c31+c32))
    float termA100 = A[8];
    float termB100 = -B[2] - B[3] + B[8] + B[9] + B[12] - B[14] - B[15] - B[18] + B[20] + B[21];
    float result100 = -(termA100 * termB100);

    // Expression: (-a21-a22-a31-a32)b23(-c13-c14+c23+c24-c31-c33-c34+c41+c43+c44)
    float termA101 = -A[6] - A[7] - A[12] - A[13];
    float termB101 = B[8];
    float result101 = termA101 * termB101;

    // Expression: (-a13-a14+a23+a24-a31+a33+a34+a41-a43-a44)(-b21-b22+b31+b32)c23
    float termA102 = -A[2] - A[3] + A[8] + A[9] - A[12] + A[14] + A[15] + A[18] - A[20] - A[21];
    float termB102 = -B[6] - B[7] + B[12] + B[13];
    float result102 = termA102 * termB102;

    // Expression: (-a42-a43+a52+a53-a62-a63)(-b31+b41)(c11+c13+c14)
    float termA103 = -A[19] - A[20] + A[25] + A[26] - A[31] - A[32];
    float termB103 = -B[12] + B[18];
    float result103 = termA103 * termB103;

    // Expression: (a11-a13-a14)(b42+b43-b52-b53+b62+b63)(-c31+c41)
    float termA104 = A[0] - A[2] - A[3];
    float termB104 = B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result104 = termA104 * termB104;

    // Expression: (a31-a41)(-b11+b13+b14)(-c42+c43+c52-c53-c62+c63)
    float termA105 = A[12] - A[18];
    float termB105 = -B[0] + B[2] + B[3];
    float result105 = termA105 * termB105;

    // Expression: (a14+a15-a24-a25-a34-a35)(-b36+b46)(-c63-c64+c66)
    float termA106 = A[3] + A[4] - A[9] - A[10] - A[15] - A[16];
    float termB106 = -B[17] + B[23];
    float result106 = termA106 * termB106;

    // Expression: (2*a45+2*a46-2*a55-2*a56+a63+a64+2*a65+a66)(b14+b15-b24-b25+b34+b35)(-c36+c46)
    float termA107 = 2*A[22] + 2*A[23] - 2*A[28] - 2*A[29] + A[32] + A[33] + 2*A[34] + A[35];
    float termB107 = B[3] + B[4] - B[9] - B[10] + B[15] + B[16];
    float result107 = termA107 * termB107;

    // Expression: (a36-a46)(-b63-b64+b66)(c14+c15-c24-c25+c34+c35)
    float termA108 = A[17] - A[23];
    float termB108 = -B[32] - B[33] + B[35];
    float result108 = termA108 * termB108;

    // Expression: (a31-a41+a51)(b11-b13-b14+b16)(-c25-c26+c35+c36-c45-c46+c55+c56-c62+c63+c64+c65)
    float termA109 = A[12] - A[18] + A[24];
    float termB109 = B[0] - B[2] - B[3] + B[5];
    float result109 = termA109 * termB109;

    // Expression: (-a25-a26-a35-a36+a45+a46-a55-a56+a62+a63+a64+a65)(b31-b41+b51)(-c11-c13-c14+c16)
    float termA110 = -A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29] + A[31] + A[32] + A[33] + A[34];
    float termB110 = B[12] - B[18] + B[24];
    float result110 = termA110 * termB110;

    // Expression: (a11-a13-a14+a16)(-b25-b26+b35+b36-b45-b46+b55+b56+b62+b63+b64+b65)(c31-c41+c51)
    float termA111 = A[0] - A[2] - A[3] + A[5];
    float termB111 = -B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29] + B[31] + B[32] + B[33] + B[34];
    float result111 = termA111 * termB111;

    // Expression: (-a26-a36+a46)(b61-b63-b64+b66)(-c12+c13+c14+c15-c21-c22+c31+c32-c41-c42+c51+c52)
    float termA112 = -A[11] - A[17] + A[23];
    float termB112 = B[30] - B[32] - B[33] + B[35];
    float result112 = termA112 * termB112;

    // Expression: (-a12-a13-a14-a15+a21+a22+a31+a32-a41-a42+a51+a52)(b26-b36+b46)(-c61-c63-c64+c66)
    float termA113 = -A[1] - A[2] - A[3] - A[4] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25];
    float termB113 = B[11] - B[17] + B[23];
    float result113 = termA113 * termB113;

    // Expression: (a61-a63-a64+a66)(b12+b13+b14+b15-b21-b22+b31+b32-b41-b42+b51+b52)(c26-c36+c46)
    float termA114 = A[30] - A[32] - A[33] + A[35];
    float termB114 = B[1] + B[2] + B[3] + B[4] - B[6] - B[7] + B[12] + B[13] - B[18] - B[19] + B[24] + B[25];
    float result114 = termA114 * termB114;

    // Expression: (a25+a26+a35+a36-a45-a46+a55+a56-a64-a65)(b31-b41+b53+b54)(c12-c13-c14+c16-c21-c22+c31+c32)
    float termA115 = A[10] + A[11] + A[16] + A[17] - A[22] - A[23] + A[28] + A[29] - A[33] - A[34];
    float termB115 = B[12] - B[18] + B[26] + B[27];
    float result115 = termA115 * termB115;

    // Expression: (2*a11+a12-a13-a14+a16-a21-a22-a31-a32)(-b25-b26+b35+b36-b45-b46+b55+b56+b64+b65)(-c31+c41+c53+c54)
    float termA116 = 2*A[0] + A[1] - A[2] - A[3] + A[5] - A[6] - A[7] - A[12] - A[13];
    float termB116 = -B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29] + B[33] + B[34];
    float result116 = termA116 * termB116;

    // Expression: (-a31+a41-a53-a54)(2*b11+b12-b13-b14+b16-b21-b22+b31+b32)(-c25-c26+c35+c36-c45-c46+c55+c56+c64+c65)
    float termA117 = -A[12] + A[18] - A[26] - A[27];
    float termB117 = 2*B[0] + B[1] - B[2] - B[3] + B[5] - B[6] - B[7] + B[12] + B[13];
    float result117 = termA117 * termB117;

    // Expression: (-a12-a13+a21+a22+a31+a32-a41-a42+a51+a52)(b23+b24-b36+b46)(c45+c46-c55-c56+c61+c63+c64+c65)
    float termA118 = -A[1] - A[2] + A[6] + A[7] + A[12] + A[13] - A[18] - A[19] + A[24] + A[25];
    float termB118 = B[8] + B[9] - B[17] + B[23];
    float result118 = termA118 * termB118;

    // Expression: (a45+a46-a55-a56-a61+a63+a64+a65)(b12+b13-b21-b22+b31+b32-b41-b42+b51+b52)(c23+c24-c36+c46)
    float termA119 = A[22] + A[23] - A[28] - A[29] - A[30] + A[32] + A[33] + A[34];
    float termB119 = B[1] + B[2] - B[6] - B[7] + B[12] + B[13] - B[18] - B[19] + B[24] + B[25];
    float result119 = termA119 * termB119;

    // Expression: (a23+a24+a36-a46)(b45+b46-b55-b56+b61-b63-b64+b65+2*b66)(-c12+c13-c21-c22+c31+c32-c41-c42+c51+c52)
    float termA120 = A[8] + A[9] + A[17] - A[23];
    float termB120 = B[22] + B[23] - B[28] - B[29] + B[30] - B[32] - B[33] + B[34] + 2*B[35];
    float result120 = termA120 * termB120;

    // Expression: (a45+a46-a55-a56+a64+a65)(b33-b43+b53+b54)(c33+c34-c36+c46)
    float termA121 = A[22] + A[23] - A[28] - A[29] + A[33] + A[34];
    float termB121 = B[14] - B[20] + B[26] + B[27];
    float result121 = termA121 * termB121;

    // Expression: (a33+a34-a36+a46)(b45+b46-b55-b56+b64+b65)(c33-c43+c53+c54)
    float termA122 = A[14] + A[15] - A[17] + A[23];
    float termB122 = B[22] + B[23] - B[28] - B[29] + B[33] + B[34];
    float result122 = termA122 * termB122;

    // Expression: (a33-a43+a53+a54)(b33+b34-b36+b46)(c45+c46-c55-c56+c64+c65)
    float termA123 = A[14] - A[20] + A[26] + A[27];
    float termB123 = B[14] + B[15] - B[17] + B[23];
    float result123 = termA123 * termB123;

    // Expression: (-a12-a13+a21+a22+a31+a32)(b23+b24-b34+b44)(-c31+c41+c43+c44)
    float termA124 = -A[1] - A[2] + A[6] + A[7] + A[12] + A[13];
    float termB124 = B[8] + B[9] - B[15] + B[21];
    float result124 = termA124 * termB124;

    // Expression: (a31-a41+a43+a44)(b12+b13-b21-b22+b31+b32)(c23+c24-c34+c44)
    float termA125 = A[12] - A[18] + A[20] + A[21];
    float termB125 = B[1] + B[2] - B[6] - B[7] + B[12] + B[13];
    float result125 = termA125 * termB125;

    // Expression: (a23+a24+a34-a44)(b31-b41+b43+b44)(c12-c13-c21-c22+c31+c32)
    float termA126 = A[8] + A[9] + A[15] - A[21];
    float termB126 = B[12] - B[18] + B[20] + B[21];
    float result126 = termA126 * termB126;

    // Expression: (2*a11+a43+a44)(b12+b13-b22-b23+b32+b33)(c21+c22-c34+c44)
    float termA127 = 2*A[0] + A[20] + A[21];
    float termB127 = B[1] + B[2] - B[7] - B[8] + B[13] + B[14];
    float result127 = termA127 * termB127;

    // Expression: (a21+a22+a34-a44)(b43+b44)(-c12+c13+c22-c23-c32+c33)
    float termA128 = A[6] + A[7] + A[15] - A[21];
    float termB128 = B[20] + B[21];
    float result128 = termA128 * termB128;

    // Expression: (a12+a13-a22-a23-a32-a33)(b21+b22-b34+b44)(c43+c44)
    float termA129 = A[1] + A[2] - A[7] - A[8] - A[13] - A[14];
    float termB129 = B[6] + B[7] - B[15] + B[21];
    float result129 = termA129 * termB129;

    // Expression: (a33+a34)(b44+b45-b54-b55+b64+b65)(-c33+c43+c55+c56)
    float termA130 = A[14] + A[15];
    float termB130 = B[21] + B[22] - B[27] - B[28] + B[33] + B[34];
    float result130 = termA130 * termB130;

    // Expression: (-a33+a43+a55+a56)(b33+b34)(c44+c45-c54-c55+c64+c65)
    float termA131 = -A[14] + A[20] + A[28] + A[29];
    float termB131 = B[14] + B[15];
    float result131 = termA131 * termB131;

    // Expression: (a44+a45-a54-a55+a64+a65)(-b33+b43+b55+b56)(c33+c34)
    float termA132 = A[21] + A[22] - A[27] - A[28] + A[33] + A[34];
    float termB132 = -B[14] + B[20] + B[28] + B[29];
    float result132 = termA132 * termB132;

    // Expression: (a32-a42-a53-a54+a63+a64)(-b21+b23+b24+b25+b31)(c16-c26+c36-c46-c54+c56)
    float termA133 = A[13] - A[19] - A[26] - A[27] + A[32] + A[33];
    float termB133 = -B[6] + B[8] + B[9] + B[10] + B[12];
    float result133 = termA133 * termB133;

    // Expression: (a16-a26-a36+a46+a54-a56)(b32-b42-b53-b54+b63+b64)(-c21-c23-c24-c25+c31)
    float termA134 = A[5] - A[11] - A[17] + A[23] + A[27] - A[29];
    float termB134 = B[13] - B[19] - B[26] - B[27] + B[32] + B[33];
    float result134 = termA134 * termB134;

    // Expression: (a21-a23-a24-a25+a31)(b16-b26+b36-b46-b54+b56)(-c32+c42-c53-c54+c63+c64)
    float termA135 = A[6] - A[8] - A[9] - A[10] + A[12];
    float termB135 = B[5] - B[11] + B[17] - B[23] - B[27] + B[29];
    float result135 = termA135 * termB135;

    // Expression: (-a13-a14+a23+a24-a35+a45)(-b46-b52-b53-b54+b56)(c21+c23-c31+c41-c51+c61)
    float termA136 = -A[2] - A[3] + A[8] + A[9] - A[16] + A[22];
    float termB136 = -B[23] - B[25] - B[26] - B[27] + B[29];
    float result136 = termA136 * termB136;

    // Expression: (a21-a23+a31-a41+a51-a61)(b13+b14-b23-b24-b35+b45)(-c46+c52-c53-c54+c56)
    float termA137 = A[6] - A[8] + A[12] - A[18] + A[24] - A[30];
    float termB137 = B[2] + B[3] - B[8] - B[9] - B[16] + B[22];
    float result137 = termA137 * termB137;

    // Expression: (a46+a52+a53+a54-a56)(b21-b23-b31+b41-b51+b61)(c13+c14-c23-c24-c35+c45)
    float termA138 = A[23] + A[25] + A[26] + A[27] - A[29];
    float termB138 = B[6] - B[8] - B[12] + B[18] - B[24] + B[30];
    float result138 = termA138 * termB138;

    // Expression: (a23+a24+a25)(b16-b26+b36-b46+b51-b53-b54+b56)(c21+c22-c31-c32+c41+c42-c51-c52+c61+c62)
    float termA139 = A[8] + A[9] + A[10];
    float termB139 = B[5] - B[11] + B[17] - B[23] + B[24] - B[26] - B[27] + B[29];
    float result139 = termA139 * termB139;

    // Expression: (-a21-a22-a31-a32+a41+a42-a51-a52+a61+a62)(b23+b24+b25)(c16-c26+c36-c46-c51-c53-c54+c56)
    float termA140 = -A[6] - A[7] - A[12] - A[13] + A[18] + A[19] - A[24] - A[25] + A[30] + A[31];
    float termB140 = B[8] + B[9] + B[10];
    float result140 = termA140 * termB140;

    // Expression: (-a16+a26+a36-a46+a51-a53-a54+a56)(b21+b22-b31-b32+b41+b42-b51-b52+b61+b62)(c23+c24+c25)
    float termA141 = -A[5] + A[11] + A[17] - A[23] + A[24] - A[26] - A[27] + A[29];
    float termB141 = B[6] + B[7] - B[12] - B[13] + B[18] + B[19] - B[24] - B[25] + B[30] + B[31];
    float result141 = termA141 * termB141;

    // Expression: (a52+a53+a54)(b21-b23-b24+b26-b31+b41-b51+b61)(c15+c16-c25-c26+c35+c36-c45-c46+c55+c56)
    float termA142 = A[25] + A[26] + A[27];
    float termB142 = B[6] - B[8] - B[9] + B[11] - B[12] + B[18] - B[24] + B[30];
    float result142 = termA142 * termB142;

    // Expression: (a15+a16-a25-a26-a35-a36+a45+a46-a55-a56)(b52+b53+b54)(c21+c23+c24-c26-c31+c41-c51+c61)
    float termA143 = A[4] + A[5] - A[10] - A[11] - A[16] - A[17] + A[22] + A[23] - A[28] - A[29];
    float termB143 = B[25] + B[26] + B[27];
    float result143 = termA143 * termB143;

    // Expression: (-a21+a23+a24-a26-a31+a41-a51+a61)(b15+b16-b25-b26+b35+b36-b45-b46+b55+b56)(-c52+c53+c54)
    float termA144 = -A[6] + A[8] + A[9] - A[11] - A[12] + A[18] - A[24] + A[30];
    float termB144 = B[4] + B[5] - B[10] - B[11] + B[16] + B[17] - B[22] - B[23] + B[28] + B[29];
    float result144 = termA144 * termB144;

    // Expression: (a53+a54)(b42+b43-b52-b53+b62+b63)(c35+c36-c45-c46+c55+c56+c64+c65)
    float termA145 = A[26] + A[27];
    float termB145 = B[19] + B[20] - B[25] - B[26] + B[31] + B[32];
    float result145 = termA145 * termB145;

    // Expression: (a35+a36-a45-a46+a55+a56+a64+a65)(b53+b54)(-c42+c43+c52-c53-c62+c63)
    float termA146 = A[16] + A[17] - A[22] - A[23] + A[28] + A[29] + A[33] + A[34];
    float termB146 = B[26] + B[27];
    float result146 = termA146 * termB146;

    // Expression: (a42+a43-a52-a53+a62+a63)(b35+b36-b45-b46+b55+b56+b64+b65)(c53+c54)
    float termA147 = A[19] + A[20] - A[25] - A[26] + A[31] + A[32];
    float termB147 = B[16] + B[17] - B[22] - B[23] + B[28] + B[29] + B[33] + B[34];
    float result147 = termA147 * termB147;

    // Expression: (a23+a24)(b14+b15-b24-b25+b34+b35)(-c12+c13+c21+c22-c31-c32+c41+c42)
    float termA148 = A[8] + A[9];
    float termB148 = B[3] + B[4] - B[9] - B[10] + B[15] + B[16];
    float result148 = termA148 * termB148;

    // Expression: (a12+a13-a21-a22-a31-a32+a41+a42)(b23+b24)(c14+c15-c24-c25+c34+c35)
    float termA149 = A[1] + A[2] - A[6] - A[7] - A[12] - A[13] + A[18] + A[19];
    float termB149 = B[8] + B[9];
    float result149 = termA149 * termB149;

    // Expression: (-a14-a15+a24+a25+a34+a35)(b12+b13+b21+b22-b31-b32+b41+b42)(c23+c24)
    float termA150 = -A[3] - A[4] + A[9] + A[10] + A[15] + A[16];
    float termB150 = B[1] + B[2] + B[6] + B[7] - B[12] - B[13] + B[18] + B[19];
    float result150 = termA150 * termB150;

    // Expression: (a11+a22)(b11+b22)(c11+c22)
    float termA151 = A[0] + A[7];
    float termB151 = B[0] + B[7];
    float result151 = termA151 * termB151;

    // Expression: (a33+a44)(b33+b44)(c33+c44)
    float termA152 = A[14] + A[21];
    float termB152 = B[14] + B[21];
    float result152 = termA152 * termB152;

    // Expression: (a55+a66)(b55+b66)(c55+c66)
    float termA153 = A[28] + A[35];
    float termB153 = B[28] + B[35];
    float result153 = termA153 * termB153;


    // All C writes at the end (merged):
    C[0] = result4 +result27 -result32 +result41 -result42 -result46 +result61 +result68 +result74 +result88 +result103 -result110 +result151; // c11
    C[1] = -result4 -result13 +result18 -result23 -result25 +result33 +result42 +result45 -result47 -result50 -result52 +result57 -result66 +result69 -result81 +result82 +result87 +result89 -result96 -result100 -result112 -result115 -result120 -result126 +result127 -result134 +result136 +result139 +result143 +result148; // c21
    C[2] = result4 -result5 +result13 -result18 +result23 +result25 -result45 +result47 +result50 +result51 +result52 -result57 -result62 +result66 -result69 +result75 +result81 -result82 -result87 -result89 +result96 +result100 -result101 -result104 +result111 +result112 +result115 -result116 +result120 -result124 +result126 +result134 -result136 -result139 -result143 -result148; // c31
    C[3] = result5 +result10 +result18 +result45 -result47 -result51 -result52 +result57 +result62 -result66 +result69 -result81 +result82 +result87 +result89 +result101 +result104 -result111 -result112 +result116 -result120 +result124 +result136 +result139 +result143 +result148; // c41
    C[4] = -result10 -result16 -result18 -result43 -result45 +result47 +result52 -result62 +result66 +result81 -result82 +result83 -result87 -result89 +result111 +result112 +result120 -result136 -result139 -result140 -result143; // c51
    C[5] = result10 +result18 +result45 +result62 +result79 +result82 +result87 +result89 -result113 +result118 +result136 +result139 +result143; // c61
    C[6] = result3 +result4 -result7 -result12 -result18 -result31 +result41 -result45 +result50 -result58 +result70 -result82 +result92 -result112 +result115 -result120 +result126 -result128 -result148; // c12
    C[7] = result3 -result4 +result7 +result18 -result25 +result31 +result33 +result40 +result45 -result47 -result50 +result58 +result69 -result70 -result81 +result82 +result89 -result92 -result100 -result112 -result115 -result120 -result126 +result127 +result128 +result139 +result148 +result151; // c22
    C[8] = -result3 +result4 -result7 +result14 -result18 +result24 +result26 -result31 -result45 +result50 -result58 +result70 +result73 +result81 -result82 +result92 -result94 +result100 +result112 +result115 +result120 +result126 -result128 -result135 -result139 -result148; // c32
    C[9] = result3 -result7 -result14 +result18 -result24 -result26 +result45 -result55 -result67 -result70 -result73 -result81 +result82 -result92 +result94 -result105 -result112 -result120 +result135 +result139 -result146 +result148; // c42
    C[10] = -result3 +result7 -result18 -result45 +result55 +result63 +result67 +result70 +result81 -result82 +result85 +result90 +result92 +result105 +result112 +result120 +result137 -result139 -result144 +result146; // c52
    C[11] = -result7 +result18 +result45 -result48 -result53 -result55 +result64 -result67 -result70 +result82 -result92 -result105 -result109 +result139 -result146; // c62
    C[12] = -result3 +result7 +result12 +result18 -result22 -result26 +result31 +result45 -result50 -result54 -result56 +result58 +result61 -result70 +result74 +result82 -result92 -result101 +result103 -result110 +result112 -result115 +result120 -result126 +result128 +result138 +result148; // c13
    C[13] = -result3 -result7 +result11 -result13 -result17 +result22 -result23 +result26 +result27 -result31 -result46 -result52 +result54 +result56 +result57 -result58 -result66 +result68 +result70 +result73 -result80 +result87 +result88 +result92 -result96 +result101 +result102 +result119 +result125 -result128 -result134 +result136 -result138 +result141 +result143 +result150; // c23
    C[14] = result3 +result7 -result19 -result21 -result26 -result29 +result31 -result35 +result58 -result70 -result73 +result76 -result92 -result98 -result101 +result121 +result122 +result128 -result130 +result132 +result152; // c33
    C[15] = -result3 +result7 +result10 +result19 +result24 +result26 +result29 +result32 +result55 +result67 +result70 +result73 -result76 +result92 +result98 +result101 +result105 -result122 +result124 +result129 +result130 +result146; // c43
    C[16] = result3 -result7 +result8 -result10 +result14 -result16 -result19 -result29 -result43 -result51 -result55 -result59 -result63 -result67 -result70 -result71 +result76 +result83 -result85 -result90 -result92 -result93 -result98 -result105 +result116 +result122 -result135 -result137 -result140 +result144 -result146 +result147; // c53
    C[17] = result7 +result10 +result19 +result29 +result48 +result51 +result53 +result55 +result59 -result64 +result67 +result70 +result79 +result92 +result98 +result105 -result106 +result109 -result113 +result118 +result135 +result146; // c63
    C[18] = result7 -result10 -result22 -result26 +result45 -result50 -result54 -result56 +result58 +result61 -result67 -result70 +result82 -result95 -result101 +result103 +result108 -result110 +result112 -result115 +result138 +result149; // c14
    C[19] = -result6 -result7 +result10 +result11 -result13 -result17 +result22 +result26 -result46 +result54 +result56 -result58 -result66 +result67 +result68 +result70 +result73 -result80 +result87 +result88 +result95 -result96 +result101 -result108 +result119 +result125 -result134 -result138 +result141 +result143 -result149 +result150; // c24
    C[20] = result6 +result7 -result10 -result21 -result22 -result26 -result29 -2 * result33 -result35 +result58 -result67 -result70 -result73 +result76 -result95 -result98 -result101 +result108 +result121 -result125 -result127 +result132 +result149; // c34
    C[21] = -result6 +result10 +result22 +result24 +result26 +result29 +result32 +2 * result33 +result34 +result55 +result67 +result73 -result76 +result95 +result98 +result101 +result124 +result125 +result127 +result129 +result131 +result152; // c44
    C[22] = result6 +result8 -result10 +result14 -result16 -result19 -result20 -result29 -result30 -result34 -result43 -result49 -result51 -result55 -result59 -result60 -result63 -result67 -result71 +result76 +result83 -result85 -result90 -result93 -result95 -result98 +result99 +result116 +result122 -result131 -result133 -result135 -result137 -result140 +result144 +result147; // c54
    C[23] = result6 +result9 +result10 +result15 +result19 +result29 +result34 +result48 +result51 +result53 +result55 +result59 -result64 +result67 -result77 +result79 +result95 +result98 -result106 +result109 -result113 +result117 +result118 +result123 +result131 +result135 +result145; // c64
    C[24] = -result10 -result15 +result45 -result48 -result50 +result58 +result61 -result67 -result70 -result79 -result95 +result108 +result112 +result142 +result149; // c15
    C[25] = -result6 +result10 +result15 +result48 -result58 -result66 +result67 +result70 +result79 -result84 +result87 +result88 +result95 -result108 -result109 -result117 -result134 +result141 -result142 -result149; // c25
    C[26] = result6 -result10 -result15 -result17 -result21 -result29 -result48 +result58 -result67 -result70 +result76 -result79 +result84 -result91 -result95 +result108 +result109 +result117 -result138 +result142 +result145 +result149; // c35
    C[27] = result1 -result6 +result10 +result15 +result17 +result21 +result29 +result34 +result48 +result53 +result55 +result67 -result76 +result79 -result84 +result91 +result95 -result97 -result109 -result117 +result118 +result123 +result131 +result138 -result142 -result145; // c45
    C[28] = -result1 +result6 -result10 -result15 +result28 -result34 +result36 +result37 +result44 -result48 -result53 -result55 -result67 +result72 -result79 +result84 +result86 -result95 +result97 +result109 +result117 -result118 -result123 +result130 -result131 +result142 +result145 +result153; // c55
    C[29] = result1 +result6 +result9 +result10 +result15 +result34 +result38 +result48 +result53 +result55 +result67 +result79 +result95 +result109 +result117 +result118 +result123 +result131 +result145; // c65
    C[30] = -result7 -result15 -result48 -result65 -result79 -result82 +result86 +result90 +result110 +result115 +result133 +result140 +result142; // c16
    C[31] = result7 +result13 +result15 -result44 +result46 +result48 -result49 -result63 +result65 +result79 +result80 -result84 -result86 -result90 -result109 +result114 -result117 -result133 -result140 -result142 -result143; // c26
    C[32] = -result2 -result7 -result15 +result44 -result48 +result49 -result54 +result60 +result63 -result65 +result72 -result79 +result84 +result86 +result90 +result98 -result107 +result109 -result114 +result117 -result119 -result121 +result133 +result140 +result142 +result145; // c36
    C[33] = result1 +result2 +result15 -result16 -result20 -result28 -result44 +result48 -result49 +result53 +result54 -result60 -result63 +result65 -result72 +result78 +result79 -result84 -result86 -result90 -result93 -result97 -result98 +result107 -result109 +result114 -result117 +result118 +result119 +result121 +result123 -result133 -result137 -result140 -result142 -result145; // c46
    C[34] = -result1 -result15 +result16 +result20 +result28 +result36 -result39 +result44 -result48 +result49 -result53 +result60 +result63 +result72 -result79 +result84 +result86 +result90 +result93 +result97 +result109 +result117 -result118 -result123 +result130 +result133 +result137 +result140 +result142 +result145; // c56
    C[35] = result1 +result30 +result35 +result38 +result39 +result43 +result64 +result71 +result77 +result85 +result106 +result113 +result153; // c66
}
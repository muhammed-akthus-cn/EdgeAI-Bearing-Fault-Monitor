/* Auto-generated Standalone Embedded C Inference Engine (Zero Dependencies) */
#include "edge_bearing_inference.h"

static const float FEAT_MEAN[8] = {0.562253f, 2.767931f, 5.214224f, 4.640116f, 0.328934f, 0.561786f, 149.095031f, 343.490032f};
static const float FEAT_STD[8] = {0.152023f, 1.291102f, 2.174343f, 1.090415f, 0.339727f, 0.151811f, 12.731854f, 239.145416f};
static const float ACT_SCALE_IN = 0.01752688f;
static const float ACT_SCALE_1  = 0.03264333f;
static const float ACT_SCALE_2  = 0.05816413f;
static const float EFF_SCALE_1  = 0.00011654f;
static const float EFF_SCALE_2  = 0.00025497f;
static const float EFF_SCALE_3  = 0.00050792f;

static const int8_t W1_Q[16][8] = {
    {-36, -15, -67, -2, 17, -28, 77, 34},
    {7, -27, -29, 14, 7, 26, -19, 20},
    {43, -3, -18, 44, 106, 56, 47, 60},
    {36, -11, -33, 57, 89, 69, 37, 18},
    {-6, -28, 15, -14, -65, -5, -88, -63},
    {55, 47, 52, -31, -27, 32, 28, -10},
    {-31, -6, -32, -81, -58, -64, -33, -8},
    {-45, -54, -80, 0, -93, -66, 31, 13},
    {63, 100, 39, -29, -12, 57, -29, 72},
    {-14, 62, 27, 40, 61, 61, 48, 66},
    {53, -9, -83, 81, -119, 127, 40, -62},
    {-93, -18, -42, -23, -54, -47, -99, -35},
    {-64, -46, -56, -32, -16, -18, 76, -86},
    {104, -47, -19, 39, -122, 79, -1, -8},
    {33, -36, 4, -21, 29, -37, -37, -23},
    {72, 75, -26, 45, 3, 105, 69, 95}
};

static const float B1[16] = {0.303131f, -0.171650f, 0.112713f, -0.014117f, -0.142764f, -0.210810f, -0.149673f, 0.230400f, -0.108985f, -0.049111f, 0.442701f, 0.142108f, 0.376365f, 0.455389f, -0.132213f, 0.149626f};
static const int8_t W2_Q[8][16] = {
    {35, -17, 37, 8, -66, -8, -53, 48, -42, 19, 115, -52, 90, 127, 23, 32},
    {-74, 11, 85, 85, 12, 35, -15, 20, 5, 43, -7, 35, -29, -20, 20, 61},
    {20, 7, 2, -28, -6, -21, 33, -2, -26, -35, 81, 34, 41, 99, -3, 19},
    {2, -2, 4, -24, -2, 34, -3, -25, -33, 17, -12, 0, -4, 19, 6, 19},
    {19, -34, -1, -68, 57, -71, 39, 72, -5, 2, 31, 35, -30, -1, 22, -25},
    {-4, 60, -34, 10, 4, 40, 41, 73, -40, -31, 98, 5, 71, 41, 2, 26},
    {-52, -5, 65, -18, 8, 6, 27, -3, 95, 89, -71, 87, 52, -6, -20, 78},
    {24, -4, -8, -63, 59, 41, 49, 48, 41, 25, -46, 73, 93, -46, 20, -29}
};

static const float B2[8] = {0.342002f, -0.229703f, 0.498780f, -0.175734f, -0.103833f, 0.384441f, 0.086893f, -0.139969f};
static const int8_t W3_Q[3][8] = {
    {-127, -14, -68, -7, 71, 19, 23, 105},
    {-98, 71, -114, 4, -55, -53, 114, -19},
    {100, -28, 18, 2, 25, 68, -96, -80}
};

static const float B3[3] = {-0.322916f, 0.044955f, 0.344856f};
static inline int8_t quantize_clamp(float val, float scale) {
    int32_t q = (int32_t)(val >= 0.0f ? (val / scale + 0.5f) : (val / scale - 0.5f));
    if (q > 127) return 127;
    if (q < -127) return -127;
    return (int8_t)q;
}

int8_t predict_bearing_fault(const float raw_features[NUM_FEATURES], float out_scores[NUM_CLASSES]) {
    int8_t x_q[NUM_FEATURES];
    int8_t h1_q[HIDDEN1_SIZE];
    int8_t h2_q[HIDDEN2_SIZE];

    /* 1. Normalize & Quantize Input Features to INT8 */
    for (int i = 0; i < NUM_FEATURES; i++) {
        float norm = (raw_features[i] - FEAT_MEAN[i]) / FEAT_STD[i];
        x_q[i] = quantize_clamp(norm, ACT_SCALE_IN);
    }

    /* 2. Layer 1: INT8 Multiply-Accumulate + ReLU + Re-quantize */
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
        int32_t acc = 0;
        for (int i = 0; i < NUM_FEATURES; i++) {
            acc += (int32_t)x_q[i] * (int32_t)W1_Q[j][i];
        }
        float act = (float)acc * EFF_SCALE_1 + B1[j];
        if (act < 0.0f) act = 0.0f;
        h1_q[j] = quantize_clamp(act, ACT_SCALE_1);
    }

    /* 3. Layer 2: INT8 Multiply-Accumulate + ReLU + Re-quantize */
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        int32_t acc = 0;
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            acc += (int32_t)h1_q[i] * (int32_t)W2_Q[j][i];
        }
        float act = (float)acc * EFF_SCALE_2 + B2[j];
        if (act < 0.0f) act = 0.0f;
        h2_q[j] = quantize_clamp(act, ACT_SCALE_2);
    }

    /* 4. Output Layer: INT8 Multiply-Accumulate & Argmax */
    int8_t best_class = 0;
    for (int j = 0; j < NUM_CLASSES; j++) {
        int32_t acc = 0;
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            acc += (int32_t)h2_q[i] * (int32_t)W3_Q[j][i];
        }
        out_scores[j] = (float)acc * EFF_SCALE_3 + B3[j];
        if (out_scores[j] > out_scores[best_class]) {
            best_class = (int8_t)j;
        }
    }
    return best_class;
}

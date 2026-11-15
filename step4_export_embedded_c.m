%% Step 4: Export Standalone Zero-Dependency Embedded C Code (.h & .c)
clear; clc;

% Load quantized INT8 model & normalization constants from Step 3
load('quantized_edge_net.mat');

%% 1. Write Embedded C Header File (edge_bearing_inference.h)
fidH = fopen('edge_bearing_inference.h', 'w');
fprintf(fidH, '/* Auto-generated Embedded Edge-AI Bearing Fault Inference Header */\n');
fprintf(fidH, '/* Model Footprint: %d Bytes (INT8 Quantized) | SQNR: %.2f dB */\n\n', int8Bytes, sqnrDb);
fprintf(fidH, '#ifndef EDGE_BEARING_INFERENCE_H\n');
fprintf(fidH, '#define EDGE_BEARING_INFERENCE_H\n\n');
fprintf(fidH, '#include <stdint.h>\n\n');
fprintf(fidH, '#define NUM_FEATURES 8\n');
fprintf(fidH, '#define HIDDEN1_SIZE 16\n');
fprintf(fidH, '#define HIDDEN2_SIZE 8\n');
fprintf(fidH, '#define NUM_CLASSES  3\n\n');
fprintf(fidH, '/* Predicted Fault Class IDs */\n');
fprintf(fidH, '#define CLASS_HEALTHY          0\n');
fprintf(fidH, '#define CLASS_INNER_RACE_FAULT 1\n');
fprintf(fidH, '#define CLASS_OUTER_RACE_FAULT 2\n\n');
fprintf(fidH, '/* Runs INT8 quantized inference on 8 extracted vibration features.\n');
fprintf(fidH, ' * Returns predicted class ID (0, 1, or 2) and writes raw logits to out_scores[3]. */\n');
fprintf(fidH, 'int8_t predict_bearing_fault(const float raw_features[NUM_FEATURES], float out_scores[NUM_CLASSES]);\n\n');
fprintf(fidH, '#endif /* EDGE_BEARING_INFERENCE_H */\n');
fclose(fidH);

%% 2. Write Embedded C Source File (edge_bearing_inference.c)
fidC = fopen('edge_bearing_inference.c', 'w');
fprintf(fidC, '/* Auto-generated Standalone Embedded C Inference Engine (Zero Dependencies) */\n');
fprintf(fidC, '#include "edge_bearing_inference.h"\n\n');

% Write normalization constants (mu & sigma)
writeFloatArray(fidC, 'FEAT_MEAN', mu);
writeFloatArray(fidC, 'FEAT_STD',  sigma);

% Write quantization scale factors
fprintf(fidC, 'static const float ACT_SCALE_IN = %.8ff;\n', actScaleIn);
fprintf(fidC, 'static const float ACT_SCALE_1  = %.8ff;\n', actScale1);
fprintf(fidC, 'static const float ACT_SCALE_2  = %.8ff;\n', actScale2);
fprintf(fidC, 'static const float EFF_SCALE_1  = %.8ff;\n', actScaleIn * scaleW1);
fprintf(fidC, 'static const float EFF_SCALE_2  = %.8ff;\n', actScale1 * scaleW2);
fprintf(fidC, 'static const float EFF_SCALE_3  = %.8ff;\n\n', actScale2 * scaleW3);

% Write INT8 quantized weight matrices (stored in Flash/ROM)
writeInt8Matrix(fidC, 'W1_Q', W1_q);
writeFloatArray(fidC, 'B1',   b1);
writeInt8Matrix(fidC, 'W2_Q', W2_q);
writeFloatArray(fidC, 'B2',   b2);
writeInt8Matrix(fidC, 'W3_Q', W3_q);
writeFloatArray(fidC, 'B3',   b3);

% Write helper clamp function & forward pass
fprintf(fidC, 'static inline int8_t quantize_clamp(float val, float scale) {\n');
fprintf(fidC, '    int32_t q = (int32_t)(val >= 0.0f ? (val / scale + 0.5f) : (val / scale - 0.5f));\n');
fprintf(fidC, '    if (q > 127) return 127;\n');
fprintf(fidC, '    if (q < -127) return -127;\n');
fprintf(fidC, '    return (int8_t)q;\n');
fprintf(fidC, '}\n\n');

fprintf(fidC, 'int8_t predict_bearing_fault(const float raw_features[NUM_FEATURES], float out_scores[NUM_CLASSES]) {\n');
fprintf(fidC, '    int8_t x_q[NUM_FEATURES];\n');
fprintf(fidC, '    int8_t h1_q[HIDDEN1_SIZE];\n');
fprintf(fidC, '    int8_t h2_q[HIDDEN2_SIZE];\n\n');

fprintf(fidC, '    /* 1. Normalize & Quantize Input Features to INT8 */\n');
fprintf(fidC, '    for (int i = 0; i < NUM_FEATURES; i++) {\n');
fprintf(fidC, '        float norm = (raw_features[i] - FEAT_MEAN[i]) / FEAT_STD[i];\n');
fprintf(fidC, '        x_q[i] = quantize_clamp(norm, ACT_SCALE_IN);\n');
fprintf(fidC, '    }\n\n');

fprintf(fidC, '    /* 2. Layer 1: INT8 Multiply-Accumulate + ReLU + Re-quantize */\n');
fprintf(fidC, '    for (int j = 0; j < HIDDEN1_SIZE; j++) {\n');
fprintf(fidC, '        int32_t acc = 0;\n');
fprintf(fidC, '        for (int i = 0; i < NUM_FEATURES; i++) {\n');
fprintf(fidC, '            acc += (int32_t)x_q[i] * (int32_t)W1_Q[j][i];\n');
fprintf(fidC, '        }\n');
fprintf(fidC, '        float act = (float)acc * EFF_SCALE_1 + B1[j];\n');
fprintf(fidC, '        if (act < 0.0f) act = 0.0f;\n');
fprintf(fidC, '        h1_q[j] = quantize_clamp(act, ACT_SCALE_1);\n');
fprintf(fidC, '    }\n\n');

fprintf(fidC, '    /* 3. Layer 2: INT8 Multiply-Accumulate + ReLU + Re-quantize */\n');
fprintf(fidC, '    for (int j = 0; j < HIDDEN2_SIZE; j++) {\n');
fprintf(fidC, '        int32_t acc = 0;\n');
fprintf(fidC, '        for (int i = 0; i < HIDDEN1_SIZE; i++) {\n');
fprintf(fidC, '            acc += (int32_t)h1_q[i] * (int32_t)W2_Q[j][i];\n');
fprintf(fidC, '        }\n');
fprintf(fidC, '        float act = (float)acc * EFF_SCALE_2 + B2[j];\n');
fprintf(fidC, '        if (act < 0.0f) act = 0.0f;\n');
fprintf(fidC, '        h2_q[j] = quantize_clamp(act, ACT_SCALE_2);\n');
fprintf(fidC, '    }\n\n');

fprintf(fidC, '    /* 4. Output Layer: INT8 Multiply-Accumulate & Argmax */\n');
fprintf(fidC, '    int8_t best_class = 0;\n');
fprintf(fidC, '    for (int j = 0; j < NUM_CLASSES; j++) {\n');
fprintf(fidC, '        int32_t acc = 0;\n');
fprintf(fidC, '        for (int i = 0; i < HIDDEN2_SIZE; i++) {\n');
fprintf(fidC, '            acc += (int32_t)h2_q[i] * (int32_t)W3_Q[j][i];\n');
fprintf(fidC, '        }\n');
fprintf(fidC, '        out_scores[j] = (float)acc * EFF_SCALE_3 + B3[j];\n');
fprintf(fidC, '        if (out_scores[j] > out_scores[best_class]) {\n');
fprintf(fidC, '            best_class = (int8_t)j;\n');
fprintf(fidC, '        }\n');
fprintf(fidC, '    }\n');
fprintf(fidC, '    return best_class;\n');
fprintf(fidC, '}\n');
fclose(fidC);

%% 3. Write Standalone C Test Bench (test_inference_main.c) with Real Test Sample
load('bearing_dataset.mat', 'signals', 'labels', 'fs');
sampleIdx = 105; % InnerRaceFault sample
xSample = signals(:, sampleIdx);
rmsVal = rms(xSample); peakVal = max(abs(xSample)); kurtVal = kurtosis(xSample);
crestVal = peakVal / (rmsVal + eps); skewVal = skewness(xSample); stdVal = std(xSample);
N = length(xSample); X_mag = abs(fft(xSample)); halfN = floor(N/2);
freqs = (0:halfN-1)' * (fs / N); psd = (X_mag(1:halfN)).^2;
sampleFeat = [rmsVal, peakVal, kurtVal, crestVal, skewVal, stdVal, ...
              sum(psd(freqs <= 1500))/N, sum(psd(freqs > 1500 & freqs <= 5000))/N];

fidM = fopen('test_inference_main.c', 'w');
fprintf(fidM, '#include <stdio.h>\n');
fprintf(fidM, '#include "edge_bearing_inference.h"\n\n');
fprintf(fidM, 'int main(void) {\n');
fprintf(fidM, '    const char* labels[3] = {"Healthy", "InnerRaceFault", "OuterRaceFault"};\n');
fprintf(fidM, '    float sample_features[NUM_FEATURES] = {');
fprintf(fidM, '%.6ff, ', sampleFeat(1:end-1));
fprintf(fidM, '%.6ff};\n', sampleFeat(end));
fprintf(fidM, '    float scores[NUM_CLASSES];\n');
fprintf(fidM, '    int8_t pred = predict_bearing_fault(sample_features, scores);\n');
fprintf(fidM, '    printf("Ground Truth : %s\\n", "%s");\n', char(labels(sampleIdx)));
fprintf(fidM, '    printf("C Prediction : %%s (Class %%d)\\n", labels[pred], pred);\n');
fprintf(fidM, '    printf("Logit Scores : [%%.3f, %%.3f, %%.3f]\\n", scores[0], scores[1], scores[2]);\n');
fprintf(fidM, '    return 0;\n');
fprintf(fidM, '}\n');
fclose(fidM);

disp('=====================================================');
disp('Step 4 complete! Generated standalone embedded C files:');
disp('  1. edge_bearing_inference.h');
disp('  2. edge_bearing_inference.c');
disp('  3. test_inference_main.c');
disp('=====================================================');

%% Helper Functions for C Array Formatting
function writeFloatArray(fid, name, arr)
    arr = arr(:);
    fprintf(fid, 'static const float %s[%d] = {', name, length(arr));
    fprintf(fid, '%.6ff, ', arr(1:end-1));
    fprintf(fid, '%.6ff};\n', arr(end));
end

function writeInt8Matrix(fid, name, mat)
    [rows, cols] = size(mat);
    fprintf(fid, 'static const int8_t %s[%d][%d] = {\n', name, rows, cols);
    for r = 1:rows
        fprintf(fid, '    {');
        fprintf(fid, '%d, ', mat(r, 1:end-1));
        fprintf(fid, '%d}', mat(r, end));
        if r < rows, fprintf(fid, ',\n'); else, fprintf(fid, '\n'); end
    end
    fprintf(fid, '};\n\n');
end
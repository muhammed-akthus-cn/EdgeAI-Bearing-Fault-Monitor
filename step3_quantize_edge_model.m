%% Step 3: Post-Training INT8 Quantization & Memory Footprint Benchmark
clear; clc; close all;

% 1. Load trained FP32 model and test data from Step 2
load('trained_edge_net.mat', 'net', 'mu', 'sigma', 'classNames', 'XTrainNorm', 'XTestNorm', 'YTest', 'fp32Accuracy');

% Extract weights and biases from the three fully connected layers
W1 = double(net.Layers(2).Weights); b1 = double(net.Layers(2).Bias);
W2 = double(net.Layers(4).Weights); b2 = double(net.Layers(4).Bias);
W3 = double(net.Layers(6).Weights); b3 = double(net.Layers(6).Bias);

totalParams = numel(W1) + numel(b1) + numel(W2) + numel(b2) + numel(W3) + numel(b3);
fp32Bytes   = totalParams * 4; % 4 bytes (32 bits) per parameter
int8Bytes   = totalParams * 1; % 1 byte (8 bits) per weight + compact scale factors

% 2. Symmetric INT8 Weight Quantization (per-layer dynamic scaling)
[W1_q, scaleW1] = quantizeToInt8(W1);
[W2_q, scaleW2] = quantizeToInt8(W2);
[W3_q, scaleW3] = quantizeToInt8(W3);

% Calibrate activation scales using training data (like Deep Network Quantizer)
actScaleIn = max(abs(XTrainNorm(:))) / 127;
XTrain_q   = round(max(-127, min(127, XTrainNorm / actScaleIn)));

Z1_cal     = max(0, (XTrain_q * double(W1_q)') * (actScaleIn * scaleW1) + b1');
actScale1  = max(Z1_cal(:)) / 127;

Z1_q       = round(min(127, Z1_cal / actScale1));
Z2_cal     = max(0, (Z1_q * double(W2_q)') * (actScale1 * scaleW2) + b2');
actScale2  = max(Z2_cal(:)) / 127;

% 3. Run Quantized INT8 Inference on the 60 Unseen Test Recordings
numTest   = size(XTestNorm, 1);
int8Preds = strings(numTest, 1);
fp32Logits = zeros(numTest, 3);
int8Logits = zeros(numTest, 3);

for i = 1:numTest
    x = XTestNorm(i, :);
    
    % Reference FP32 forward pass
    h1_f = max(0, x * W1' + b1');
    h2_f = max(0, h1_f * W2' + b2');
    out_f = h2_f * W3' + b3';
    fp32Logits(i, :) = out_f;
    
    % Quantized INT8 integer multiply-accumulate forward pass
    x_q  = round(max(-127, min(127, x / actScaleIn)));
    h1_q = round(min(127, max(0, (x_q * double(W1_q)') * (actScaleIn * scaleW1) + b1') / actScale1));
    h2_q = round(min(127, max(0, (h1_q * double(W2_q)') * (actScale1 * scaleW2) + b2') / actScale2));
    out_q = (h2_q * double(W3_q)') * (actScale2 * scaleW3) + b3';
    int8Logits(i, :) = out_q;
    
    [~, idx] = max(out_q);
    int8Preds(i) = classNames{idx};
end

YPredInt8    = categorical(int8Preds, classNames);
int8Accuracy = mean(YPredInt8 == YTest) * 100;

% Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB
quantNoise = fp32Logits(:) - int8Logits(:);
sqnrDb     = 10 * log10(sum(fp32Logits(:).^2) / (sum(quantNoise.^2) + eps));

fprintf('=====================================================\n');
fprintf('Total Model Parameters : %d weights & biases\n', totalParams);
fprintf('FP32 Model Footprint   : %d Bytes (%.2f KB)\n', fp32Bytes, fp32Bytes/1024);
fprintf('INT8 Model Footprint   : %d Bytes (%.2f KB) [-75.0%%]\n', int8Bytes, int8Bytes/1024);
fprintf('FP32 Test Accuracy     : %.2f%%\n', fp32Accuracy);
fprintf('INT8 Test Accuracy     : %.2f%%\n', int8Accuracy);
fprintf('Quantization SQNR      : %.2f dB\n', sqnrDb);
fprintf('=====================================================\n');

% 4. Create Side-by-Side Benchmark Figure for LinkedIn & GitHub
figure('Name', 'FP32 vs INT8 Edge Compression Benchmark', 'Color', 'w', 'Position', [120 120 860 400]);

subplot(1,2,1);
bBar = bar(categorical({'FP32 (32-bit Float)', 'INT8 (8-bit Integer)'}), [fp32Bytes, int8Bytes], 0.5);
bBar.FaceColor = 'flat';
bBar.CData(1,:) = [0.20 0.45 0.75];
bBar.CData(2,:) = [0.13 0.65 0.35];
ylabel('Parameter Memory (Bytes)');
title('Model Weight Memory (-75% Reduction)');
ylim([0 max(fp32Bytes)*1.25]); grid on;
text(1, fp32Bytes + 45, sprintf('%d B', fp32Bytes), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, int8Bytes + 45, sprintf('%d B (-75%%)', int8Bytes), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

subplot(1,2,2);
aBar = bar(categorical({'FP32 Baseline', 'INT8 Quantized'}), [fp32Accuracy, int8Accuracy], 0.5);
aBar.FaceColor = 'flat';
aBar.CData(1,:) = [0.20 0.45 0.75];
aBar.CData(2,:) = [0.13 0.65 0.35];
ylabel('Test Classification Accuracy (%)');
title(sprintf('Accuracy Retention (SQNR: %.1f dB)', sqnrDb));
ylim([0 115]); grid on;
text(1, fp32Accuracy + 3, sprintf('%.2f%%', fp32Accuracy), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, int8Accuracy + 3, sprintf('%.2f%%', int8Accuracy), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

exportgraphics(gcf, '3_quantization_benchmark.png', 'Resolution', 300);

% Save quantized parameters for Step 4 (Standalone Embedded C Code Generation)
save('quantized_edge_net.mat', 'W1_q', 'W2_q', 'W3_q', 'b1', 'b2', 'b3', ...
    'scaleW1', 'scaleW2', 'scaleW3', 'actScaleIn', 'actScale1', 'actScale2', ...
    'mu', 'sigma', 'classNames', 'totalParams', 'fp32Bytes', 'int8Bytes', 'int8Accuracy', 'sqnrDb');
disp('Step 3 complete! Saved quantized_edge_net.mat and 3_quantization_benchmark.png.');

%% Helper Function: Symmetric INT8 Quantizer
function [W_int8, scale] = quantizeToInt8(W)
    maxAbs = max(abs(W(:)));
    scale  = maxAbs / 127;
    W_int8 = int8(round(W / scale));
end
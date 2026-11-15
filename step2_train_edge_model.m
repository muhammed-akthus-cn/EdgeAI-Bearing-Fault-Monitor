%% Step 2: Edge Feature Extraction & Compact Neural Network Training
clear; clc; close all;
rng(42);

% 1. Load the dataset generated in Step 1
load('bearing_dataset.mat', 'signals', 'labels', 'fs');
numSignals = size(signals, 2);

% 2. Extract 8 lightweight Edge-friendly features per recording
features = zeros(numSignals, 8);
for i = 1:numSignals
    features(i, :) = extractEdgeFeatures(signals(:, i), fs);
end

% 3. Split into Training (80% = 240 samples) and Testing (20% = 60 samples)
cv = cvpartition(labels, 'HoldOut', 0.2);
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest  = features(test(cv), :);
YTest  = labels(test(cv));

% Normalize features (zero mean, unit variance) and save scaling parameters for C export
mu    = mean(XTrain, 1);
sigma = std(XTrain, 0, 1);
XTrainNorm = (XTrain - mu) ./ sigma;
XTestNorm  = (XTest - mu) ./ sigma;

% 4. Build an Ultra-Compact Edge Neural Network (< 3 KB footprint)
layers = [
    featureInputLayer(8, 'Name', 'input')
    fullyConnectedLayer(16, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(8, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(3, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    ];

% Train using fast Adam optimizer (takes ~3-5 seconds on CPU)
options = trainingOptions('adam', ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

net = trainnet(XTrainNorm, YTrain, layers, 'crossentropy', options);

% 5. Evaluate Accuracy on Unseen Test Recordings
scores = predict(net, XTestNorm);
classNames = categories(YTrain);
[~, predIdx] = max(scores, [], 2);
YPred = categorical(classNames(predIdx), classNames);

fp32Accuracy = mean(YPred == YTest) * 100;
fprintf('>>> FP32 Floating-Point Test Accuracy: %.2f%%\n', fp32Accuracy);

% Plot & Save Confusion Matrix for GitHub / LinkedIn
figure('Name', 'Edge AI Fault Diagnosis Confusion Matrix', 'Color', 'w', 'Position', [150 150 600 480]);
cm = confusionchart(YTest, YPred);
cm.Title = sprintf('Edge AI Bearing Fault Classifier (Test Accuracy: %.1f%%)', fp32Accuracy);
cm.RowSummary = 'row-normalized';
exportgraphics(gcf, '2_confusion_matrix.png', 'Resolution', 300);

% Save trained network & normalization constants for Step 3 (Quantization & C Code)
save('trained_edge_net.mat', 'net', 'mu', 'sigma', 'classNames', 'XTrainNorm', 'XTestNorm', 'YTest', 'fp32Accuracy');
disp('Step 2 complete! Saved trained_edge_net.mat and 2_confusion_matrix.png.');

%% Helper Function: Lightweight Edge Feature Extractor (8 Features)
function feat = extractEdgeFeatures(x, fs)
% Time-domain statistics
rmsVal   = rms(x);
peakVal  = max(abs(x));
kurtVal  = kurtosis(x);
crestVal = peakVal / (rmsVal + eps);
skewVal  = skewness(x);
stdVal   = std(x);

% Frequency-domain energy split (Low shaft band vs High resonance band)
N = length(x);
X_mag = abs(fft(x));
halfN = floor(N/2);
freqs = (0:halfN-1)' * (fs / N);
psd = (X_mag(1:halfN)).^2;

lowBandEnergy  = sum(psd(freqs <= 1500)) / N;
highBandEnergy = sum(psd(freqs > 1500 & freqs <= 5000)) / N;

feat = [rmsVal, peakVal, kurtVal, crestVal, skewVal, stdVal, lowBandEnergy, highBandEnergy];
end
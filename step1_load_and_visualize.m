%% Step 1: Generate Industrial Bearing Vibration Dataset & Visualize
clear; clc; close all;
rng(42); % Fixed random seed so your results are 100% reproducible

fs = 12000;          % Sampling rate: 12,000 samples per second (12 kHz)
duration = 0.25;     % 0.25 seconds per recording (3,000 samples)
N = fs * duration;
t = (0:N-1)' / fs;

numPerClass = 100;   % 100 recordings per condition (300 total)
signals = zeros(N, numPerClass * 3);
labels  = strings(numPerClass * 3, 1);

idx = 1;
for c = 1:3
    for i = 1:numPerClass
        % Slight random variation in motor RPM & noise for each recording
        shaftFreq = 28 + 4*rand(); % ~1680 to 1920 RPM

        % Base healthy motor vibration: shaft rotation + harmonics + sensor noise
        sig = 0.35 * sin(2*pi*shaftFreq*t) + ...
            0.15 * sin(2*pi*2*shaftFreq*t) + ...
            0.25 * randn(N, 1);

        if c == 1
            % Class 1: Healthy Bearing (smooth hum only)
            labels(idx) = "Healthy";

        elseif c == 2
            % Class 2: Inner Race Fault (sharp periodic impacts modulated by shaft rotation)
            bpfi = 5.4 * shaftFreq; % Ball Pass Frequency Inner race (~162 Hz)
            impactTimes = 0 : (1/bpfi) : duration;
            impulses = zeros(N, 1);
            for k = 1:length(impactTimes)
                t0 = impactTimes(k) + 0.0005*randn();
                active = t >= t0;
                % High-frequency 3.2 kHz structural ringing that decays quickly
                ringing = 2.8 * exp(-900*(t(active)-t0)) .* sin(2*pi*3200*(t(active)-t0));
                % Amplitude varies as the fault spins in and out of the load zone
                loadMod = 1 + 0.5*sin(2*pi*shaftFreq*t0);
                impulses(active) = impulses(active) + loadMod * ringing;
            end
            sig = sig + impulses;
            labels(idx) = "InnerRaceFault";

        elseif c == 3
            % Class 3: Outer Race Fault (stationary fault with consistent ringing impulses)
            bpfo = 3.6 * shaftFreq; % Ball Pass Frequency Outer race (~108 Hz)
            impactTimes = 0 : (1/bpfo) : duration;
            impulses = zeros(N, 1);
            for k = 1:length(impactTimes)
                t0 = impactTimes(k) + 0.0004*randn();
                active = t >= t0;
                % Lower 2.1 kHz housing resonance ringing
                ringing = 2.2 * exp(-650*(t(active)-t0)) .* sin(2*pi*2100*(t(active)-t0));
                impulses(active) = impulses(active) + ringing;
            end
            sig = sig + impulses;
            labels(idx) = "OuterRaceFault";
        end

        signals(:, idx) = sig;
        idx = idx + 1;
    end
end

labels = categorical(labels);
save('bearing_dataset.mat', 'signals', 'labels', 'fs');

%% Plot 0.1-Second Snapshot of Each Condition
snap = round(0.1 * fs);
figure('Name', 'Bearing Vibration Signatures', 'Color', 'w', 'Position', [100 100 850 560]);

subplot(3,1,1);
plot(t(1:snap), signals(1:snap, 1), 'Color', [0.13 0.55 0.13], 'LineWidth', 1);
title('1. Healthy Bearing (Smooth Shaft Rotation & Background Hum)');
ylabel('Accel (g)'); grid on; ylim([-4.5 4.5]);

subplot(3,1,2);
plot(t(1:snap), signals(1:snap, numPerClass + 1), 'Color', [0.85 0.33 0.10], 'LineWidth', 1);
title('2. Inner Race Fault (Sharp Modulated Impact Spikes at BPFI)');
ylabel('Accel (g)'); grid on; ylim([-4.5 4.5]);

subplot(3,1,3);
plot(t(1:snap), signals(1:snap, 2*numPerClass + 1), 'Color', [0.00 0.45 0.74], 'LineWidth', 1);
title('3. Outer Race Fault (Consistent Resonant Impacts at BPFO)');
xlabel('Time (seconds)'); ylabel('Accel (g)'); grid on; ylim([-4.5 4.5]);

exportgraphics(gcf, '1_vibration_signatures.png', 'Resolution', 300);
disp('Step 1 complete! Generated bearing_dataset.mat (300 signals) and saved 1_vibration_signatures.png.');
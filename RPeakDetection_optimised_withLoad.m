%% this is RPEAKDETECTION optimised

%% load your data

load('youremgdata.mat'); % add your ECG-contaminated EMG file name here
filename = 'youremgdata.mat'; %make sure loaded data has the word load in it

if contains(filename, 'load', 'IgnoreCase', true)
    load('avg_template_load.mat'); % in case of different morphology, load your own avgd ecg template (with load) here
    avg_template = avgd_sample; % it must be 141 samples long if you upload your own
    window_before = 51; % change the windows according to its relative distance from the R Peak as well
    window_after = 89;
    
else
    load ('avg_template.mat'); % this is for unloaded trials
    avg_template = avgd_sample;
    window_before = 33;
    window_after = 107;
end



%% Notch filtering all the channels
targfreq = [50 100 150] ;% to be removed, in Hz 
Q = 35 ; % quality 

for f0 = targfreq

wo = f0 / (SamplingFrequency/2); % normalised the signal
bw = wo / Q; % this is the bandwidthd

[b, a] = iirnotch(wo, bw); % made the notch filter
Data = filter(b, a, Data);

end

%% assign signal to any channel of the Data


signal = Data(:, 1);
samples = 1:length(signal);

%% Mean smoothing signal (twice for loaded data to remove any trends)

filtered_emg1 = signal;

if contains(filename, 'load', 'IgnoreCase', true)
    for i = 1:length(signal)
    k = 150;
    uppbnd = min(length(signal), i+k/2);
    lowbnd = max(1, i-k/2);
    filtered_emg1(i) = (sum(signal(lowbnd:uppbnd)))/k;
    end

    filtered_emg1 = signal - filtered_emg1; % detrends and linearises loaded data
   
end


for i = 1:length(filtered_emg1)
    k = 50;
    uppbnd = min(length(signal), i+k/2);
    lowbnd = max(1, i-k/2);
    filtered_emg1(i) = (sum(signal(lowbnd:uppbnd)))/k;
end

%% plots for mean smoothing

% figure;
% plot(samples, signal, '-b', samples, filtered_emg1 , '-m')
% title("Filtered EMG Signal, Detrended")
% xlabel("Samples")
% ylabel("Potential in mV")
% legend("Original Signal", "Filtered Signal")
% xlim([600 14000])



%% now to bandpass filter these signals

lowCutoff = 0.3;
highCutoff = 50; %0.3 to 50 Hz is the approximate freq range of ECG artifacts
order = 4;

[b, a] = butter(order/2, [lowCutoff, highCutoff]/(SamplingFrequency/2), 'bandpass');

bbfsignal = filtfilt(b, a, filtered_emg1);

% figure, plot(samples, signal, samples, bbfsignal );
% title("Comparison of Raw Signal and Bandpass Filtered")
% xlabel("Samples"), ylabel("Potential in mV"),xlim([10 10000]);
% legend("Raw", "Bandpass")


%% now to find the derivative of this signal

diff_signal = zeros(size(bbfsignal));
diff_signal(2:end-1) = (bbfsignal(3:end) - bbfsignal(1:end-2)) / (2/SamplingFrequency);  
diff_signal([1, end]) = diff_signal(2);

% figure, plot(samples, signal, 'LineWidth', 1)
% hold on, plot(samples, diff_signal, 'Color', 'm', 'LineWidth', 0.5);
% hold off;
% title("Comparison of Raw Signal and Derivative Signal")
% xlabel("Samples"), ylabel("Potential in mV"),xlim([10 10000]);
% legend("Raw", "Derivative")



%% creating the histogram

diff_signal(diff_signal>=0) = 0;
diff_signal = -diff_signal;

% Determine the max value
max_val = max(diff_signal);

% Define bin edges for 5 equal-width bins from 0 to max
bin_edges = linspace(0, max_val, 6);  % 6 edges = 5 bins

% Plot the histogram
figure;
histogram(diff_signal, bin_edges);

title('Note the number of the bin where there are outliers'); %choose a value close to the 3rd quartile of the 1st bin
xlabel('Signal Amplitude');
ylabel('Frequency');


%% creating the threshold value to detect the R peaks

%  switch up for different signals
threshold = input("Put in the threshold value for R peaks experimentally (a positive number):");
diff_signal = -diff_signal;
thresholded_signal = diff_signal;
threshold = -threshold;
%turns any part of the signal below threshold to 0
thresholded_signal((diff_signal) > threshold) = 0;

%% plotting the thresholded signal vs the original

% figure;
% plot (samples, signal, '-b', samples, thresholded_signal, '-m' )
% title("Signal vs Detected Peaks")
% xlabel("Samples")
% ylabel("Potential in mV")
% legend("Original Signal", "Thresholded Signal")
% xlim([600 24000])


%% finding the peaks

bool_peak = thresholded_signal == 0 & [false; thresholded_signal(1:end-1) < 0];
r_peaks = find(bool_peak);
rpeakcheck = nan(size(signal));

for i = r_peaks
    rpeakcheck(i) = signal(i);
end

%% confirming position of peaks kept in r_peaks using a plot
 
figure;
plot(samples, signal, 'Color', 'b', 'LineWidth', 1)
hold on;
plot(samples, rpeakcheck, 'Color', 'm', 'Marker', '.', 'MarkerSize', 15)
hold off;
xlabel("Samples")
ylabel("Potential in mV")
title("Note the number of the first valid peak")
xlim([10 10000])


%% deleting extra peaks

% make sure the first peak detected is right



% Example variables
   % Ensure column vector


% Parameters


thresh = 0.6;   % Adjust experimentally

% Initialize storage

first_valid = input("Enter the number of the first valid peak detected.(Only positive, whole nos) :");
% input the index of first valid r peak(from the r_peaks variable) here
last_valid = r_peaks(first_valid); % finds the value of r_peak at first_valid
valid_peaks = [last_valid];

%% detecting the R Peaks


% Initialize loop index
k = first_valid;

while k <= length(r_peaks)
    
    idx = r_peaks(k);
    
    % % If too close to last accepted, skip
    % if idx - last_valid <= 1000
    %     k = k + 1;
    %     continue
    % end

    r_peaks(r_peaks > last_valid & r_peaks < last_valid+1000) = [];
    
    % Check window boundaries
    if (idx - window_before) < 1 || (idx + window_after) > length(signal)
        k = k + 1;
        continue
    end

    % Instead of evaluating this single idx,
    % now collect all candidates from idx+1000 to idx+1900
    search_start = idx + 1001;
    search_end = min(idx + 2100, length(signal)-window_after);

    % Find which R_peaks fall in this interval
    search_candidates = r_peaks(r_peaks >= search_start & r_peaks <= search_end);

    % If no candidates, just proceed to next iteration
    if isempty(search_candidates)
        k = k + 1;
        continue
    end

    % Initialize best match
    best_corr = -Inf;
    best_idx = NaN;

    % Loop over all candidates
    for s = 1:length(search_candidates)
        this_idx = search_candidates(s);

        % Check boundaries again (safety)
        if this_idx - window_before < 1 || this_idx + window_after > length(signal)
            continue
        end

        % Extract and normalize
        segment = signal(this_idx - window_before : this_idx + window_after);
        seg_norm = (segment - mean(segment)) / std(segment);
        template_norm = (avg_template - mean(avg_template)) / std(avg_template);

        % Compute cross-correlation
        c = max(xcorr(seg_norm, template_norm, 'coeff'));

        % Update if this is the best so far
        if c > best_corr
            best_corr = c;
            best_idx = this_idx;
        end
    end
       
    % Accept the best peak if it exceeds threshold
    if ~isnan(best_idx) && best_corr >= thresh
        valid_peaks(end+1,1) = best_idx;
        last_valid = best_idx;
        
    end

    % Increment k
    k = k + 1;
end

%% checking if all the right peaks are identified


valpeakcheck = nan(size(signal));

for i = valid_peaks
    valpeakcheck(i) = signal(i);
end

figure;
plot(samples, signal, 'Color', 'b', 'LineWidth', 1)
hold on;
plot(samples, valpeakcheck, 'Color', 'm', 'Marker', '.', 'MarkerSize', 15)
hold off;
xlabel("Samples")
ylabel("Potential in mV")
title("Valid Peaks Detected")
xlim([10 10000])


%% initialization

CleanedData = Data;  % Make a copy
copyData = Data;     % To preserve the original
rpk = valid_peaks(1:300);  % ECG R-peak indices
avgd_sample_array = zeros(141, 32);  % Store templates
window_length = 141;



%% averaging and subtracting ECG artifacts
for i = 1:32
    signal = Data(:, i);  % Work on each channel
    avgd_sample = zeros(window_length, 1);  % Reset template

    % Calculate average ECG template
    for j = 1:length(rpk)
        idx = rpk(j);
        if idx - window_before >= 1 && idx + window_after <= length(signal)
            segment = signal(idx - window_before : idx + window_after);
            avgd_sample = avgd_sample + segment;
        end
    end
    avgd_sample = avgd_sample ./ length(rpk);  % Average template
    avgd_sample_array(:, i) = avgd_sample;     % Store template

    % Subtract template from signal
    for j = 1:length(rpk)
        idx = rpk(j);
        if idx - window_before >= 1 && idx + window_after <= length(signal)
            signal(idx - window_before : idx + window_after) = ...
                signal(idx - window_before : idx + window_after) - avgd_sample;
        end
    end

    CleanedData(:, i) = signal;  % Store cleaned signal
end

% Preserve extra columns if present
if size(Data, 2) >= 38
    CleanedData(:, 33:38) = Data(:, 33:38);
end

%% plotting only 2 seconds of the data

figure;
plot(Time, copyData(:, 5), "Color", "b", "LineWidth", 1);
hold on;
plot(Time, CleanedData(:, 5), "Color", "m", "LineWidth", 1);
hold off;
title("Comparison of Original vs Cleaned Signal")

xlabel("Time in seconds")
ylabel("Potential in mV")
legend("Original", "Cleaned")
xlim([11 13])


%% Plotting Average Templates
figure;
for plots = 1:32
    subplot(4,8,plots);
    plot(1:141, avgd_sample_array(:, plots));
    title("Channel " + num2str(plots));
    xlabel("Samples");
    ylabel("Potential (mV)");
end

%% Plotting Cleaned vs Original Data
figure;
for data = 1:32
    subplot(4, 8, data);
    plot(Time, copyData(:, data), 'b', Time, CleanedData(:, data), 'r');
    xlim([10 16]);
    title("Channel " + num2str(data));
    xlabel("Time (s)");
    ylabel("Potential (mV)");
end
legend("Original", "Cleaned");

% %% snr, before and after
% 
% snr_before = zeros(1, 32);
% snr_after = zeros(1, 32);
% 
% for ch = 1:32
%     original = copyData(:, ch);
%     cleaned = CleanedData(:, ch);
% 
%     % Signal = cleaned EMG, Noise = original - cleaned (assumed ECG artifact)
%     noise = original - cleaned;
% 
%     % Estimate power
%     signal_power = mean(cleaned.^2);
%     noise_power = mean(noise.^2);
% 
%     % SNR in dB
%     snr_after(ch) = 10 * log10(signal_power / noise_power);
% 
%     % SNR before cleaning (assume cleaned = signal, and total = signal + noise)
%     % So original is signal + noise
%     total_power = mean(original.^2);
%     snr_before(ch) = 10 * log10(signal_power / (total_power - signal_power));
% end
% 
% % Plot comparison
% figure;
% bar([snr_before; snr_after]');
% legend('Before Cleaning', 'After Cleaning');
% xlabel('Channel Number');
% ylabel('SNR (dB)');
% title('SNR Before vs After ECG Artifact Removal');
% grid on;
% 
% %% power spectrum of each channel
% 
% % Parameters
% fs = 2000;  % Replace with your actual sampling frequency if different
% nfft = 2048;
% 
% % Plot power spectra for all 32 channels
% figure;
% for ch = 1:32
%     % Get signals
%     original = copyData(:, ch);
%     cleaned = CleanedData(:, ch);
% 
%     % Compute power spectrum using FFT
%     f = (0:nfft/2-1) * (fs / nfft);
%     orig_fft = abs(fft(original, nfft)).^2 / length(original);
%     clean_fft = abs(fft(cleaned, nfft)).^2 / length(cleaned);
% 
%     % Use only the positive frequencies
%     orig_power = orig_fft(1:nfft/2);
%     clean_power = clean_fft(1:nfft/2);
% 
%     % Plot
%     subplot(4, 8, ch);
%     plot(f, orig_power, '-bs', 'LineWidth', 1); hold on;
%     plot(f, clean_power, '-gs', 'LineWidth', 1);
%     title(['Channel ' num2str(ch)]);
%     xlim([0 250]);  % Adjust depending on your signal range
%     if ch > 24
%         xlabel('Frequency (Hz)');
%     end
%     if mod(ch, 8) == 1
%         ylabel('Power');
%     end
% end
% 
% legend('Original', 'Cleaned');
% sgtitle('Power Spectrum Comparison: Original vs Cleaned (All Channels)');
















    
    











    









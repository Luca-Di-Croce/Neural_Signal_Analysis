from Utilities import *

def Data_Pipeline():
    raw1, epoched1, _ = import_mat('data/1_data_left_0.mat')
    raw2, epoched2, _ = import_mat('data/2_data_left_45.mat')
    raw3, epoched3, _ = import_mat('data/3_data_left_90.mat')
    raw4, epoched4, _ = import_mat('data/4_data_left_135.mat')
    epoched_half = np.concatenate((epoched1, epoched2, epoched3, epoched4), axis=2)
    raw_half = np.concatenate((raw1, raw2, raw3, raw4), axis=2)
    del epoched1, epoched2, epoched3, epoched4
    del raw1, raw2, raw3, raw4
    y_half = np.tile(np.arange(5), epoched_half.shape[2])
    raw5, epoched5, _ = import_mat('data/5_data_right_0.mat')
    raw6, epoched6, _ = import_mat('data/6_data_right_45.mat')
    raw7, epoched7, _ = import_mat('data/7_data_right_90.mat')
    raw8, epoched8, _ = import_mat('data/8_data_right_135.mat')
    raw = np.concatenate((raw5, raw6, raw7, raw8), axis=2)
    del raw5, raw6, raw7, raw8
    raw = np.concatenate((raw_half, raw), axis=2)
    del raw_half
    good_channels, _ = std_bad_channels(raw)
    del raw
    epoched = np.concatenate((epoched5, epoched6, epoched7, epoched8), axis=2)
    y_10 = np.tile(np.arange(5, 10), epoched.shape[2])
    epoched = np.concatenate((epoched_half, epoched), axis=2)
    y_10 = np.concatenate((y_half, y_10))
    del epoched5, epoched6, epoched7, epoched8
    y = np.tile(np.arange(5), 315)
    epoched = epoched[good_channels]
    epoched = epoched.reshape(epoched.shape[2] * epoched.shape[3], epoched.shape[1], epoched.shape[0])   # We rearrange such that we have n_trials x Time x Channels
    print("EEG data loaded and shaped with final shape:", epoched.shape)
    EEG_smoothed = []
    for i in range(epoched.shape[0]):
        EEG_smoothed.append(smooth_outliers(epoched[i, :, :]))
    del epoched
    print("EEG data smoothed.")
    EEG_smoothed = np.array(EEG_smoothed)
    EEG_smoothed = notch_filter(EEG_smoothed, fs=2000.0, line_freqs=[60, 120, 180, 240, 300], Q=35)
    print("EEG data notch filtered.")
    return EEG_smoothed, y, y_10, good_channels


def Data_FFT_pipeline(fs = 2000, window_size = 200, noverlap = 100, classes = 5):
    # Deprecated function, use Data_Pipeline() instead
    epoched, y_labeled = load_all_datasets(classes)
    epoched = epoched.reshape(epoched.shape[2] * epoched.shape[3], epoched.shape[1], epoched.shape[0])   # We rearrange such that we have n_trials x Time x Channels
    EEG_smoothed = []
    for i in range(epoched.shape[0]):
        EEG_smoothed.append(smooth_outliers(epoched[i, :, :]))
    del epoched
    EEG_smoothed = np.array(EEG_smoothed)
    X_cleaned_indx, _ = std_bad_channels(EEG_smoothed)
    EEG_cleaned = [EEG_smoothed[i] for i in X_cleaned_indx]
    EEG_cleaned = np.array(EEG_cleaned)
    y_cleaned = [y_labeled[i] for i in X_cleaned_indx]
    y_cleaned = np.array(y_cleaned)
    f, t, spectograms = compute_eeg_spectrograms(EEG_cleaned, fs=fs, nperseg=window_size, noverlap = noverlap)
    cleaned_spectogram, f = clean_spectograms(spectograms, f, remove_freqs=60, fs=fs, window_size=window_size, upper_bound=500)
    del spectograms
    return f, t, cleaned_spectogram, y_cleaned
from Utilities_DePass import *

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import filtfilt, iirnotch, spectrogram
from scipy.signal import iirnotch, filtfilt
from scipy.signal import welch


# ---------------------------------------------------- Load dataset utility function ----------------------------------------------------

def load_all_datasets(n_classes):
    """
    Load and concatenate all 8 LFP recordings from .mat files (both left and right trials).
    Returns the epoched data and corresponding labels for 5- or 10-class configurations.
    """
    # Load raw left trials (0-135 degrees)
    raw1, _, _ = import_mat('data/1_data_left_0.mat')
    raw2, _, _ = import_mat('data/2_data_left_45.mat')
    raw = np.concatenate((raw1, raw2), axis=2)
    del raw1, raw2

    raw3, _, _ = import_mat('data/3_data_left_90.mat')
    raw = np.concatenate((raw, raw3), axis=2)
    del raw3

    raw4, _, _ = import_mat('data/4_data_left_135.mat')
    raw = np.concatenate((raw, raw4), axis=2)
    del raw4

    # Load raw right trials (0-135 degrees)
    raw5, _, _ = import_mat('data/5_data_right_0.mat')
    raw = np.concatenate((raw, raw5), axis=2)
    del raw5

    raw6, _, _ = import_mat('data/6_data_right_45.mat')
    raw = np.concatenate((raw, raw6), axis=2)
    del raw6

    raw7, _, _ = import_mat('data/7_data_right_90.mat')
    raw = np.concatenate((raw, raw7), axis=2)
    del raw7

    raw8, _, _ = import_mat('data/8_data_right_135.mat')
    raw = np.concatenate((raw, raw8), axis=2)
    del raw8

    # Detect good channels from full raw dataset
    good_channels, _ = std_bad_channels(raw)
    del raw

    # Load epoched trials for left-hand data
    _, epoched1, _ = import_mat('data/1_data_left_0.mat')
    _, epoched2, _ = import_mat('data/2_data_left_45.mat')
    epoched_left = np.concatenate((epoched1, epoched2), axis=2)
    del epoched1, epoched2

    _, epoched3, _ = import_mat('data/3_data_left_90.mat')
    epoched_left = np.concatenate((epoched_left, epoched3), axis=2)
    del epoched3

    _, epoched4, _ = import_mat('data/4_data_left_135.mat')
    epoched_left = np.concatenate((epoched_left, epoched4), axis=2)
    del epoched4

    # Create 10-class labels for left-hand trials if needed
    if n_classes == 10:
        y_left = np.repeat(np.arange(0, 5), int(epoched_left.shape[2]))
        print("Y_LEFT", y_left.shape)
        print("epoched_left", epoched_left.shape)

    # Load epoched trials for right-hand data
    _, epoched5, _ = import_mat('data/5_data_right_0.mat')
    _, epoched6, _ = import_mat('data/6_data_right_45.mat')
    epoched_right = np.concatenate((epoched5, epoched6), axis=2)
    del epoched6

    _, epoched7, _ = import_mat('data/7_data_right_90.mat')
    epoched_right = np.concatenate((epoched_right, epoched7), axis=2)
    del epoched7

    _, epoched8, _ = import_mat('data/8_data_right_135.mat')
    epoched_right = np.concatenate((epoched_right, epoched8), axis=2)
    del epoched8

    epoched = np.concatenate((epoched_left, epoched_right), axis=2)

    print("EPOCHED SHAPE RIGHT", epoched_right.shape)
    print("EPOCHED SHAPE LEFT", epoched_left.shape)

    if n_classes == 10:
        y_right = np.repeat(np.arange(5, 10), int(epoched_right.shape[2]))
        print("Y_Right", y_right.shape)
        print("epoched_right", epoched_right.shape)

        y_labels_10 = np.concatenate((y_left, y_right), axis=0)
        print("y_labels", y_labels_10.shape)

        # 5-class labels disregard handedness
        y_labels_5 = np.repeat(np.arange(0, 5), int(epoched.shape[2]))
        epoched = epoched[good_channels]
        return epoched, y_labels_5, y_labels_10

    else:
        y_labels = np.repeat(np.arange(0, 4), int(epoched.shape[2]))

    del epoched_left, epoched_right
    del y_left, y_right
    epoched = epoched[good_channels]
    return epoched, y_labels, _

def load_all_datasets_quick():
    """
    Quickly load the first 4 datasets (left hand only).
    Used for fast testing without loading full dataset.
    """
    raw1, epoched1, _ = import_mat('data/1_data_left_0.mat')
    raw2, epoched2, _ = import_mat('data/2_data_left_45.mat')
    raw = np.concatenate((raw1, raw2), axis=2)
    del raw1, raw2
    epoched = np.concatenate((epoched1, epoched2), axis=2)
    del epoched1, epoched2

    raw3, epoched3, _ = import_mat('data/3_data_left_90.mat')
    raw = np.concatenate((raw, raw3), axis=2)
    del raw3
    epoched = np.concatenate((epoched, epoched3), axis=2)
    del epoched3

    raw4, epoched4, _ = import_mat('data/4_data_left_135.mat')
    raw = np.concatenate((raw, raw4), axis=2)
    del raw4
    epoched = np.concatenate((epoched, epoched4), axis=2)
    del epoched4

    # Use same channel quality filtering
    good_channels, _ = std_bad_channels(raw)
    epoched = epoched[good_channels]
    return epoched


# ---------------------------------------------------- Data Preprocessing ----------------------------------------------------

def smooth_outliers(EEG, multiplier=2.0, prev_points=5, visualize=False, fs=2000):
    """
    Smooths extreme outlier values in EEG data by replacing them with the mean of previous values.

    Parameters:
    - EEG: np.ndarray of shape (timepoints, channels)
    - multiplier: float, threshold multiplier for standard deviation
    - prev_points: int, number of previous points used for replacement
    - visualize: bool, if True, plots original vs cleaned signal
    - fs: int, sampling frequency (used only for plotting)

    Returns:
    - EEG: np.ndarray, cleaned EEG data
    """
    if visualize:
        EEG_original = EEG.copy()

    n_timepoints, n_channels = EEG.shape

    # Compute global standard deviation across channels
    std_mean = np.mean(np.std(EEG, axis=0, keepdims=True))
    thresh = multiplier * std_mean  # outlier threshold

    for ch in range(n_channels):
        for t in range(n_timepoints):
            if t < prev_points:
                # Clip initial points if insufficient history
                EEG[t, ch] = np.clip(EEG[t, ch], -thresh, thresh)
                continue
            if np.abs(EEG[t, ch]) > thresh:
                # Replace outlier with mean of previous points
                EEG[t, ch] = np.mean(EEG[t - prev_points:t, ch])

    # Optional visualization
    if visualize:
        time = np.arange(n_timepoints) / fs
        for ch in range(n_channels):
            plt.figure(figsize=(10, 4))
            plt.plot(time, EEG_original[:, ch], label='Original', alpha=0.5, color='gray')
            plt.plot(time, EEG[:, ch], label='Cleaned', color='green')
            plt.axhline(thresh, linestyle='--', color='red', label='+Threshold')
            plt.axhline(-thresh, linestyle='--', color='blue', label='−Threshold')
            plt.title(f"EEG Cleaning - Channel {ch}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return EEG

def plot_notch_filter_effect(raw_data, filtered_data, fs=2000.0, trial_idx=0, channel_idx=0, fmax=200):
    """
    Plots the power spectral density (PSD) before and after applying the notch filter.

    Parameters:
    - raw_data: np.ndarray, shape (samples, timepoints, channels), unfiltered EEG data
    - filtered_data: np.ndarray, same shape as raw_data, after notch filtering
    - fs: float, sampling frequency in Hz (default: 2000)
    - trial_idx: int, index of the trial to visualize
    - channel_idx: int, index of the channel to visualize
    - fmax: float, maximum frequency to plot in Hz (default: 200)
    """
    # Extract signals for selected trial and channel
    raw_signal = raw_data[trial_idx, :, channel_idx]
    filt_signal = filtered_data[trial_idx, :, channel_idx]

    # Compute PSDs using Welch’s method
    freqs_raw, psd_raw = welch(raw_signal, fs=fs, nperseg=fs//2)
    freqs_filt, psd_filt = welch(filt_signal, fs=fs, nperseg=fs//2)

    # Plot the PSD before and after filtering
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs_raw, psd_raw, label='Raw', alpha=0.7)
    plt.semilogy(freqs_filt, psd_filt, label='Filtered', alpha=0.7)

    # Visual markers for typical powerline frequencies
    for line_freq in [50, 100, 150]:
        plt.axvline(line_freq, color='r', ls='--', alpha=0.3)

    plt.xlim([0, fmax])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Zap Filter Effect (Trial {trial_idx}, Channel {channel_idx})")
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()


def notch_filter(eeg_data, fs=2000.0, line_freqs=[60.0, 120.0, 180.0], Q=35.0):
    """
    Applies IIR notch filters to EEG data to suppress line noise frequencies.

    Parameters:
    - eeg_data: np.ndarray, shape (samples, timepoints, channels)
    - fs: float, sampling frequency in Hz
    - line_freqs: list of float, frequencies to notch filter (e.g. [60, 120, 180])
    - Q: float, quality factor (higher values = narrower notches)

    Returns:
    - filtered_data: np.ndarray, same shape as eeg_data
    """
    samples, timepoints, channels = eeg_data.shape
    filtered_data = np.copy(eeg_data)

    # Apply each notch filter to all trials and channels
    for freq in line_freqs:
        b, a = iirnotch(w0=freq, Q=Q, fs=fs)
        for ch in range(channels):
            for trial in range(samples):
                filtered_data[trial, :, ch] = filtfilt(b, a, filtered_data[trial, :, ch])

    return filtered_data

def PCA_application(EEG_smoothed, n_components):
    n_samples, n_channels, n_timepoints = EEG_smoothed.shape

    # Reshape to (n_samples * n_timepoints, n_channels)
    X_reshaped = EEG_smoothed.transpose(0, 2, 1).reshape(-1, n_channels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reduce to, say, 10 components (i.e., virtual channels)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)  # Shape: (n_samples * n_timepoints, 10)

    X_reduced = X_reduced.reshape(n_samples, n_timepoints, -1)
    return X_reduced


# ---------------------------------------------------- Dataset Bandwidth Region Separation ----------------------------------------------------

def brainwave_frequencies(EEG_smoothed, fs=2000):
    """
    Filters EEG trials into standard brainwave frequency bands.

    Parameters:
    - EEG_smoothed: np.ndarray (samples, timepoints, channels)
    - fs: float, sampling rate in Hz

    Returns:
    - result: dict, keys are band names, values are filtered EEG arrays (samples, timepoints, channels)
    """
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 70),
        "High Gamma": (70, 100),
        "Ripples": (100, 150),
        "Fast Ripples": (150, 200),
        "Multi-Unit": (200, 500)
    }

    print("Using Bands", bands)
    result = {band: [] for band in bands}
    n_trials = EEG_smoothed.shape[0]

    for trial in range(n_trials):
        trial_data = EEG_smoothed[trial]
        for band, (low, high) in bands.items():
            filtered = bandpass_filter(trial_data, fs, low, high)
            result[band].append(filtered)

    for band in result:
        result[band] = np.stack(result[band], axis=0)

    return result

def split_EEG_by_bandwave(EEG_smoothed, fs=2000):
    brain_freqs = brainwave_frequencies(EEG_smoothed, fs)
    band_arrays = [brain_freqs[band] for band in brain_freqs.keys()]  # list of arrays, each (trials, timepoints, channels)

    EEG_bandpassed = np.stack(band_arrays, axis=1)  # shape: (trials, freqs, timepoints, channels)
    return EEG_bandpassed

def brainwave_frequencies_trial(trial_data, fs):
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 70),
        "High Gamma": (70, 100),
        "Ripples": (100, 150),
        "Fast Ripples": (150, 200),
        "Multi-Unit": (200, 500)
    }
    filtered = {}
    for band, (low, high) in bands.items():
        filtered[band] = bandpass_filter(trial_data, fs, low, high)
    return filtered

def test_bandpass_filters(EEG_smoothed, fs = 2000, channel=0, trial=0):
    """
    Plot filtered signals across all frequency bands for a single trial and channel.

    Parameters:
    - EEG_smoothed: np.ndarray (samples, timepoints, channels)
    - fs: float, sampling frequency
    - channel: int, channel to plot
    - trial: int, trial index
    """

    trial_data = EEG_smoothed[trial]
    filtered_bands = brainwave_frequencies_trial(trial_data, fs)
    time = np.arange(trial_data.shape[0]) / fs

    plt.figure(figsize=(15, 8))
    plt.subplot(6, 2, 1)
    plt.plot(time, trial_data[:, channel])
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    for i, band in enumerate(filtered_bands.keys()):
        plt.subplot(6, 2, i+2)
        plt.plot(time, filtered_bands[band][:, channel])
        plt.title(f"{band} Band")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(15, 8))
    for i, band in enumerate(reversed(filtered_bands.keys())):
        plt.plot(time, filtered_bands[band][:, channel], label=band)

    plt.legend()
    plt.show()

def split_eeg_by_region(EEG_smoothed, good_channels):
    """
    Splits EEG data into separate arrays per brain region, based on good channels.

    Parameters:
    - EEG_smoothed: np.ndarray (samples, timepoints, channels)
    - good_channels: list of int, indices of clean channels

    Returns:
    - region_dict: dict, keys are region names, values are EEG data per region (samples, timepoints, region_channels)
    """
    import pandas as pd

    region_df = pd.read_csv("data/regions.csv")
    region_dict = {}

    for _, row in region_df.iterrows():
        region_name = row['region']
        start_e = int(row['start_electrode']) - 1
        end_e = int(row['end_electrode'])

        region_channel_indices = [
            i for i, ch in enumerate(good_channels)
            if start_e <= ch < end_e
        ]

        if not region_channel_indices:
            print(f"Warning: No good channels found for region {region_name}. Skipping.")
            continue

        region_data = EEG_smoothed[:, :, region_channel_indices]
        region_dict[region_name] = region_data
        print(f"Extracted region {region_name} → shape: {region_data.shape}")

    return region_dict


def split_EEG_by_band_region(EEG_smoothed, good_channels, fs):
    """
    Splits EEG into region-wise signals, then applies bandpass filtering to each.

    Parameters:
    - EEG_smoothed: np.ndarray (samples, timepoints, channels)
    - good_channels: list of int, retained clean channels
    - fs: float, sampling frequency

    Returns:
    - regionwise_band_EEG: dict, region names → (n_bands, samples, timepoints, region_channels)
    """
    regionwise_EEG = split_eeg_by_region(EEG_smoothed, good_channels)
    regionwise_band_EEG = {}

    for region_name, region_data in regionwise_EEG.items():
        print(f"Processing region: {region_name} with shape {region_data.shape}")
        region_band_arrays = brainwave_frequencies(region_data, fs=fs)
        bands = [region_band_arrays[band] for band in region_band_arrays]
        regionwise_band_EEG[region_name] = np.array(bands)
    return regionwise_band_EEG



# ---------------------------------------------------- Spectrogram Computation and Visualization ----------------------------------------------------

def compute_eeg_spectrograms(data, fs=2000, nperseg=200, noverlap=100):
    """
    Compute spectrograms for each trial and channel using Welch’s method.

    Parameters:
    - data: np.ndarray (n_trials, n_timepoints, n_channels)
    - fs: int, sampling frequency in Hz
    - nperseg: int, length of each FFT segment
    - noverlap: int, number of points to overlap between segments

    Returns:
    - f: frequency vector
    - t: time vector
    - spectrograms: list of list of spectrogram matrices (trials x channels)
    """
    spectrograms = []
    for trial_idx in range(data.shape[0]):
        spectrograms_per_trial = []
        for channel_idx in range(data.shape[2]):
            f, t, Sxx = spectrogram(data[trial_idx, :, channel_idx],
                                    fs=fs, nperseg=nperseg, noverlap=noverlap)
            spectrograms_per_trial.append(Sxx)
        spectrograms.append(spectrograms_per_trial)
    return f, t, spectrograms

def visualize_spectogram(Sxx, f, t, title="Spectrogram", normalize=True):
    """
    Plot a single spectrogram with optional normalization.

    Parameters:
    - Sxx: 2D array (freq x time)
    - f: frequency vector
    - t: time vector
    - title: plot title
    - normalize: whether to normalize each frequency bin to [0, 1]
    """
    if normalize:
        Sxx_max = np.max(Sxx, axis=1, keepdims=True)
        Sxx_min = np.min(Sxx, axis=1, keepdims=True)
        Sxx = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)

    plt.imshow(Sxx, aspect='auto', origin='lower',
               extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title(title)

    yticks = np.linspace(f[0], f[-1], len(f))
    xticks = np.linspace(t[0], t[-1], len(t))
    plt.yticks(yticks, labels=np.round(yticks))
    plt.xticks(xticks, labels=np.round(xticks, 3), rotation=45)
    plt.show()

def visualize_spectogram_multiple(Sxx_list, f, t, normalize=True):
    """
    Display up to 6 spectrograms in a 3x2 grid.

    Parameters:
    - Sxx_list: list of 2D spectrogram arrays
    - f: frequency vector
    - t: time vector
    - normalize: whether to normalize each spectrogram
    """
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    yticks = np.linspace(f[0], f[-1], len(f))
    xticks = np.linspace(t[0], t[-1], len(t))

    for i in range(len(Sxx_list)):
        Sxx = Sxx_list[i]
        if normalize:
            Sxx_max = np.max(Sxx, axis=1, keepdims=True)
            Sxx_min = np.min(Sxx, axis=1, keepdims=True)
            Sxx = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)

        ax = axs[int(i / 2)][i % 2]
        im = ax.imshow(Sxx, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')

        ax.set_title(f'State {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def visualize_spectogram_difference(Sxx_list, f, t, normalize=True):
    """
    For each spectrogram in list, compute and display absolute difference
    with all others to highlight frequency-time region variations.

    Parameters:
    - Sxx_list: list of 2D arrays (freq x time)
    - f: frequency vector
    - t: time vector
    - normalize: bool, whether to normalize both spectrograms before comparison
    """
    for i in range(len(Sxx_list)):
        print("Difference of states compared to state", i)
        Sxx_base = Sxx_list[i]
        Sxx_diff = []

        for j in range(len(Sxx_list)):
            Sxx_temp = Sxx_list[j]
            if normalize:
                max_combined = np.maximum(np.max(Sxx_temp, axis=1, keepdims=True),
                                          np.max(Sxx_base, axis=1, keepdims=True))
                min_combined = np.minimum(np.min(Sxx_temp, axis=1, keepdims=True),
                                          np.min(Sxx_base, axis=1, keepdims=True))
                Sxx_temp = (Sxx_temp - min_combined) / (max_combined - min_combined)
                Sxx_base = (Sxx_base - min_combined) / (max_combined - min_combined)

            Sxx_diff_temp = np.abs(Sxx_base - Sxx_temp)
            Sxx_diff.append(Sxx_diff_temp)

        visualize_spectogram_multiple(Sxx_diff, f, t, normalize=True)

def clean_spectograms(spectograms, f, remove_freqs, fs, window_size, upper_bound):
    """
    Clean a list of spectrograms by:
    - Removing narrow frequency bands (e.g., 50Hz line noise)
    - Trimming to a specified frequency range

    Parameters:
    - spectograms: list of list of 2D arrays (trial x channel x freq x time)
    - f: frequency vector
    - remove_freqs: frequency to remove (e.g., 50 Hz)
    - fs: sampling frequency
    - window_size: size of FFT window
    - upper_bound: upper frequency cutoff

    Returns:
    - cleaned_spectograms: np.ndarray
    - f: trimmed frequency vector
    """
    cleaned_spectograms = []

    # Determine lower and upper bounds
    lower_bound = int(math.ceil(fs / window_size / 10.0) * 10)
    lower_bound_idx = np.where(f >= lower_bound)[0][0]
    upper_bound_idx = np.where(f <= upper_bound)[0][-1]

    # Frequency mask for notch removal
    mask = (f >= remove_freqs - 2) & (f <= remove_freqs + 2)

    for trial in spectograms:
        trial_cleaned = []
        for channel in trial:
            spect = channel.copy()
            spect[mask, :] = 0  # Remove unwanted freq band
            spect = spect[lower_bound_idx:upper_bound_idx]
            trial_cleaned.append(spect)
        cleaned_spectograms.append(trial_cleaned)

    # Return trimmed version of frequency vector
    f = f[lower_bound_idx:upper_bound_idx]
    return np.array(cleaned_spectograms), f


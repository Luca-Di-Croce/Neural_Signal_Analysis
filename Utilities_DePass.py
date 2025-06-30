import os
import builtins

import numpy as np

from matplotlib import pyplot as plt


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from scipy.signal import filtfilt, butter
import numpy as np

import hdf5storage

if not hasattr(builtins, "unicode"):
    builtins.unicode = str
    np.unicode_ = np.str_

def import_mat(fn):
    data = hdf5storage.loadmat(fn)
    raw = data["data"]["raw"][0][0]
    epoched = data["data"]["epoched"][0][0]
    event_times = data["data"]["event_times"][0][0]
    event_names = data["data"]["event_names"][0][0]
    param_dict = {}
    param_dict["hand"] = data["data"]["hand"][0][0][0]
    param_dict["spikes"] = data["data"]["spikes"][0][0][0][0][0]
    param_dict["aligned_to"] = data["data"]["alignedTo"][0][0][0][0][0]
    param_dict["angle"] = data["data"]["angle"][0][0][0]
    param_dict["event_times"] = event_times
    param_dict["event_names"] = event_names
    return raw, epoched, param_dict


def automatic_bad_channel_detection(lfp, verbose=False):
    """
    Automatic Iterative Standard Deviation method (Komosar, et al. 2022)
    Input: lfp (channels x time x trials)
    Output: remaining_channels (i.e. "clean" channels)
    """
    remaining_channels = np.arange(lfp.shape[0])
    k = 0  # iteration counter
    sd_pk = np.inf  # std of all individual channel std's
    while sd_pk > 5:
        if verbose:
            print(sd_pk)
        if k > 0:
            remaining_channels = np.setdiff1d(remaining_channels, bad_channels_k)
        sd_k = np.median(
            np.std(lfp[remaining_channels, :, :], axis=1), axis=1
        )  # std of each channel (median across trials)
        m_k = np.median(sd_k)  # median of channel std's
        third_quartile = np.percentile(sd_k, 75)
        if sd_pk == np.std(sd_k):  # if no channels are removed (not in paper)
            break
        sd_pk = np.std(sd_k)
        bad_channels_k = []
        for ch in remaining_channels:
            sd_jk = np.std(lfp[ch, :, :])
            if sd_jk < 10e-4:
                bad_channels_k.append(ch)
            elif sd_jk > 100:
                bad_channels_k.append(ch)
            elif abs(sd_jk - m_k) > third_quartile:
                bad_channels_k.append(ch)
        k += 1
    return remaining_channels


def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Compute the filter coefficients for a Butterworth bandpass filter.
    """
    # Compute the Nyquist frequency
    nyq = 0.5 * fs
    # Compute the low and high frequencies
    low = lowcut / nyq
    high = highcut / nyq
    # Compute the filter coefficients
    b, a = butter(order, [low, high], btype="band")
    # Return the filter coefficients
    return b, a


def bandpass_filter(lfp, fs, lowcut, highcut):
    """
    Apply a bandpass filter to the LFP signal.
    """
    # Compute the filter coefficients
    b, a = butter_bandpass(lowcut, highcut, fs)
    # Apply the filter
    lfp_filtered = filtfilt(b, a, lfp, axis=0)
    # Return the filtered LFP signal
    return lfp_filtered


def trajectory_plot_2d(mua, cond_names, data_type, save_dir):
    mua_mean = np.mean(mua, axis=2)
    mua_mean = mua_mean.reshape(
        mua_mean.shape[0], mua_mean.shape[1] * mua_mean.shape[2]
    )
    pca = PCA(n_components=2)
    pca.fit(mua_mean.T)
    ev = pca.explained_variance_ratio_
    X_new = pca.transform(mua_mean.T).T
    X_new = X_new.reshape((2, mua.shape[1], mua.shape[3]))
    # plot
    f = plt.figure()
    for c in range(mua.shape[3]):
        plt.plot(X_new[0, :, c], X_new[1, :, c], label=cond_names[c])
    plt.xlabel(f"PC 1 ({ev[0] * 100:.2f}%)")
    plt.ylabel(f"PC 2 ({ev[1] * 100:.2f}%)")
    plt.legend()
    # box off
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title(f"{data_type} Trajectories")
    plt.savefig(os.path.join(save_dir, data_type + "_pca.png"))
    plt.close(f)


def classify_trajectories(mua, import_conds, nw, nt_ac):
    npc = 2  # number of principal components
    # project onto first 2 PCs
    mua_mean = np.mean(mua, axis=2)
    mua_mean = mua_mean.reshape(
        mua_mean.shape[0], mua_mean.shape[1] * mua_mean.shape[2]
    )
    pca = PCA(n_components=npc)
    pca.fit(mua_mean.T)
    pc = pca.components_
    ev = pca.explained_variance_ratio_

    projected = np.empty((npc, nw, nt_ac, len(import_conds)))
    for t in range(nt_ac):
        X = mua[:, :, t, :].reshape((mua.shape[0], mua.shape[1] * mua.shape[3])).T
        X_new = np.matmul(X, pc.T)
        X_new = X_new.reshape((nw, len(import_conds), npc))
        projected[:, :, t, :] = X_new.transpose((2, 0, 1))

    X = projected.transpose((2, 3, 0, 1))
    X = X.reshape((nt_ac * len(import_conds), X.shape[2], X.shape[3]))
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    # y = np.repeat(np.arange(len(import_conds)), nt_ac)
    y = np.tile(np.arange(len(import_conds)), nt_ac)

    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    acc = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train = X[train_index, :]
        X_train = StandardScaler().fit_transform(X_train)
        X_test = X[test_index, :]
        X_test = StandardScaler().fit_transform(X_test)
        y_train = y[train_index]
        y_test = y[test_index]
        clf = LogisticRegression(max_iter=2000, random_state=0).fit(X_train, y_train)
        acc.append(clf.score(X_test, y_test))
    return acc


def box_plot(data, edge_color, fill_color):
    bp = plt.boxplot(data, patch_artist=True, showfliers=False)

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color)

    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)

    return bp


# Placeholder for any processing we might want to do while loading
def load_data(filename):
  raw, epoched, param_dict = import_mat(filename)
  return raw, epoched, param_dict

def std_bad_channels(lfp):
    # Automatic Iterative Standard Deviation method (Komosar, et al. 2022)
    # Adapted to Python by @madepass
    # Improved by @ManuelHernadezA
    # Arguments
    # =========
    # lfp: (n_channels, n timepoints, n_trials)
    all_channels = np.arange(lfp.shape[0])
    remaining_channels = all_channels.copy()
    k = 0  # iteration counter
    sd_pk = np.inf  # std of all individual channel std's
    std_all = np.std(lfp, axis=1)
    if len(std_all.shape) > 1:  # if lfp split into epochs (channels x time x trials)
        std_all = np.median(std_all, axis=1)
    while sd_pk > 5:
        sd_k = std_all[remaining_channels]  # std of each channel
        m_k = np.median(sd_k)  # median of channel std's
        third_quartile = np.percentile(sd_k, 75)
        temp = np.std(sd_k)
        if sd_pk == temp:  # if no channels are removed (not in paper)
            break
        sd_pk = temp
        bad_channels_k = []
        for ch in remaining_channels:
            sd_jk = std_all[ch]
            if sd_jk < 10e-1:
                bad_channels_k.append(ch)
            elif sd_jk > 100:
                bad_channels_k.append(ch)
            elif abs(sd_jk - m_k) > third_quartile:
                bad_channels_k.append(ch)
        remaining_channels = np.setdiff1d(remaining_channels, bad_channels_k)
        k += 1
    bad_channels = np.setdiff1d(all_channels, remaining_channels)
    return remaining_channels, bad_channels


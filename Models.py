from Utilities import *

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from tensorflow.keras.utils import to_categorical

class EchoStateNetwork:
    def __init__(self, input_dim, reservoir_size=150, spectral_radius=0.95, sparsity=0.1, alpha=0.5):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.alpha = alpha

        self.Win = np.random.uniform(-1, 1, (reservoir_size, input_dim))

        W = np.random.rand(reservoir_size, reservoir_size) - 0.5
        W[np.random.rand(*W.shape) > sparsity] = 0
        eigs = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / eigs)

        self.state = np.zeros((reservoir_size,))

    def reset_state(self):
        self.state = np.zeros((self.reservoir_size,))

    def compute_full_states(self, input_sequence):
        self.reset_state()
        states = []
        for u in input_sequence:
            state = self.update(u)
            states.append(state.copy())
        return np.array(states)

    def update(self, u):
        preact = np.dot(self.Win, u) + np.dot(self.W, self.state)
        self.state = (1 - self.alpha) * self.state + self.alpha * np.tanh(preact)
        return self.state

def run_ESN(EEG_smoothed, y_smooth, n_classes = 5, reservoir_size = 50, epochs = 25, visualize=True):
    n_samples, n_timepoints, n_channels = EEG_smoothed.shape

    y_categorical = to_categorical(y_smooth - 1, num_classes=n_classes)

    # Compute ESN reservoir states
    esn = EchoStateNetwork(input_dim=n_channels, reservoir_size = reservoir_size)
    X_reservoir = []

    print("Computing reservoir outputs (this may take a minute)...")
    for i in range(n_samples):
        full_states = esn.compute_full_states(EEG_smoothed[i])
        X_reservoir.append(full_states)

    X_reservoir = np.stack(X_reservoir)  # shape: (samples, timepoints, reservoir_size)

    X_reservoir = X_reservoir.reshape(-1, esn.reservoir_size)
    scaler = StandardScaler()
    X_reservoir = scaler.fit_transform(X_reservoir)
    X_reservoir = X_reservoir.reshape(n_samples, n_timepoints, esn.reservoir_size)

    # Train-test split
    y_labels = y_smooth
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(
        X_reservoir, y_categorical, test_size=0.2, random_state=42, stratify=y_labels
    )
    y_train = np.argmax(y_train_cat, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)

    print("Training SGDClassifier (linear model with warm_start)...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    clf = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, learning_rate='optimal', random_state=42)
    train_accuracies = []
    val_accuracies = []
    top_val_acc = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(epochs):
            clf.fit(X_train_flat, y_train)
            train_acc = clf.score(X_train_flat, y_train)
            val_acc = clf.score(X_test_flat, y_test)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            if val_acc > top_val_acc:
                top_val_acc = val_acc
                top_model = clf.get_params()
            # print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    clf.set_params(**top_model)
    print("Restoring model to", top_val_acc)
    y_pred = clf.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Final Test Accuracy (SGD Linear): {acc:.4f}")

    # Plot accuracy
    if visualize:
        plt.figure(figsize=(8, 5))
        plt.plot(train_accuracies, label='Train Acc')
        plt.plot(val_accuracies, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Linear Model Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # ---------------------------
        # Confusion Matrix
        # ---------------------------
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize rows
        class_names = [f"Class {i+1}" for i in range(n_classes)]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return acc

def run_bandwise_ESN(EEG_data, y_labels, n_classes=5, reservoir_size=50, use_shared_ESN=True, visualize=True):
    n_trials, n_bands, n_timepoints, n_channels = EEG_data.shape

    X_band_features = []

    esn = EchoStateNetwork(input_dim=n_channels, reservoir_size=reservoir_size)

    print("Extracting ESN features for each band...")
    for band in range(n_bands):
        band_features = []
        for i in range(n_trials):
            band_data = EEG_data[i, band]
            full_states = esn.compute_full_states(band_data)
            flattened = full_states.flatten()
            band_features.append(flattened)

        band_features = np.stack(band_features)
        X_band_features.append(band_features)

    X_combined = np.concatenate(X_band_features, axis=1)

    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    clf = LogisticRegression(max_iter=1000)
    train_accuracies = []
    val_accuracies = []

    for i in range(25):
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_test, y_test)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Final Meta-Model Test Accuracy: {acc:.4f}")

    if visualize:
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize rows
        class_names = [f"Class {i+1}" for i in range(n_classes)]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return acc, clf

def run_regionwise_ESN(regionwise_data, y_labels, n_classes=5, reservoir_size=50, use_shared_ESN=True, visualize=True):
    """
    regionwise_data: dict mapping region names to arrays of shape (trials, timepoints, region_channels)
    y_labels: array of shape (trials,) with class labels (0 to n_classes-1)
    """
    n_trials = y_labels.shape[0]

    X_region_features = []

    print("Extracting ESN features for each region...")
    for region_name, region_array in regionwise_data.items():
        region_features = []

        if region_array.shape[2] == 0:
            print(f"Skipping {region_name} (no channels after filtering)")
            continue

        if use_shared_ESN:
            esn = EchoStateNetwork(input_dim=region_array.shape[2], reservoir_size=reservoir_size)

        for i in range(n_trials):
            region_data = region_array[i]  # shape: (timepoints, region_channels)
            if not use_shared_ESN:
                esn = EchoStateNetwork(input_dim=region_data.shape[1], reservoir_size=reservoir_size)

            full_states = esn.compute_full_states(region_data)

            flattened = full_states.flatten()
            region_features.append(flattened)

        region_features = np.stack(region_features)
        X_region_features.append(region_features)

    X_combined = np.concatenate(X_region_features, axis=1)

    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

    clf = LogisticRegression(max_iter=1000)
    train_accuracies, val_accuracies = [], []

    for _ in range(25):
        clf.fit(X_train, y_train)
        train_accuracies.append(clf.score(X_train, y_train))
        val_accuracies.append(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Final Meta-Model Test Accuracy: {acc:.4f}")

    if visualize:
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize rows
        class_names = [f"Class {i+1}" for i in range(n_classes)]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    return acc, clf

def run_band_regionwise_ESN(regionwise_EEG, y_labels, n_classes=5, reservoir_size=50, use_shared_ESN=True, visualize=True):
    """
    Combines bandwise and regionwise ESN processing.
    regionwise_EEG: dict mapping region names to arrays of shape (freqs, samples, timepoints, channels)
    y_labels: array of shape (samples,) with class labels (0 to n_classes-1)
    """
    n_samples = y_labels.shape[0]

    # Split early to prevent leakage
    sample_indices = np.arange(n_samples)
    train_idx, test_idx, y_train, y_test = train_test_split(
        sample_indices, y_labels, test_size=0.2, stratify=y_labels, random_state=42
    )

    all_region_features_train = []
    all_region_features_test = []

    for region_name, region_data in regionwise_EEG.items():
        print(f"Processing region: {region_name}, shape: {region_data.shape}")
        freqs, samples, timepoints, channels = region_data.shape
        bandwise_features_train = []
        bandwise_features_test = []

        esn = EchoStateNetwork(input_dim=channels, reservoir_size=reservoir_size)

        for f in range(freqs):
            band_features = []
            for i in range(samples):
                band_data = region_data[f, i]  # shape: (timepoints, channels)
                full_states = esn.compute_full_states(band_data)
                band_features.append(full_states.flatten())

            band_features = np.stack(band_features)

            X_band_train = band_features[train_idx]
            X_band_test = band_features[test_idx]
            y_band_train = y_labels[train_idx]

            # Train on training data only
            band_model = LogisticRegression(max_iter=1000)
            band_model.fit(X_band_train, y_band_train)

            # Project both train and test
            proj_train = band_model.decision_function(X_band_train)
            proj_test = band_model.decision_function(X_band_test)

            if proj_train.ndim == 1:  # binary classification case
                proj_train = proj_train[:, None]
                proj_test = proj_test[:, None]

            bandwise_features_train.append(proj_train)
            bandwise_features_test.append(proj_test)

        # Stack along features
        region_feature_train = np.concatenate(bandwise_features_train, axis=1)
        region_feature_test = np.concatenate(bandwise_features_test, axis=1)

        # Region-level projection (trained on training data only)
        region_model = LogisticRegression(max_iter=1000)
        region_model.fit(region_feature_train, y_train)

        proj_region_train = region_model.decision_function(region_feature_train)
        proj_region_test = region_model.decision_function(region_feature_test)

        if proj_region_train.ndim == 1:
            proj_region_train = proj_region_train[:, None]
            proj_region_test = proj_region_test[:, None]

        all_region_features_train.append(proj_region_train)
        all_region_features_test.append(proj_region_test)

    # Combine all regions
    X_train_combined = np.concatenate(all_region_features_train, axis=1)
    X_test_combined = np.concatenate(all_region_features_test, axis=1)

    scaler = StandardScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)
    X_test_combined = scaler.transform(X_test_combined)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_combined, y_train)
    y_pred = clf.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Final Meta-Model Test Accuracy: {acc:.4f}")

    if visualize:
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize rows
        class_names = [f"Class {i+1}" for i in range(n_classes)]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # PCA with 3 components
        X_vis = PCA(n_components=3).fit_transform(np.concatenate([X_train_combined, X_test_combined], axis=0))
        y_all = np.concatenate([y_train, y_test])

        # 3D Scatter Plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=y_all, cmap='tab10', alpha=0.8)
        ax.set_title("3D PCA of ESN Features")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        fig.colorbar(scatter, ax=ax, label='Class')
        plt.tight_layout()
        plt.show()

def run_ESN_coefs(EEG_smoothed, y_smooth_5, y_smooth_10,
                                reservoir_size=50, epochs=25):
    n_samples, n_timepoints, n_channels = EEG_smoothed.shape

    # Shared ESN
    esn = EchoStateNetwork(input_dim=n_channels, reservoir_size=reservoir_size)
    X_reservoir = []

    print("Computing shared reservoir outputs...")
    for i in range(n_samples):
        full_states = esn.compute_full_states(EEG_smoothed[i])
        X_reservoir.append(full_states)
    X_reservoir = np.stack(X_reservoir)

    X_reservoir = X_reservoir.reshape(-1, esn.reservoir_size)
    scaler = StandardScaler()
    X_reservoir = scaler.fit_transform(X_reservoir)
    X_reservoir = X_reservoir.reshape(n_samples, n_timepoints, esn.reservoir_size)

    accs = []
    Wouts = []

    for label_type, y_labels in zip(['5-class', '10-class'], [y_smooth_5, y_smooth_10]):
        n_classes = len(np.unique(y_labels))
        y_categorical = to_categorical(y_labels - 1, num_classes=n_classes)

        X_train, X_test, y_train_cat, y_test_cat = train_test_split(
            X_reservoir, y_categorical, test_size=0.2, random_state=42, stratify=y_labels
        )
        y_train = np.argmax(y_train_cat, axis=1)
        y_test = np.argmax(y_test_cat, axis=1)

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        clf = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True,
                            learning_rate='optimal', random_state=42)

        top_val_acc = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for epoch in range(epochs):
                clf.fit(X_train_flat, y_train)
                val_acc = clf.score(X_test_flat, y_test)
                if val_acc > top_val_acc:
                    top_val_acc = val_acc
                    top_model = clf.get_params()

        clf.set_params(**top_model)
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        print(f" Final Test Accuracy for {label_type}: {acc:.4f}")
        accs.append(acc)

        # Extract W_out
        Wout_dict = {}
        for i in range(n_classes):
            W_class = clf.coef_[i].reshape(n_timepoints, reservoir_size).T
            Wout_dict[f'class_{i+1}'] = W_class
        Wouts.append(Wout_dict)

    return accs[0], Wouts[0], accs[1], Wouts[1], esn.W


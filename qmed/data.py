"""Dataset loading and 3-stage quantum preprocessing (standardize -> PCA -> [0, pi] angles)."""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qmed.config import RANDOM_STATE, TEST_SIZE


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    feature_names: list
    X_train_q: np.ndarray  # training patients as rotation angles
    X_test_q: np.ndarray   # held-out patients as rotation angles
    y_train: np.ndarray
    y_test: np.ndarray
    pca: PCA
    prep_time: float       # seconds spent in this whole step
    X_test_raw: np.ndarray = None  # held-out patients, original units
    X_test_std: np.ndarray = None  # after StandardScaler
    X_test_pca: np.ndarray = None  # after PCA (before angle scaling)


def load_dataset(csv_path: Path):
    """Return (df, X, y, feature_names). Column ``target``: 1 = disease, 0 = healthy."""
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"]).values
        y = df["target"].values.astype(int)
        feature_names = [c for c in df.columns if c != "target"]
    else:
        # Offline fallback if scripts/download_datasets.py wasn't run yet
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = 1 - raw.target
        X = raw.data
        y = (1 - raw.target).astype(int)
        feature_names = list(raw.feature_names)
    return df, X, y, feature_names


def subsample(X, y, max_samples: int):
    """Stratified subsample of at most ``max_samples`` patients (deterministic)."""
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, random_state=RANDOM_STATE, stratify=y
        )
    return X, y


def _angle_pipeline(X_train, X_test, n_qubits: int):
    """Fit Standardize -> PCA -> [0, pi] on X_train; also return the test-set intermediates."""
    std_scaler = StandardScaler()
    pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_train_q = angle_scaler.fit_transform(pca.fit_transform(std_scaler.fit_transform(X_train)))
    X_test_std = std_scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_std)
    X_test_q = angle_scaler.transform(X_test_pca)
    return X_train_q, X_test_q, pca, X_test_std, X_test_pca


def angle_transform(X_train, X_test, n_qubits: int):
    """Standardize -> PCA(n_qubits) -> [0, pi] angles, fit on X_train only (no test leakage).

    Returns (X_train_q, X_test_q, fitted PCA).
    """
    X_train_q, X_test_q, pca, _, _ = _angle_pipeline(X_train, X_test, n_qubits)
    return X_train_q, X_test_q, pca


def prepare_data(csv_path: Path, max_samples: int, n_qubits: int) -> PreparedData:
    """Subsample, split 75/25 (stratified), and map each patient to n_qubits angles in [0, pi]."""
    t_start = time.time()
    df, X, y, feature_names = load_dataset(csv_path)
    X, y = subsample(X, y, max_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train_q, X_test_q, pca, X_test_std, X_test_pca = _angle_pipeline(X_train, X_test, n_qubits)

    return PreparedData(
        df=df, feature_names=feature_names,
        X_train_q=X_train_q, X_test_q=X_test_q,
        y_train=y_train, y_test=y_test,
        pca=pca, prep_time=time.time() - t_start,
        X_test_raw=X_test, X_test_std=X_test_std, X_test_pca=X_test_pca,
    )

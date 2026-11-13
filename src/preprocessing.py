"""Przygotowanie danych do treningu: wybór zapisów, normalizacja, wagi klas, warianty eksperymentów."""

import numpy as np

from src.data import CLASSES, load_metadata, load_signals

CLIP_MV = 5.0
SPLITS = ["train", "val", "test"]


def single_label(df):
    """Zapisy z dokładnie jedną nadklasą (zadanie jednoetykietowe)."""
    return df[df["label_idx"] >= 0]


def fit_normalizer(X_train, clip=CLIP_MV):
    """Średnia i odchylenie standardowe każdego odprowadzenia, liczone po przycięciu."""
    # float64, bo przy ~155 mln wartości float32 zaniża odchylenie o ok. 2%
    clipped = np.clip(X_train, -clip, clip).astype(np.float64)
    return clipped.mean(axis=(0, 1)), clipped.std(axis=(0, 1))


def normalize(X, mu, sigma, clip=CLIP_MV):
    return ((np.clip(X, -clip, clip) - mu) / sigma).astype(np.float32)


def class_weights(y, power=1.0):
    """Wagi w_c = (N / (K * N_c)) ** power. power=1 to pełne wagi, 0.5 łagodniejsze, 0 brak wag."""
    counts = np.bincount(y, minlength=len(CLASSES))
    weights = (len(y) / (len(CLASSES) * counts)) ** power
    return {i: float(w) for i, w in enumerate(weights)}


def subsample(meta, n, seed=0):
    """Losowy podzbiór n zapisów z zachowaniem proporcji klas."""
    if n is None or n >= len(meta):
        return meta
    fraction = n / len(meta)
    return meta.groupby("label", group_keys=False).sample(frac=fraction, random_state=seed)


def add_label_noise(y, rate, seed=0):
    """Zamienia etykiety losowo wybranego odsetka `rate` przykładów na inną, losową klasę."""
    rng = np.random.default_rng(seed)
    y = y.copy()
    flip = rng.random(len(y)) < rate
    shift = rng.integers(1, len(CLASSES), size=flip.sum())
    y[flip] = (y[flip] + shift) % len(CLASSES)
    return y


def add_signal_noise(X, sigma, seed=0):
    """Szum gaussowski na znormalizowanym sygnale (sigma w jednostkach odchylenia standardowego odprowadzenia)."""
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0, sigma, size=X.shape)).astype(np.float32)


def load_dataset(n_train=None, seed=0, clip=CLIP_MV):
    """Gotowe dane do treningu.

    Zwraca słownik {'train', 'val', 'test'}, każdy z kluczami:
      X     sygnały (N, 1000, 12), przycięte i znormalizowane
      y     etykiety 0..4 (kolejność jak w CLASSES)
      meta  metadane tych zapisów (m.in. likelihood, noisy, human_validated), w tej samej kolejności
    oraz 'mu', 'sigma' (statystyki normalizacji). Normalizację liczymy zawsze na pełnym zbiorze
    treningowym, także gdy `n_train` wybiera mniejszy podzbiór.
    """
    meta = single_label(load_metadata())
    X_all = load_signals(meta)
    position = {ecg_id: i for i, ecg_id in enumerate(meta.index)}

    mu, sigma = fit_normalizer(X_all[(meta["split"] == "train").to_numpy()], clip)

    data = {"mu": mu, "sigma": sigma}
    for split in SPLITS:
        part = meta[meta["split"] == split]
        if split == "train":
            part = subsample(part, n_train, seed).sort_index()
        rows = [position[i] for i in part.index]
        data[split] = {
            "X": normalize(X_all[rows], mu, sigma, clip),
            "y": part["label_idx"].to_numpy(),
            "meta": part,
        }
    return data

"""Metryki i zapis predykcji, wspólne dla obu modeli."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data import CLASSES

EPS = 1e-12


def entropy(p):
    return -(p * np.log(p + EPS)).sum(axis=-1)


def uncertainty(samples):
    """Rozkład niepewności z próbek (T, N, K) na całkowitą, aleatoryczną i epistemiczną (MI)."""
    mean = samples.mean(axis=0)
    total = entropy(mean)
    aleatoric = entropy(samples).mean(axis=0)
    return {"total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}


def ece(probs, y, n_bins=15):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return sum(
        (bins == b).mean() * abs(correct[bins == b].mean() - confidence[bins == b].mean())
        for b in range(n_bins) if (bins == b).any()
    )


def reliability(probs, y, n_bins=10):
    """Tabela do reliability diagram: średnia pewność i trafność w każdym koszyku."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "pewność": confidence, "trafność": correct})
    return frame.groupby("bin").agg(pewność=("pewność", "mean"), trafność=("trafność", "mean"),
                                    przykładów=("trafność", "size"))


def metrics(probs, y):
    """Metryki z planu (README, sekcja 4) dla uśrednionych prawdopodobieństw (N, K)."""
    pred = probs.argmax(axis=1)
    one_hot = np.eye(len(CLASSES))[y]
    return {
        "accuracy": (pred == y).mean(),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "nll": -np.log(probs[np.arange(len(y)), y] + EPS).mean(),
        "brier": ((probs - one_hot) ** 2).sum(axis=1).mean(),
        "ece": ece(probs, y),
        "entropy": entropy(probs).mean(),
    }


def predictions_frame(samples, split):
    """Predykcje dla każdego przykładu razem z metadanymi potrzebnymi do analizy niepewności."""
    meta = split["meta"]
    probs = samples.mean(axis=0)
    unc = uncertainty(samples)
    frame = pd.DataFrame(probs, index=meta.index, columns=[f"p_{c}" for c in CLASSES])
    frame["y"] = split["y"]
    frame["pred"] = probs.argmax(axis=1)
    frame["correct"] = frame["pred"] == frame["y"]
    frame["confidence"] = probs.max(axis=1)
    frame["entropy"] = unc["total"]
    frame["aleatoric"] = unc["aleatoric"]
    frame["epistemic"] = unc["epistemic"]
    for column in ["label", "likelihood", "noisy", "human_validated"]:
        frame[column] = meta[column]
    return frame

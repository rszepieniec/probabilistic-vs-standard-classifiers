"""Wczytywanie i przygotowanie bazy PTB-XL."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
CACHE_DIR = ROOT / "data" / "processed"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NOISE_COLUMNS = ["static_noise", "burst_noise", "baseline_drift", "electrodes_problems"]

TRAIN_FOLDS = range(1, 9)
VAL_FOLD = 9
TEST_FOLD = 10


def _check_data_dir():
    if not (DATA_DIR / "ptbxl_database.csv").exists():
        raise FileNotFoundError(
            f"Nie znaleziono danych w {DATA_DIR}. Instrukcja w README, sekcja 'Gdzie umieścić dane'."
        )


def load_metadata():
    """Metadane wszystkich zapisów z dodatkowymi kolumnami.

    Dodane kolumny:
      superclasses     lista nadklas diagnostycznych zapisu
      label            nadklasa, jeśli zapis ma dokładnie jedną (inaczej None)
      label_idx        indeks `label` w CLASSES (lub -1)
      likelihood       najwyższa pewność diagnozy należącej do `label` (0-100, NaN gdy brak)
      noisy            czy zapis ma zgłoszony jakikolwiek problem z jakością sygnału
      human_validated  czy opis potwierdził człowiek (kopia validated_by_human)
      split            'train' (foldy 1-8), 'val' (fold 9) albo 'test' (fold 10)
    """
    _check_data_dir()
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    statements = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
    diagnostic = statements.loc[statements["diagnostic"] == 1, "diagnostic_class"]

    def superclasses(codes):
        return sorted({diagnostic[c] for c in codes if c in diagnostic.index})

    df["superclasses"] = df["scp_codes"].apply(superclasses)
    df["label"] = df["superclasses"].apply(lambda s: s[0] if len(s) == 1 else None)
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int)

    def likelihood(row):
        values = [v for c, v in row.scp_codes.items() if diagnostic.get(c) == row.label and v > 0]
        return max(values) if values else np.nan

    df["likelihood"] = df.apply(likelihood, axis=1)
    df["noisy"] = df[NOISE_COLUMNS].notna().any(axis=1)
    df["human_validated"] = df["validated_by_human"].astype(bool)
    df["split"] = np.select(
        [df["strat_fold"].isin(TRAIN_FOLDS), df["strat_fold"] == VAL_FOLD],
        ["train", "val"],
        default="test",
    )
    return df


def load_signals(df=None, sampling_rate=100):
    """Sygnały EKG o kształcie (N, długość, 12), w kolejności wierszy `df`.

    Za pierwszym razem czyta wszystkie pliki WFDB (kilka minut) i zapisuje je do
    data/processed/, kolejne wywołania wczytują gotową tablicę.
    """
    _check_data_dir()
    full = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    cache = CACHE_DIR / f"signals_{sampling_rate}hz.npy"

    if not cache.exists():
        column = "filename_lr" if sampling_rate == 100 else "filename_hr"
        signals = np.stack(
            [wfdb.rdsamp(str(DATA_DIR / f))[0] for f in tqdm(full[column], desc="Wczytywanie EKG")]
        ).astype(np.float32)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, signals)

    signals = np.load(cache, mmap_mode="r")
    if df is None:
        return np.asarray(signals)
    positions = full.index.get_indexer(df.index)
    return np.asarray(signals[positions])

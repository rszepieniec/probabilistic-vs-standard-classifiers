"""Logowanie przebiegów treningu.

Każdy trening zapisuje się w osobnym folderze runs/<data>_<godzina>_<nazwa>/:
  config.json           konfiguracja modelu, wariant danych, seed, commit gita, wersje bibliotek
  history.csv           strata i trafność w każdej epoce
  curves.png            wykres krzywych uczenia
  metrics.json          metryki na walidacji, czas treningu, liczba epok i parametrów
  predictions_val.csv   predykcje dla każdego przykładu walidacyjnego (z pewnością diagnozy itd.)
  model.weights.h5      wagi modelu (nie trafiają do gita)

Wszystkie przebiegi są też zebrane w runs/index.csv (jeden wiersz na przebieg).
Tabela przebiegów w terminalu: python -m src.runs
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime

from matplotlib.figure import Figure

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from src.data import ROOT
from src.evaluate import metrics, predictions_frame
from src.models import DEFAULT_CONFIG, build_model, predict_samples, train
from src.preprocessing import class_weights

RUNS_DIR = ROOT / "runs"
INDEX = RUNS_DIR / "index.csv"


@dataclass
class Run:
    run_id: str
    model: object
    config: dict
    metrics: dict
    history: pd.DataFrame
    predictions: pd.DataFrame

    @property
    def path(self):
        return RUNS_DIR / self.run_id


def _git():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "src"))}


def _key(description):
    return hashlib.sha1(json.dumps(description, sort_keys=True, default=str).encode()).hexdigest()[:10]


def list_runs():
    """Tabela wszystkich przebiegów (runs/index.csv)."""
    return pd.read_csv(INDEX, index_col="run_id") if INDEX.exists() else pd.DataFrame()


def load_run(run_id, data):
    """Wczytuje zapisany przebieg razem z modelem."""
    path = RUNS_DIR / run_id
    saved = json.loads((path / "config.json").read_text())
    model = build_model(saved["model_config"], saved["probabilistic"], n_train=len(data["train"]["y"]))
    model.load_weights(path / "model.weights.h5")
    return Run(
        run_id=run_id, model=model, config=saved,
        metrics=json.loads((path / "metrics.json").read_text()),
        history=pd.read_csv(path / "history.csv", index_col="epoch"),
        predictions=pd.read_csv(path / "predictions_val.csv", index_col="ecg_id"),
    )


def run_or_load(name, data, config=None, probabilistic=False, seed=0, data_params=None, verbose=0):
    """Trenuje i loguje przebieg albo wczytuje identyczny przebieg z runs/.

    Przebieg uznajemy za identyczny, gdy zgadzają się: typ modelu, konfiguracja, seed i wariant danych
    (`data_params`, np. {"n_train": 1000, "label_noise": 0.2}). Zmiana czegokolwiek tworzy nowy przebieg.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    data_params = {"n_train": len(data["train"]["y"]), **(data_params or {})}
    description = {"probabilistic": probabilistic, "model_config": config, "seed": seed, "data": data_params}
    key = _key(description)

    runs = list_runs()
    if len(runs) and (runs["key"] == key).any():
        return load_run(runs[runs["key"] == key].index[-1], data)

    tf.keras.utils.set_random_seed(seed)
    model = build_model(config, probabilistic, n_train=len(data["train"]["y"]))
    weights = class_weights(data["train"]["y"], config["class_weight_power"])
    history, seconds = train(model, data, config, weights, verbose)

    T = config["mc_samples"] if probabilistic else 1
    samples = predict_samples(model, data["val"]["X"], T=T)
    val_metrics = {k: float(v) for k, v in metrics(samples.mean(axis=0), data["val"]["y"]).items()}
    predictions = predictions_frame(samples, data["val"])

    run_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{name}"
    path = RUNS_DIR / run_id
    path.mkdir(parents=True)

    saved = {**description, "name": name, "key": key, "git": _git(),
             "versions": {"tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__}}
    info = {"val": val_metrics, "seconds": seconds, "epochs": len(history["loss"]),
            "parameters": model.count_params(), "mc_samples": T}
    history = pd.DataFrame(history).rename_axis("epoch")

    (path / "config.json").write_text(json.dumps(saved, indent=2, default=str))
    (path / "metrics.json").write_text(json.dumps(info, indent=2))
    history.to_csv(path / "history.csv")
    predictions.to_csv(path / "predictions_val.csv")
    model.save_weights(path / "model.weights.h5")
    _plot_history(history, run_id, path / "curves.png")

    row = {"run_id": run_id, "name": name, "model": "B" if probabilistic else "A", "seed": seed,
           **{f"data_{k}": v for k, v in data_params.items()},
           **{f"cfg_{k}": str(v) if isinstance(v, (tuple, list)) else v for k, v in config.items()},
           **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
           "epochs": info["epochs"], "minutes": round(seconds / 60, 2), "parameters": info["parameters"],
           "commit": saved["git"]["commit"] + ("*" if saved["git"]["dirty"] else ""), "key": key}
    previous = list_runs()
    rows = [previous.reset_index(), pd.DataFrame([row])] if len(previous) else [pd.DataFrame([row])]
    pd.concat(rows, ignore_index=True).to_csv(INDEX, index=False)

    return Run(run_id, model, saved, info, history, predictions)


def _plot_history(history, title, path):
    # Figure zamiast pyplot: nie zmienia backendu matplotlib, więc wykresy w notebookach dalej się wyświetlają
    fig = Figure(figsize=(16, 3.5))
    axes = fig.subplots(1, 3)
    for key, ax in zip(["loss", "nll", "accuracy"], axes):
        ax.plot(history[key], label="trening")
        ax.plot(history[f"val_{key}"], label="walidacja")
        ax.set(title=key, xlabel="epoka")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)


if __name__ == "__main__":
    columns = ["model", "seed", "data_n_train", "val_accuracy", "val_macro_f1", "val_nll", "val_ece",
               "epochs", "minutes", "commit"]
    runs = list_runs()
    with pd.option_context("display.width", 250, "display.max_rows", None, "display.max_columns", None):
        print(runs[[c for c in columns if c in runs]] if len(runs) else "Brak przebiegów w runs/")

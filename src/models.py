"""Model A (deterministyczny) i model B (probabilistyczny) o tej samej architekturze."""

import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.data import CLASSES

keras = tf.keras

DEFAULT_CONFIG = {
    "filters": (32, 64, 64, 128),
    "kernel_size": 7,
    "dense_units": 64,
    "dropout": 0.3,          # tylko model A
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 60,
    "patience": 8,
    "class_weight_power": 1.0,
    "kl_scale": 1.0,         # tylko model B: mnożnik składnika KL (1.0 = pełna wersja bayesowska)
    "mc_samples": 50,        # tylko model B: liczba przebiegów T przy predykcji
}


def build_model(config=None, probabilistic=False, n_train=None, input_shape=(1000, 12)):
    """Sieć 1D-CNN: bloki [konwolucja -> BatchNorm -> ReLU -> MaxPool], potem warstwa gęsta.

    Wyjście to logity (bez softmax). W modelu B konwolucje i warstwy gęste są typu Flipout,
    a składnik KL każdej warstwy jest dzielony przez liczbę przykładów treningowych `n_train`.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}

    if probabilistic:
        if n_train is None:
            raise ValueError("Model probabilistyczny wymaga n_train (skalowanie składnika KL)")
        scale = config["kl_scale"] / n_train

        def kl(q, p, _):
            return tfp.distributions.kl_divergence(q, p) * scale

        conv = lambda f: tfp.layers.Convolution1DFlipout(
            f, config["kernel_size"], padding="same", kernel_divergence_fn=kl)
        dense = lambda units, activation=None: tfp.layers.DenseFlipout(
            units, activation=activation, kernel_divergence_fn=kl)
    else:
        conv = lambda f: keras.layers.Conv1D(f, config["kernel_size"], padding="same")
        dense = lambda units, activation=None: keras.layers.Dense(units, activation=activation)

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for f in config["filters"]:
        x = conv(f)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = dense(config["dense_units"], "relu")(x)
    if not probabilistic and config["dropout"] > 0:
        x = keras.layers.Dropout(config["dropout"])(x)
    outputs = dense(len(CLASSES))(x)

    model = keras.Model(inputs, outputs, name="model_B" if probabilistic else "model_A")
    model.compile(
        optimizer=keras.optimizers.Adam(config["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # Osobna metryka NLL: w modelu B strata zawiera też składnik KL, a early stopping
        # ma patrzeć tylko na jakość predykcji. W modelu A nll == loss.
        metrics=["accuracy", keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="nll")],
    )
    return model


def train(model, data, config=None, class_weight=None, verbose=2):
    """Trening z early stopping na walidacyjnym NLL. Zwraca historię i czas treningu."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_nll", patience=config["patience"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_nll", factor=0.5,
                                          patience=max(2, config["patience"] // 3)),
    ]
    start = time.time()
    history = model.fit(
        data["train"]["X"], data["train"]["y"],
        validation_data=(data["val"]["X"], data["val"]["y"]),
        epochs=config["epochs"], batch_size=config["batch_size"],
        class_weight=class_weight, callbacks=callbacks, verbose=verbose,
    )
    return history.history, time.time() - start


def predict_samples(model, X, T=1, batch_size=256):
    """Prawdopodobieństwa klas z T przebiegów, kształt (T, N, K).

    Dla modelu A wszystkie przebiegi są identyczne, więc wystarcza T=1.
    W modelu B każdy przebieg losuje inne wagi.
    """
    return np.stack([
        tf.nn.softmax(model.predict(X, batch_size=batch_size, verbose=0)).numpy()
        for _ in range(T)
    ])

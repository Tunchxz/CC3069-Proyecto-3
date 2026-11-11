"""Definición, entrenamiento y evaluación de la CNN para el alfabeto ASL.

El dispositivo (CPU o GPU) se determina por la variable de entorno
``CUDA_VISIBLE_DEVICES``, que debe fijarse antes de importar este módulo.

Ejemplo::

    from src.models.train_model import build_cnn, train
    model = build_cnn(n_classes=29)
    result = train(model, X_train, y_train, X_val, y_val)
"""

import time
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config


class EpochTimer(keras.callbacks.Callback):
    """Callback de Keras que registra la duración de cada época en segundos."""

    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._start)


@dataclass
class TrainResult:
    """Resultado de una ejecución de entrenamiento.

    Attributes:
        total_time: Duración total de ``model.fit`` en segundos.
        epoch_times: Duración de cada época en segundos.
        history: Métricas por época devueltas por Keras.
    """

    total_time: float
    epoch_times: list[float]
    history: dict[str, list[float]] = field(default_factory=dict)


def build_cnn(
    n_classes: int,
    input_shape: tuple[int, int, int] = (config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS),
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """Construye y compila la CNN de tres bloques convolucionales.

    Arquitectura: 3 x [Conv2D, BatchNorm, ReLU, MaxPool] con 32, 64 y 128
    filtros, seguidos de Dropout, Dense(128) y una capa softmax.

    Args:
        n_classes: Número de clases de salida.
        input_shape: Forma de cada imagen de entrada.
        learning_rate: Tasa de aprendizaje del optimizador Adam.

    Returns:
        Modelo de Keras compilado con pérdida ``sparse_categorical_crossentropy``.
    """
    layers = [keras.Input(shape=input_shape)]
    for filters in (32, 64, 128):
        layers += [
            keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(),
        ]
    layers += [
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ]
    model = keras.Sequential(layers, name="asl_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    verbose: int = 2,
) -> TrainResult:
    """Entrena el modelo con un número fijo de épocas y mide su duración.

    Args:
        model: Modelo compilado.
        X_train: Imágenes de entrenamiento.
        y_train: Etiquetas enteras de entrenamiento.
        X_val: Imágenes de validación.
        y_val: Etiquetas enteras de validación.
        epochs: Número de épocas.
        batch_size: Tamaño del lote.
        verbose: Nivel de detalle de Keras.

    Returns:
        ``TrainResult`` con el tiempo total, los tiempos por época y el historial.
    """
    timer = EpochTimer()
    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[timer],
        verbose=verbose,
    )
    total = time.perf_counter() - start
    return TrainResult(total_time=total, epoch_times=timer.times, history=history.history)


def evaluate(model: keras.Model, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Evalúa el modelo y mide la duración de la inferencia.

    Args:
        model: Modelo entrenado.
        X: Imágenes de evaluación.
        y: Etiquetas enteras.

    Returns:
        Tupla ``(loss, accuracy, segundos)``.
    """
    start = time.perf_counter()
    loss, acc = model.evaluate(X, y, batch_size=config.BATCH_SIZE, verbose=0)
    return loss, acc, time.perf_counter() - start


def set_seed(seed: int = config.SEED) -> None:
    """Fija las semillas de Python, NumPy y TensorFlow."""
    keras.utils.set_random_seed(seed)


def device_name() -> str:
    """Devuelve ``"GPU"`` si TensorFlow detecta una GPU y ``"CPU"`` en otro caso."""
    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

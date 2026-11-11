"""Configuración global del proyecto: rutas, semilla e hiperparámetros."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_DATASET = "grassknoted/asl-alphabet"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

SEED = 42
N_PER_CLASS = 500
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

IMG_SIZE = 64
N_CHANNELS = 3

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

N_REPS_PREPROCESS = 5
N_REPS_TRAIN = 3

"""Descarga del dataset ASL Alphabet, submuestreo estratificado y partición.

Uso desde la línea de comandos::

    python -m src.data.make_dataset

Genera ``data/processed/manifest.csv`` con las columnas ``path``, ``label``,
``label_idx`` y ``split``.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_dataset() -> Path:
    """Descarga el dataset con kagglehub y devuelve la carpeta de entrenamiento.

    Returns:
        Ruta de la carpeta que contiene una subcarpeta por clase.
    """
    import kagglehub

    root = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    return find_train_dir(root)


def find_train_dir(root: Path) -> Path:
    """Localiza la carpeta con las 29 subcarpetas de clases dentro de ``root``.

    Args:
        root: Carpeta raíz del dataset descargado.

    Returns:
        Ruta de la carpeta de entrenamiento.

    Raises:
        FileNotFoundError: Si no existe una carpeta con 29 subcarpetas.
    """
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        subdirs = [p for p in candidate.iterdir() if p.is_dir()]
        if len(subdirs) == 29 and "test" not in candidate.name.lower():
            return candidate
    raise FileNotFoundError(f"No se encontró la carpeta de entrenamiento en {root}")


def list_images(train_dir: Path) -> pd.DataFrame:
    """Lista todas las imágenes del dataset con su clase.

    Args:
        train_dir: Carpeta con una subcarpeta por clase.

    Returns:
        DataFrame con las columnas ``path`` y ``label``.
    """
    rows = [
        {"path": str(f), "label": class_dir.name}
        for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir())
        for f in sorted(class_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return pd.DataFrame(rows)


def subsample(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Selecciona al azar ``n_per_class`` imágenes de cada clase.

    Args:
        df: DataFrame con las columnas ``path`` y ``label``.
        n_per_class: Número de imágenes por clase.
        seed: Semilla del generador aleatorio.

    Returns:
        DataFrame submuestreado y ordenado por clase.
    """
    return (
        df.groupby("label", group_keys=False)
        .sample(n=n_per_class, random_state=seed)
        .sort_values(["label", "path"])
        .reset_index(drop=True)
    )


def split(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Asigna cada imagen a ``train``, ``val`` o ``test`` de forma estratificada.

    Args:
        df: DataFrame con la columna ``label``.
        fractions: Proporciones con las claves ``train``, ``val`` y ``test``.
        seed: Semilla de la partición.

    Returns:
        Copia de ``df`` con las columnas ``label_idx`` y ``split``.
    """
    df = df.copy()
    classes = sorted(df["label"].unique())
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)})

    holdout = fractions["val"] + fractions["test"]
    train_idx, rest_idx = train_test_split(
        df.index, test_size=holdout, stratify=df["label"], random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=fractions["test"] / holdout,
        stratify=df.loc[rest_idx, "label"],
        random_state=seed,
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def make_dataset(force: bool = False) -> pd.DataFrame:
    """Crea o carga el manifiesto de la submuestra particionada.

    Args:
        force: Si es ``True``, regenera el manifiesto aunque ya exista.

    Returns:
        DataFrame con las columnas ``path``, ``label``, ``label_idx`` y ``split``.
    """
    if config.MANIFEST_PATH.exists() and not force:
        return pd.read_csv(config.MANIFEST_PATH)

    train_dir = download_dataset()
    df = list_images(train_dir)
    df = subsample(df, config.N_PER_CLASS, config.SEED)
    df = split(df, config.SPLIT_FRACTIONS, config.SEED)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MANIFEST_PATH, index=False)
    return df


def class_names(df: pd.DataFrame) -> list[str]:
    """Devuelve los nombres de clase ordenados por ``label_idx``."""
    return df.drop_duplicates("label_idx").sort_values("label_idx")["label"].tolist()


if __name__ == "__main__":
    manifest = make_dataset(force=True)
    print(manifest.groupby("split").size())
    print(f"Clases: {manifest['label'].nunique()}")

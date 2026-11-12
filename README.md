# Proyecto 3: Mejorando Algoritmos con CUDA

Clasificador de letras del alfabeto ASL (Laboratorio 3) ejecutado de forma secuencial en CPU y acelerado con CUDA.

## Estructura

```
notebooks/
  01-analisis-secuencial.ipynb   Línea base en CPU (pasos 1 y 2)
src/
  config.py                      Rutas, semilla e hiperparámetros
  data/make_dataset.py           Descarga, submuestra (500 por clase) y partición 70/15/15
  features/build_features.py     Preprocesamiento secuencial (lectura, 64x64, normalización)
  models/train_model.py          CNN de Keras, entrenamiento y evaluación con medición de tiempo
  utils/timing.py                Medición y guardado de tiempos
  utils/monitor.py               Muestreo del uso de CPU por núcleo y de GPU
  utils/system_info.py           Reporte de hardware y software
reports/
  results/                       Tiempos y uso de recursos en CSV
  figures/                       Gráficas generadas por los notebooks
```

## Requisitos

- Linux o WSL2 con una GPU NVIDIA y el controlador instalado (`nvidia-smi` debe funcionar).
- Python 3.12.

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name cc3069-p3 --display-name "CC3069 P3 (.venv)"
```

Verificación del entorno:

```bash
python -m src.utils.system_info
```

## Ejecución

Los notebooks se ejecutan desde la carpeta `notebooks/` con el kernel `CC3069 P3 (.venv)`. La primera ejecución descarga el dataset de Kaggle (alrededor de 1 GB) en `~/.cache/kagglehub` y crea `data/processed/manifest.csv`.

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/01-analisis-secuencial.ipynb
```

Los resultados se guardan en `reports/results/` y las figuras en `reports/figures/`.

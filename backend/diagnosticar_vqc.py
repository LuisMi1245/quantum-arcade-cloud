#!/usr/bin/env python
r"""Compare the original Qiskit VQC.predict path against Cloud Run's NumPy statevector path.

Run from the project's backend folder, for example:
  python diagnosticar_vqc.py ..\gestos_test_3.csv

The CSV must contain the five angle columns and optionally a 'label' column.
This script loads the .dill locally; do NOT commit datasets or model files publicly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
MODEL_PATH = BACKEND_DIR / "models" / "vqc_piedra_papel_tijera_seed_0.dill"
FEATURES = ["pulgar", "indice", "medio", "anular", "menique"]
LABELS = {"piedra": 0, "papel": 1, "tijera": 2, "tijeras": 2, "rock": 0, "paper": 1, "scissors": 2}
NAMES = {0: "PIEDRA", 1: "PAPEL", 2: "TIJERA"}

# dill snapshots can refer to this helper through __main__.
def interpretar_tres_clases(indice):
    return indice % 3


def normalizar_predicciones(modelo, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(modelo.predict(X))
    if raw.ndim == 2 and raw.shape[0] == len(X) and raw.shape[1] == 3:
        return np.argmax(raw, axis=1).astype(int)
    raw = raw.reshape(-1)
    if len(raw) != len(X):
        raise ValueError(f"Formato de predict() inesperado: {np.asarray(modelo.predict(X)).shape}")
    return raw.astype(int)


def main() -> int:
    csv_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else PROJECT_DIR / "gestos_test_3.csv"
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not MODEL_PATH.is_file():
        print(f"ERROR: no encuentro el modelo en {MODEL_PATH}")
        return 2
    if not csv_path.is_file():
        print(f"ERROR: no encuentro el CSV en {csv_path}")
        print("Indica la ruta como argumento, por ejemplo: python diagnosticar_vqc.py ..\\gestos_test_3.csv")
        return 2

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: faltan columnas en el CSV: {missing}")
        return 2
    X = df[FEATURES].to_numpy(dtype=float)
    if len(X) == 0 or not np.isfinite(X).all():
        print("ERROR: el CSV no contiene entradas válidas o incluye valores no finitos")
        return 2

    # Resolve the backend module's globals and use its exact production NumPy implementation.
    os.environ["VQC_MODEL_PATH"] = str(MODEL_PATH)
    os.environ["USE_FIRESTORE"] = "false"
    sys.path.insert(0, str(BACKEND_DIR))

    print(f"CSV: {csv_path}")
    print(f"Muestras: {len(X)}")
    print("Cargando VQC para la predicción original...")
    from qiskit_machine_learning.algorithms import VQC
    vqc = VQC.from_dill(str(MODEL_PATH))

    print("Ejecutando VQC.predict() original...")
    original = normalizar_predicciones(vqc, X)

    print("Ejecutando la inferencia NumPy usada por Cloud Run...")
    import main as cloud_main
    numpy_predictions = np.array([
        cloud_main.predict_class_from_saved_circuit(X[i:i+1])[0]
        for i in range(len(X))
    ], dtype=int)

    mismatches = np.flatnonzero(original != numpy_predictions)
    print("\nCOMPARACIÓN DE INFERENCIAS")
    print(f"Predicciones distintas: {len(mismatches)}/{len(X)} ({100*len(mismatches)/len(X):.2f}%)")
    print("Distribución original:", {NAMES[i]: int(np.sum(original == i)) for i in range(3)})
    print("Distribución NumPy:   ", {NAMES[i]: int(np.sum(numpy_predictions == i)) for i in range(3)})

    if "label" in df.columns:
        y_series = df["label"].astype(str).str.strip().str.lower().map(LABELS)
        if y_series.isna().any():
            unknown = sorted(df.loc[y_series.isna(), "label"].astype(str).unique())
            print(f"\nNo se pudo evaluar accuracy: etiquetas no reconocidas: {unknown}")
        else:
            y = y_series.to_numpy(dtype=int)
            print(f"Accuracy VQC.predict(): {100*np.mean(original == y):.2f}%")
            print(f"Accuracy inferencia NumPy: {100*np.mean(numpy_predictions == y):.2f}%")

    if len(mismatches):
        print("\nPrimeras diferencias (fila CSV empieza en 2 por la cabecera):")
        for i in mismatches[:12]:
            label = str(df.iloc[i]["label"]) if "label" in df.columns else "(sin label)"
            print(f"fila {i+2}: real={label}, original={NAMES.get(int(original[i]), original[i])}, NumPy={NAMES.get(int(numpy_predictions[i]), numpy_predictions[i])}, ángulos={np.round(X[i], 4).tolist()}")
    else:
        print("\nLas dos rutas coinciden en todas las muestras del archivo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Diagnosticar la precisión y latencia del VQC

Antes de volver a entrenar el modelo, hay que comprobar que la inferencia NumPy de Cloud Run produce la misma clase que el `VQC.predict()` original. El endpoint se cambió a una simulación de estadovector NumPy para evitar un `SIGSEGV`; que ambas rutas utilicen el mismo circuito no basta para demostrar que las probabilidades están implementadas de forma idéntica.

## 1. Preparar

Descarga `diagnosticar_vqc.py` y guárdalo en:

`C:\Users\qucit\Downloads\quantum_arcade_cloud\backend\diagnosticar_vqc.py`

Asegúrate de que el entorno local tiene las mismas dependencias principales instaladas (`qiskit`, `qiskit-machine-learning`, `qiskit-aer`, `numpy`, `pandas`) y conserva el `.dill` en `backend\models`.

## 2. Ejecutar

Desde PowerShell:

```powershell
cd C:\Users\qucit\Downloads\quantum_arcade_cloud\backend
python diagnosticar_vqc.py ..\gestos_test_3.csv
```

Cambia el CSV por el conjunto etiquetado que realmente se reservó para evaluar el modelo. No subas CSV que contengan datos personales a GitHub.

## 3. Interpretar

- Si `Predicciones distintas` es 0 %, las dos rutas coinciden en ese archivo; la confusión de piedra/papel probablemente se debe a cómo los datos de cámara difieren de los datos de entrenamiento o al rendimiento del propio modelo.
- Si hay diferencias, conserva el resultado: confirma que el reemplazo de `VQC.predict()` altera la clasificación y permite corregir el simulador NumPy antes de ajustar la suavización.
- Si `VQC.predict()` da SIGSEGV localmente, detén la prueba y comparte el error; no ejecutes el script repetidamente.

Este diagnóstico no modifica el modelo ni sus pesos.

# Parche para SIGSEGV al inferir con el VQC y ocultar landmarks

## Qué cambia

- `backend/main.py`: reconstruye el `QMLSampler(shots=None)` dentro de Cloud Run al cargar el `.dill`, añade un bloqueo para que el mismo VQC no reciba inferencias concurrentes y registra el sampler activo.
- `backend/Dockerfile`: limita los hilos numéricos nativos y usa un solo worker síncrono de Gunicorn para el VQC.
- `frontend/src/main.js`, `frontend/index.html`, `frontend/src/style.css`: elimina el dibujo de puntos y conexiones de MediaPipe sin quitar el cálculo de los cinco ángulos.

## Aplicación

1. Descomprime este ZIP en la raíz `C:\Users\qucit\Downloads\quantum-arcade-cloud`, conservando las carpetas `backend` y `frontend`; acepta reemplazar estos archivos.
2. Despliega el backend desde la carpeta `backend`:

```powershell
gcloud run deploy quantum-arcade-api --source . --region us-central1 --allow-unauthenticated --service-account quantum-arcade-runtime@quantum-rock-paper-scissors.iam.gserviceaccount.com --cpu 1 --memory 2Gi --max 2 --concurrency 1 --timeout 120 --set-env-vars "USE_FIRESTORE=true,VQC_MODEL_PATH=models/vqc_piedra_papel_tijera_seed_0.dill,TOTAL_ROUNDS=15,ALLOWED_ORIGINS=https://quantum-arcade-cloud.pages.dev"
```

3. En la raíz del repositorio, publicar el frontend:

```powershell
cd ..
git add frontend/src/main.js frontend/index.html frontend/src/style.css
git commit -m "Hide hand landmarks in webcam view"
git push
```

Cloudflare Pages reconstruirá la web automáticamente.

4. Probar primero `/health`, después una llamada POST a `/api/predict`. Confirmar en logs de Cloud Run que aparece `runtime sampler reset to QMLSampler(shots=None)` y que no vuelve a aparecer `SIGSEGV`.

Si el `SIGSEGV` persiste aun con el sampler recién creado, el siguiente paso es dejar de serializar el objeto VQC completo y exportar únicamente sus circuitos, pesos e interpretación, reconstruyendo el clasificador dentro del contenedor.

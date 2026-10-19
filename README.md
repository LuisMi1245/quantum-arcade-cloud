# Quantum Arcade — Cloud deployment

Consulta [`README_DEPLOY.md`](README_DEPLOY.md) para el procedimiento completo de despliegue en Cloudflare Pages y Google Cloud Run.

- `frontend/`: Vite + MediaPipe en el navegador.
- `backend/`: Flask + Qiskit Aer + VQC `.dill` + Firestore.

Antes de desplegar se copia `hand_landmarker.task` a `frontend/public/models/` y `vqc_piedra_papel_tijera_seed_0.dill` a `backend/models/`.

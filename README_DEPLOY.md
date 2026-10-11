# Quantum Arcade — despliegue público en Cloudflare Pages + Google Cloud Run

Esta carpeta separa la aplicación en dos partes:

- `frontend/`: HTML/CSS/JavaScript/Vite. MediaPipe se ejecuta en el navegador de cada jugador. La cámara nunca se transmite al servidor.
- `backend/`: Flask en Cloud Run. Recibe únicamente cinco ángulos en radianes para que el VQC entrenado haga la clasificación y administra las partidas con el motor cuántico `QuantumPrisonersDilemma`.

## Archivos que debes aportar tú

Este ZIP no puede incluir tus dos modelos, porque no fueron adjuntados como archivos a esta conversación:

1. Copia tu modelo público de MediaPipe `hand_landmarker.task` a `frontend/public/models/hand_landmarker.task`.
2. Copia tu modelo entrenado `vqc_piedra_papel_tijera_seed_0.dill` a `backend/models/vqc_piedra_papel_tijera_seed_0.dill`.

El archivo `.task` se publicará junto al frontend (el límite individual de Cloudflare Pages es 25 MiB). El archivo `.dill` se incluirá en la imagen de Cloud Run desde tu ordenador, pero está excluido del GitHub mediante `.gitignore`: no lo subas a un repositorio público.

## 0. Requisitos

- Cuenta de GitHub.
- Cuenta de Cloudflare.
- Cuenta de Google Cloud con facturación habilitada. El nivel gratuito no es una garantía de coste cero: vigila los límites y configura alertas de presupuesto.
- Node.js 20 o 22 y npm para probar/construir el frontend.
- Google Cloud CLI (`gcloud`) para desplegar el backend.

## 1. Preparar los modelos

Copia los dos modelos como se indica arriba. No renombres el `.dill`, salvo que también cambies `VQC_MODEL_PATH`.

### Comprueba la compatibilidad del `.dill`

En el entorno donde entrenaste el VQC, ejecuta:

```bash
python -c "import qiskit, qiskit_machine_learning, qiskit_aer; print('qiskit', qiskit.__version__); print('qiskit-machine-learning', qiskit_machine_learning.__version__); print('qiskit-aer', qiskit_aer.__version__)"
```

El `requirements.txt` del backend fija versiones actuales compatibles con Qiskit 2.x. Si tu modelo fue guardado con otras versiones, ajusta esas dependencias a las versiones con las que `VQC.from_dill(...)` funciona en tu ordenador. La deserialización `.dill` depende de la compatibilidad de las bibliotecas.

## 2. Probar el backend localmente (recomendado antes de desplegar)

Abre una terminal en `backend/`.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:USE_FIRESTORE = "false"
$env:VQC_MODEL_PATH = "models/vqc_piedra_papel_tijera_seed_0.dill"
$env:ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
python main.py
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export USE_FIRESTORE=false
export VQC_MODEL_PATH=models/vqc_piedra_papel_tijera_seed_0.dill
export ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
python main.py
```

Debe escuchar en `http://127.0.0.1:8080`. Abre otra terminal y prueba `http://127.0.0.1:8080/health`. Para poder probar la cámara/VQC, `vqc_loaded` debe ser `true`.

## 3. Probar el frontend localmente

En otra terminal, ve a `frontend/` y ejecuta:

```bash
npm install
```

Crea `frontend/.env.local` con:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8080
```

Ejecuta el desarrollo:

```bash
npm run dev
```

Abre la dirección local que muestre Vite (normalmente `http://localhost:5173`). Pulsa START, permite el uso de la cámara y prueba cada gesto. Si la cámara o el VQC no están disponibles, los botones manuales de piedra/papel/tijeras siguen presentes.

El build script copia los WASM de `@mediapipe/tasks-vision` al directorio `public/mediapipe/wasm`, así que el sitio no depende de una CDN de MediaPipe durante su uso. El modelo se carga desde `/models/hand_landmarker.task`.

## 4. Crear Google Cloud Project y Firestore

1. Abre Google Cloud Console y crea un proyecto nuevo, por ejemplo `quantum-arcade-public`.
2. Vincula una cuenta de facturación. Cloud Run requiere facturación aunque puedes mantener un uso pequeño dentro de las cuotas gratuitas; no existe una garantía absoluta de que nunca se generen costes.
3. Instala la Google Cloud CLI e inicia sesión:

```bash
gcloud auth login
gcloud config set project TU_PROJECT_ID
```

4. Habilita los servicios:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com
```

5. En la consola de Google Cloud, abre **Firestore → Create database**. Selecciona **Native mode**, la base de datos predeterminada `(default)` y una región cercana a Cloud Run. Para el ejemplo, usa `us-central1`.
6. Crea una cuenta de servicio para Cloud Run y dale acceso a Firestore:

```bash
gcloud iam service-accounts create quantum-arcade-runtime --display-name="Quantum Arcade Cloud Run"
gcloud projects add-iam-policy-binding TU_PROJECT_ID --member="serviceAccount:quantum-arcade-runtime@TU_PROJECT_ID.iam.gserviceaccount.com" --role="roles/datastore.user"
```

Sustituye `TU_PROJECT_ID` por el ID exacto del proyecto en ambos comandos. No descargues ni subas claves JSON; Cloud Run usa la identidad asignada al servicio.

## 5. Desplegar Python a Cloud Run

**Importante:** despliega desde tu carpeta local `backend/`, donde sí está el `.dill`. El archivo `.gcloudignore` excluye cachés y entornos virtuales, pero permite enviar el modelo a Cloud Build aunque `.gitignore` lo excluya de GitHub.

```bash
cd backend
gcloud run deploy quantum-arcade-api --source . --region us-central1 --allow-unauthenticated --service-account quantum-arcade-runtime@TU_PROJECT_ID.iam.gserviceaccount.com --cpu 1 --memory 2Gi --max 2 --timeout 120 --set-env-vars "USE_FIRESTORE=true,VQC_MODEL_PATH=models/vqc_piedra_papel_tijera_seed_0.dill,TOTAL_ROUNDS=15"
```

Acepte los avisos de habilitación de APIs si aparecen. Al finalizar, Google mostrará una URL similar a `https://quantum-arcade-api-....run.app`. Guárdala como `API_URL`.

Comprueba `https://TU-URL.run.app/health`. Esperas `ok: true`, `firestore_enabled: true` y `vqc_loaded: true`. Si `vqc_loaded` es `false`, revisa los logs de Cloud Run y comprueba que el `.dill` se incluyó en la carpeta local antes de desplegar; puede que las versiones de Qiskit del `requirements.txt` deban coincidir con las del entrenamiento.

`--max 2` limita el crecimiento ordinario de instancias, pero puede afectar a la disponibilidad si muchas personas juegan a la vez. Con `min instances=0`, puede haber una pausa de arranque en frío tras un periodo sin tráfico.

## 6. Publicar el frontend en GitHub

1. Crea un repositorio nuevo en GitHub. Puedes mantenerlo público para el hackathon.
2. Copia el archivo `hand_landmarker.task` a `frontend/public/models/hand_landmarker.task` antes de subir el proyecto. Es un archivo de modelo público de MediaPipe y debe estar en el despliegue estático para que el navegador lo descargue.
3. No copies el `.dill` a GitHub. La regla del `.gitignore` excluye `backend/models/*.dill`.
4. Desde la carpeta raíz del proyecto (un nivel encima de `frontend/` y `backend/`):

```bash
git init
git add .
git status
```

**Antes de continuar, revisa `git status` y comprueba que no aparezca `vqc_piedra_papel_tijera_seed_0.dill` ni ninguna credencial.** Luego:

```bash
git commit -m "Prepare Quantum Arcade for cloud deployment"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

El backend se despliega desde tu carpeta local, no desde el repositorio de GitHub, para mantener el `.dill` fuera del repositorio público.

## 7. Crear Cloudflare Pages

1. En Cloudflare Dashboard, abre **Workers & Pages → Create → Pages → Connect to Git**.
2. Selecciona el repositorio de GitHub.
3. Configura:
   - **Root directory:** `frontend`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Node.js version:** `22` (variable de entorno de build `NODE_VERSION=22`, si la pantalla la solicita).
4. En **Environment variables** añade `VITE_API_BASE_URL` con la URL completa de Cloud Run, sin `/` final, por ejemplo `https://quantum-arcade-api-xxxx-uc.a.run.app`.
5. Despliega y espera a que Pages muestre la URL pública `https://TU-PROYECTO.pages.dev`.

En Cloudflare Pages, el modelo `hand_landmarker.task` es un asset estático. El máximo actual por archivo de Pages es 25 MiB, así que este modelo habitual de unos pocos MiB se mantiene por debajo del límite.

## 8. Restringir CORS a la página real

Copia el host Pages completo, por ejemplo `https://quantum-arcade.pages.dev`, y configúralo en Cloud Run para que solo esa página pueda llamar a la API desde un navegador:

```bash
gcloud run services update quantum-arcade-api --region us-central1 --update-env-vars "ALLOWED_ORIGINS=https://TU-PROYECTO.pages.dev"
```

Si utilizas un dominio personalizado, inclúyelo también, separado por coma. Las peticiones con `Origin` distinto no recibirán cabeceras CORS. Los comandos de consola como `curl` no dependen de CORS.

## 9. Prueba final

1. Abre la URL de Cloudflare Pages.
2. Pulsa `START MATCH`.
3. Activa `CAMERA: ON` y acepta el permiso de cámara. La página debe estar en HTTPS.
4. Mantén piedra, papel o tijeras hasta que el clasificador estabilice el gesto. Retira la mano entre jugadas para rearmar el detector.
5. Comprueba el acuerdo, los puntos, la medición de Q-BOT, las actualizaciones de estrategia y la adivinanza final.
6. Prueba también los botones manuales si la webcam no está disponible.
7. Abre la consola del navegador y revisa Network si algún endpoint devuelve errores CORS/4xx/5xx.

## 10. Costes y apagado tras el evento

Cloudflare Pages ofrece alojamiento estático gratuito dentro de sus límites. Cloud Run tiene un nivel gratuito con límites y requiere que exista una cuenta de facturación. Configura alertas de presupuesto (por ejemplo, un límite de aviso de 1 USD y otro de 5 USD) y revisa el panel de Billing. Las alertas avisan; no son un tope rígido de consumo.

Al finalizar las dos semanas, si ya no necesitas el backend, ejecuta:

```bash
gcloud run services delete quantum-arcade-api --region us-central1
```

Si ya no necesitas los registros de partidas, puedes eliminar la base de Firestore desde la consola. No elimines un proyecto o base de datos que contenga otros recursos importantes.

## Qué hace el sistema en la nube

- **Vídeo:** permanece en el navegador del visitante; no se transmite al backend.
- **MediaPipe:** detecta los 21 landmarks en JavaScript con los archivos WASM servidos por Cloudflare Pages.
- **Características:** la página calcula los cinco ángulos 3D en radianes en el mismo orden del recolector original.
- **VQC:** solo envía el vector de cinco ángulos a Cloud Run; Python ejecuta `VQC.predict()` con el `.dill` cargado.
- **Juego:** Firestore mantiene el estado individual de cada partida, incluida la personalidad secreta de Q-BOT. La personalidad no se devuelve al navegador hasta que el usuario hace su intento final.
- **Máquina cuántica:** este backend usa `qiskit-aer` para el circuito estratégico de dos qubits; es simulación local en Cloud Run, no hardware cuántico físico ni un backend remoto de IonQ.

## Limitaciones honestas

- No existe una garantía de disponibilidad continua con un plan gratuito.
- Cloud Run puede escalar a cero y presentar un cold start.
- La cámara solo funciona si el usuario concede permiso y el navegador soporta `getUserMedia`.
- Si el `.dill` fue serializado con versiones incompatibles, el servicio permitirá jugar manualmente, pero la inferencia VQC no estará disponible hasta corregir las dependencias.
- El API es pública para permitir jugar sin inicio de sesión. El identificador de sesión aleatorio separa las partidas, pero esto no equivale a un sistema anti-trampas de producción.

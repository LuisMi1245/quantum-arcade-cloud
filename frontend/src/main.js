import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './style.css';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080').replace(/\/$/, '');
const MODEL_URL = `${import.meta.env.BASE_URL}models/hand_landmarker.task`;
const WASM_URL = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const TOTAL_ROUNDS = 15;

const MOVES = {
  rock: { label: 'ROCK', emoji: '✊', spanish: 'piedra' },
  paper: { label: 'PAPER', emoji: '✋', spanish: 'papel' },
  scissors: { label: 'SCISSORS', emoji: '✌', spanish: 'tijeras' },
};
const SPANISH_TO_MOVE = { piedra: 'rock', papel: 'paper', tijeras: 'scissors', tijera: 'scissors' };
const PERSONALITIES = { indiferente: 'INDIFERENTE', reciproco: 'RECÍPROCA', rencoroso: 'RENCOROSA' };
const $ = (id) => document.getElementById(id);

let sessionId = sessionStorage.getItem('qaSessionId');
if (!sessionId) {
  sessionId = crypto.randomUUID();
  sessionStorage.setItem('qaSessionId', sessionId);
}

let gameState = null;
let roundBusy = false;
let audioContext = null;
let cameraStream = null;
let handLandmarker = null;
let cameraOn = false;
let inferenceEnabled = false;
let rafId = 0;
let lastVideoTime = -1;
let lastPredictionAt = 0;
let lastDetectionAt = 0;
let predictionPending = false;
let predictionHistory = [];
let gestureArmed = true;
let noHandFrames = 0;
let vqcAvailable = false;

function setIntegrationStatus(message) {
  const node = $('integrationStatus');
  if (node) node.textContent = message;
}

function setIntegrationDetails(message) {
  const node = $('integrationDetails');
  if (node) node.textContent = message;
}

function setScreen(id) {
  document.querySelectorAll('.screen').forEach((screen) => screen.classList.remove('active'));
  $(id).classList.add('active');
  window.scrollTo(0, 0);
}

async function api(path, payload = undefined, method = 'POST') {
  const options = { method, headers: { 'Content-Type': 'application/json' }, cache: 'no-store' };
  if (payload !== undefined) options.body = JSON.stringify(payload);
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function unlockAudio() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  if (!audioContext) audioContext = new AudioContextClass();
  if (audioContext.state === 'suspended') audioContext.resume().catch(() => {});
}

function playRoundSound() {
  unlockAudio();
  if (!audioContext || audioContext.state !== 'running') return;
  const now = audioContext.currentTime;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = 'square';
  oscillator.frequency.setValueAtTime(740, now);
  oscillator.frequency.exponentialRampToValueAtTime(520, now + 0.085);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.045, now + 0.008);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.11);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start(now);
  oscillator.stop(now + 0.12);
}

function moveKey(value) {
  if (!value) return null;
  const normalized = String(value).trim().toLowerCase();
  if (MOVES[normalized]) return normalized;
  return SPANISH_TO_MOVE[normalized] || null;
}

function moveLabel(value) {
  const key = moveKey(value);
  return key ? MOVES[key].label : String(value || '—').toUpperCase();
}

function setMoveProbabilities(move) {
  const key = moveKey(move);
  $('pr').textContent = key === 'rock' ? '100%' : key ? '0%' : '33%';
  $('pp').textContent = key === 'paper' ? '100%' : key ? '0%' : '33%';
  $('pt').textContent = key === 'scissors' ? '100%' : key ? '0%' : '33%';
}

function winnerClass(winner, forBot = false) {
  if (winner === 'Gana jugador') return forBot ? 'lost' : 'won';
  if (winner === 'Gana máquina') return forBot ? 'won' : 'lost';
  return 'draw';
}

function renderHistory(state) {
  const history = [...(state.history || [])].reverse();
  $('hist').innerHTML = history.length ? history.map((row) => {
    const playerMove = moveKey(row.player_move);
    const botMove = moveKey(row.machine_move);
    const resultLabel = row.winner === 'Gana jugador' ? 'YOU WIN'
      : row.winner === 'Gana máquina' ? 'Q-Bot Wins' : 'DRAW';
    const cls = winnerClass(row.winner);
    return `<tr><td>ROUND ${String(row.round).padStart(2, '0')}</td><td>${playerMove ? MOVES[playerMove].emoji : '—'} vs ${botMove ? MOVES[botMove].emoji : '—'}</td><td class="${cls}">${resultLabel}</td></tr>`;
  }).join('') : '';

  const dotsFor = (forBot) => Array.from({ length: state.total_rounds || TOTAL_ROUNDS }, (_, i) => {
    const row = state.history?.[i];
    if (!row) return '<span class="dot" title="Ronda sin jugar"></span>';
    const cls = winnerClass(row.winner, forBot);
    const label = row.winner === 'Gana jugador' ? (forBot ? 'Derrota de Q-Bot' : 'Victoria del jugador')
      : row.winner === 'Gana máquina' ? (forBot ? 'Victoria de Q-Bot' : 'Derrota del jugador') : 'Empate';
    return `<span class="dot ${cls === 'won' ? 'win' : cls === 'lost' ? 'loss' : 'tie'}" title="Ronda ${i + 1}: ${label}"></span>`;
  }).join('');
  $('pdots').innerHTML = dotsFor(false);
  $('bdots').innerHTML = dotsFor(true);
}

function updateEnding(state) {
  const playerScore = Number(state?.scores?.player || 0);
  const botScore = Number(state?.scores?.machine || 0);
  const title = playerScore > botScore ? 'CONGRATULATIONS!<br>YOU WIN!'
    : playerScore < botScore ? 'BETTER LUCK<br>NEXT TIME' : 'IT\'S A TIE!<br>GOOD GAME!';
  $('endtitle').innerHTML = title;
  const completed = Number(state?.rounds_completed ?? state?.history?.length ?? 0);
  $('endscore').innerHTML = `PLAYER 1: ${playerScore}<br>Q-BOT: ${botScore}<br><small>${completed} ROUNDS COMPLETE</small>`;
}

function renderState(state) {
  gameState = state;
  $('ps').textContent = state.scores?.player ?? 0;
  $('bs').textContent = state.scores?.machine ?? 0;

  const total = state.total_rounds || TOTAL_ROUNDS;
  const agreement = state.current_agreement;
  if (state.game_over) {
    $('roundlabel').textContent = `ROUND ${total} / ${total} — MATCH COMPLETE`;
  } else {
    const pactLabel = agreement ? moveLabel(agreement) : '—';
    $('roundlabel').textContent = `ROUND ${state.round_number || 1} / ${total} — PACT: ${pactLabel} · Choose your move`;
  }

  const last = state.last_record;
  if (last) {
    const playerKey = moveKey(last.player_move);
    const botKey = moveKey(last.machine_move);
    $('state').textContent = `${moveLabel(last.player_move)} / ${String(last.player_state || '').toUpperCase()}`;
    $('botstate').textContent = `${String(last.machine_state || '').toUpperCase()} · ${moveLabel(last.machine_move)}`;
    document.querySelectorAll('[data-m]').forEach((button) => {
      button.classList.toggle('selected', button.dataset.m === playerKey);
    });
    setIntegrationDetails(
      `Pacto de esta ronda: ${moveLabel(last.agreement)}. Última medición cuántica: |${last.quantum_measurement}⟩. ` +
      `Jugada Q-BOT: ${botKey ? MOVES[botKey].label : '—'}. Puntos de la ronda: ${last.player_points} / ${last.machine_points}.`
    );
  } else {
    $('state').textContent = 'READY';
    $('botstate').textContent = 'ACTIVE / THINKING';
    setIntegrationDetails(agreement
      ? `Acuerdo de la ronda actual: ${moveLabel(agreement)}. Retira la mano después de una jugada para habilitar la siguiente.`
      : 'La cámara se procesa localmente. Los cinco ángulos se envían al backend para la clasificación VQC.');
  }

  renderHistory(state);
  document.querySelectorAll('[data-m]').forEach((button) => {
    button.disabled = Boolean(state.game_over) || roundBusy;
  });

  if (state.game_over) {
    updateEnding(state);
    setScreen('ending');
    if (cameraOn) stopCamera();
  }
}

async function startNewGame() {
  unlockAudio();
  if (cameraOn || cameraStream) stopCamera();
  roundBusy = false;
  gestureArmed = true;
  predictionHistory = [];
  predictionPending = false;
  setMoveProbabilities(null);
  document.querySelectorAll('[data-m]').forEach((button) => {
    button.classList.remove('selected');
    button.disabled = false;
  });
  try {
    const result = await api('/api/game/reset', { session_id: sessionId });
    renderState(result.state);
    setScreen('game');
    setIntegrationStatus(vqcAvailable
      ? 'Gesture model: VQC loaded · Qiskit backend: connected.'
      : 'Gesture model: unavailable · Qiskit backend: connected; manual play is available.');
  } catch (error) {
    setIntegrationStatus(`No se pudo iniciar la partida: ${error.message}`);
    setIntegrationDetails('Comprueba la API de Cloud Run y vuelve a pulsar START.');
  }
}

async function submitMove(move, source = 'buttons') {
  if (!MOVES[move] || roundBusy || !gameState || gameState.game_over) return;
  roundBusy = true;
  gestureArmed = false;
  document.querySelectorAll('[data-m]').forEach((button) => { button.disabled = true; });
  $('state').textContent = source === 'camera' ? 'GESTURE CONFIRMED' : 'PROCESSING...';
  $('botstate').textContent = 'ACTIVE / THINKING';
  setIntegrationStatus(source === 'camera'
    ? `VQC detected ${MOVES[move].label}; resolving quantum round…`
    : 'Jugada manual confirmada; resolviendo la ronda cuántica…');

  try {
    const result = await api('/api/game/round', { session_id: sessionId, player_move: move });
    playRoundSound();
    setMoveProbabilities(move);
    renderState(result.state);
    const record = result.record;
    if (!result.state.game_over) {
      setIntegrationStatus(vqcAvailable
        ? `Backend connected · VQC ready · Round ${record.round} complete.`
        : `Backend connected · Round ${record.round} complete; manual play remains available.`);
    }
  } catch (error) {
    setIntegrationStatus(`No se pudo resolver la ronda: ${error.message}`);
    setIntegrationDetails('Puedes seguir utilizando los botones manuales cuando la conexión se restablezca.');
  } finally {
    roundBusy = false;
    if (gameState) renderState(gameState);
  }
}

function calculateAngle(p1, p2, p3) {
  const v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z];
  const v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z];
  const dot = v1.reduce((sum, value, i) => sum + value * v2[i], 0);
  const n1 = Math.hypot(...v1);
  const n2 = Math.hypot(...v2);
  if (n1 < 1e-9 || n2 < 1e-9) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (n1 * n2))));
}

function extractAngles(landmarks) {
  const triplets = [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]];
  return triplets.map(([a, b, c]) => calculateAngle(landmarks[a], landmarks[b], landmarks[c]));
}

function majorityPrediction() {
  const counts = new Map();
  predictionHistory.forEach((move) => counts.set(move, (counts.get(move) || 0) + 1));
  let best = null;
  let votes = 0;
  for (const [move, count] of counts.entries()) {
    if (count > votes) { best = move; votes = count; }
  }
  return { move: best, votes };
}

async function requestVqcPrediction(angles) {
  predictionPending = true;
  try {
    const result = await api('/api/predict', { angles });
    const move = result.gesture;
    if (!MOVES[move]) throw new Error('El backend devolvió una clase inválida.');
    setMoveProbabilities(move);
    predictionHistory.push(move);
    if (predictionHistory.length > 9) predictionHistory.shift();
    const { move: stable, votes } = majorityPrediction();
    if (stable && votes >= 5) {
      $('state').textContent = `${MOVES[stable].label} · STABLE ${votes}/9`;
    } else {
      $('state').textContent = `${MOVES[move].label} · STABILIZING`;
    }
    setIntegrationStatus(`Camera active · VQC: ${MOVES[move].label} · ${predictionHistory.length}/9 votes collected.`);

    if (stable && votes >= 5 && gestureArmed && $('game').classList.contains('active') && !roundBusy && !gameState?.game_over) {
      gestureArmed = false;
      await submitMove(stable, 'camera');
    }
  } catch (error) {
    inferenceEnabled = false;
    $('state').textContent = 'VQC ERROR';
    setIntegrationStatus(`VQC request failed: ${error.message}. Use the manual buttons.`);
    setIntegrationDetails('The camera remains local; only the five numerical angles are sent to the backend.');
  } finally {
    predictionPending = false;
  }
}

function detectionLoop() {
  if (!cameraOn || !handLandmarker) return;
  const video = $('gestureVideo');
  const now = performance.now();
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime && now - lastDetectionAt >= 66) {
    lastVideoTime = video.currentTime;
    lastDetectionAt = now;
    try {
      const result = handLandmarker.detectForVideo(video, now);
      const landmarks = result.landmarks?.[0] || null;
      if (landmarks) {
        noHandFrames = 0;
        if (inferenceEnabled && !predictionPending && now - lastPredictionAt >= 200) {
          lastPredictionAt = now;
          requestVqcPrediction(extractAngles(landmarks));
        }
      } else {
        noHandFrames += 1;
        predictionHistory = [];
        if (noHandFrames >= 8) gestureArmed = true;
        $('state').textContent = 'NO HAND DETECTED';
      }
    } catch (error) {
      setIntegrationStatus(`MediaPipe error: ${error.message}. Manual buttons remain available.`);
    }
  }
  rafId = requestAnimationFrame(detectionLoop);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setIntegrationStatus('This browser cannot access the camera here. Use HTTPS or the manual buttons.');
    return;
  }
  $('cameraToggle').disabled = true;
  setIntegrationStatus('Requesting camera permission…');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    const video = $('gestureVideo');
    video.srcObject = cameraStream;
    video.style.display = 'block';
    await video.play();

    setIntegrationStatus('Loading MediaPipe hand detector in the browser…');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    cameraOn = true;
    inferenceEnabled = vqcAvailable;
    $('cameraToggle').textContent = 'CAMERA: ON';
    setIntegrationStatus(vqcAvailable
      ? 'Camera active · VQC ready. Hold a gesture steady; remove your hand to play again.'
      : 'Camera active, but the VQC is unavailable. You can still use the manual buttons.');
    setIntegrationDetails('The video is processed in your browser. Only the five angles are sent to Cloud Run for VQC classification.');
    lastVideoTime = -1;
    lastDetectionAt = 0;
    lastPredictionAt = 0;
    predictionHistory = [];
    predictionPending = false;
    gestureArmed = true;
    noHandFrames = 0;
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(detectionLoop);
  } catch (error) {
    stopCamera();
    setIntegrationStatus(`Could not start camera/MediaPipe: ${error.message || error}.`);
    setIntegrationDetails('Check camera permissions and make sure the model files were included in the frontend deployment.');
  } finally {
    $('cameraToggle').disabled = false;
  }
}

function stopCamera() {
  cameraOn = false;
  cancelAnimationFrame(rafId);
  rafId = 0;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  const video = $('gestureVideo');
  if (video) {
    video.pause();
    video.srcObject = null;
    video.style.display = 'none';
  }
  if (handLandmarker) {
    try { handLandmarker.close(); } catch (_) { /* detector is already closed */ }
  }
  handLandmarker = null;
  $('cameraToggle').textContent = 'CAMERA: OFF';
  setIntegrationStatus(vqcAvailable
    ? 'Camera off · VQC ready · manual buttons available.'
    : 'Camera off · VQC unavailable · manual buttons available.');
  predictionHistory = [];
  predictionPending = false;
  gestureArmed = true;
  noHandFrames = 0;
}

function returnToIntro() {
  if (cameraOn || cameraStream) stopCamera();
  gameState = null;
  roundBusy = false;
  predictionHistory = [];
  setMoveProbabilities(null);
  $('ps').textContent = '0';
  $('bs').textContent = '0';
  $('state').textContent = 'READY';
  $('botstate').textContent = 'ACTIVE / THINKING';
  $('roundlabel').textContent = 'ROUND 1 / 15 — Choose your move';
  $('hist').innerHTML = '';
  $('pdots').innerHTML = '';
  $('bdots').innerHTML = '';
  document.querySelectorAll('[data-m]').forEach((button) => {
    button.classList.remove('selected');
    button.disabled = false;
  });
  $('endtitle').innerHTML = 'CONGRATULATIONS!<br>YOU WIN!';
  $('endscore').textContent = '';
  setIntegrationStatus('Gesture model: not connected · Qiskit backend: demo mode');
  setIntegrationDetails('La cámara se procesa localmente. Los ángulos se envían al backend para la clasificación VQC.');
  setScreen('intro');
}

$('start').addEventListener('click', startNewGame);
$('restart').addEventListener('click', returnToIntro);
document.querySelectorAll('[data-m]').forEach((button) => {
  button.addEventListener('click', () => submitMove(button.dataset.m, 'buttons'));
});
$('cameraToggle').addEventListener('click', () => cameraOn ? stopCamera() : startCamera());
$('end').addEventListener('click', () => {
  if (!gameState) { setScreen('ending'); return; }
  updateEnding(gameState);
  if (cameraOn || cameraStream) stopCamera();
  setScreen('ending');
});
window.addEventListener('beforeunload', stopCamera);

async function boot() {
  try {
    const health = await api('/health', undefined, 'GET');
    vqcAvailable = Boolean(health.vqc_loaded);
    inferenceEnabled = vqcAvailable;
    setIntegrationStatus(vqcAvailable
      ? 'Gesture model: VQC loaded · Qiskit backend: connected.'
      : 'Gesture model: not loaded · Qiskit backend: connected; manual play available.');
    setIntegrationDetails(vqcAvailable
      ? 'Ready. Enable the camera to recognize gestures or use the three manual buttons.'
      : 'Add the trained .dill model to Cloud Run to enable camera classification.');
  } catch (error) {
    vqcAvailable = false;
    inferenceEnabled = false;
    setIntegrationStatus(`API offline: ${error.message}. Check VITE_API_BASE_URL in Cloudflare Pages.`);
    setIntegrationDetails('The game requires the Cloud Run API; check the service URL and its availability.');
  }
}

setMoveProbabilities(null);
boot();

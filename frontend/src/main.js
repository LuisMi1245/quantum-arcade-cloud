import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './style.css';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080').replace(/\/$/, '');
const MODEL_URL = `${import.meta.env.BASE_URL}models/hand_landmarker.task`;
const WASM_URL = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const TOTAL_ROUNDS = 15;
const MOVES = {
  rock: { label: 'ROCK', symbol: 'piedra', emoji: '✊', classId: 0 },
  paper: { label: 'PAPER', symbol: 'papel', emoji: '✋', classId: 1 },
  scissors: { label: 'SCISSORS', symbol: 'tijeras', emoji: '✌', classId: 2 },
};
const SYMBOLS = {
  piedra: { label: 'ROCK', emoji: '✊' },
  papel: { label: 'PAPER', emoji: '✋' },
  tijeras: { label: 'SCISSORS', emoji: '✌' },
};
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
let inferenceEnabled = true;
let rafId = 0;
let lastVideoTime = -1;
let lastPredictionAt = 0;
let predictionPending = false;
let predictionHistory = [];
let gestureArmed = true;
let noHandFrames = 0;
let cameraErrorShown = false;
let vqcAvailable = false;
let lastDetectionAt = 0;

function setNotice(message) { $('notice').textContent = message; }
function setCameraStatus(message) { $('cameraStatus').textContent = message; }
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
  gain.gain.exponentialRampToValueAtTime(0.035, now + 0.008);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.11);
  oscillator.connect(gain); gain.connect(audioContext.destination);
  oscillator.start(now); oscillator.stop(now + 0.12);
}

function renderState(state) {
  gameState = state;
  $('ps').textContent = state.scores?.player ?? 0;
  $('bs').textContent = state.scores?.machine ?? 0;
  $('roundLabel').textContent = state.game_over
    ? `ROUND ${state.total_rounds} / ${state.total_rounds} — MATCH COMPLETE`
    : `ROUND ${state.round_number} / ${state.total_rounds} — ELIGE TU JUGADA PARA EL ACUERDO ACTUAL`;

  const agreement = state.current_agreement;
  if (agreement && SYMBOLS[agreement]) {
    const sym = SYMBOLS[agreement];
    $('agreementEmoji').textContent = sym.emoji;
    $('agreementName').textContent = sym.label;
    $('agreementRound').textContent = `RONDA ${state.round_number}/${state.total_rounds} · acuerdo aleatorio válido para esta ronda`;
    const beats = { piedra: 'papel', papel: 'tijeras', tijeras: 'piedra' };
    const loses = { piedra: 'tijeras', papel: 'piedra', tijeras: 'papel' };
    $('agreementActions').innerHTML = `
      <div class="action-chip coop"><strong>COOPERAR</strong><span>${SYMBOLS[agreement].emoji} ${SYMBOLS[agreement].label}<br>Igualar el acuerdo</span></div>
      <div class="action-chip betray"><strong>TRAICIONAR</strong><span>${SYMBOLS[beats[agreement]].emoji} ${SYMBOLS[beats[agreement]].label}<br>Vencer al acuerdo</span></div>
      <div class="action-chip neutral"><strong>NEUTRAL / NO DECISORIA</strong><span>${SYMBOLS[loses[agreement]].emoji} ${SYMBOLS[loses[agreement]].label}<br>Perder ante el acuerdo</span></div>`;
  } else {
    $('agreementEmoji').textContent = '✓';
    $('agreementName').textContent = 'ACUERDOS COMPLETADOS';
    $('agreementRound').textContent = 'Adivina la personalidad secreta de Q-BOT.';
  }

  const probabilities = state.strategy_probabilities || { cooperation: 1/3, betrayal: 1/3, no_decision: 1/3 };
  $('pcBar').style.width = `${100 * probabilities.cooperation}%`;
  $('ptBar').style.width = `${100 * probabilities.betrayal}%`;
  $('pnBar').style.width = `${100 * probabilities.no_decision}%`;
  $('pcPct').textContent = `${(100 * probabilities.cooperation).toFixed(1)}%`;
  $('ptPct').textContent = `${(100 * probabilities.betrayal).toFixed(1)}%`;
  $('pnPct').textContent = `${(100 * probabilities.no_decision).toFixed(1)}%`;

  const last = state.last_record;
  if (last) {
    $('playerState').textContent = `${last.player_state.toUpperCase()} · ${SYMBOLS[last.player_move]?.emoji || ''} ${SYMBOLS[last.player_move]?.label || last.player_move}`;
    $('botState').textContent = `${last.machine_state.toUpperCase()}`;
    $('machineMove').textContent = `${SYMBOLS[last.machine_move]?.emoji || ''} ${SYMBOLS[last.machine_move]?.label || last.machine_move}`;
    $('measurementState').textContent = `Medición: |${last.quantum_measurement}⟩ · ${last.quantum_operations} operaciones`;
    $('backendStatus').textContent = 'QISKIT / AER SIMULATOR';
    const winner = last.winner === 'Gana jugador' ? 'PLAYER 1 GANA' : last.winner === 'Gana máquina' ? 'Q-BOT GANA' : 'EMPATE';
    setNotice(`R${last.round}: ${winner} · +${last.player_points} P1 / +${last.machine_points} Q-BOT.`);
  } else {
    $('playerState').textContent = 'Elige una jugada';
    $('botState').textContent = 'PERSONALIDAD SECRETA';
    $('machineMove').textContent = 'AWAITING MEASUREMENT';
    $('measurementState').textContent = 'Medición: —';
    $('backendStatus').textContent = 'QISKIT BACKEND READY';
  }

  const history = [...(state.history || [])].reverse();
  $('historyBody').innerHTML = history.length ? history.map((row) => {
    const resultClass = row.winner === 'Gana jugador' ? 'won' : row.winner === 'Gana máquina' ? 'lost' : 'draw';
    return `<tr><td>${String(row.round).padStart(2, '0')}</td><td>${SYMBOLS[row.agreement]?.emoji || '🎲'}</td><td>${SYMBOLS[row.player_move]?.emoji || '—'} ${row.player_state}</td><td>${SYMBOLS[row.machine_move]?.emoji || '—'} ${row.machine_state}</td><td class="${resultClass}">${row.player_points}/${row.machine_points}</td></tr>`;
  }).join('') : '<tr><td colspan="5" class="empty-row">Todavía no hay rondas.</td></tr>';

  const makeDots = (key) => Array.from({ length: state.total_rounds }, (_, i) => {
    const row = state.history?.[i];
    const action = row?.[key];
    const cls = action === 'Cooperación' ? 'coop' : action === 'Traición' ? 'betray' : action === 'No decisoria' ? 'neutral' : '';
    return `<span class="dot ${cls}" title="Ronda ${i+1}: ${action || 'sin jugar'}"></span>`;
  }).join('');
  $('playerDots').innerHTML = makeDots('player_state');
  $('botDots').innerHTML = makeDots('machine_state');

  document.querySelectorAll('[data-m]').forEach((button) => { button.disabled = Boolean(state.game_over) || roundBusy; });
  if (state.game_over) {
    updateEnding(state);
    setScreen('ending');
  }
}

function updateEnding(state) {
  $('endtitle').textContent = '15 ROUNDS COMPLETE';
  $('endscore').innerHTML = `PLAYER 1: ${state.scores.player} PTS<br>Q-BOT: ${state.scores.machine} PTS`;
  if (state.guess_submitted) {
    const correct = Boolean(state.guess_correct);
    $('guessResult').textContent = `${correct ? '¡ACERTASTE! +15 PUNTOS.' : 'NO ACERTASTE.'} Personalidad real: ${PERSONALITIES[state.personality] || state.personality}.`;
    $('guessPanel').querySelectorAll('[data-personality]').forEach((button) => { button.disabled = true; });
    $('endscore').innerHTML = `PLAYER 1: ${state.scores.player} PTS<br>Q-BOT: ${state.scores.machine} PTS<br><small>${state.rounds_completed} RONDAS</small>`;
  } else {
    $('guessResult').textContent = 'Acierta y gana 15 puntos extra.';
    $('guessPanel').querySelectorAll('[data-personality]').forEach((button) => { button.disabled = false; });
  }
}

async function startNewGame() {
  unlockAudio();
  roundBusy = false;
  gestureArmed = true;
  predictionHistory = [];
  try {
    const result = await api('/api/game/reset', { session_id: sessionId });
    renderState(result.state);
    setScreen('game');
    setNotice('Nueva partida: observa el acuerdo y elige cooperar, traicionar o neutral.');
  } catch (error) {
    setNotice(`No se pudo iniciar la partida: ${error.message}. Comprueba la API de Cloud Run.`);
  }
}

async function submitMove(move, source = 'buttons') {
  if (!MOVES[move] || roundBusy || !gameState || gameState.game_over) return;
  roundBusy = true;
  gestureArmed = false;
  document.querySelectorAll('[data-m]').forEach((button) => { button.disabled = true; });
  $('playerState').textContent = source === 'camera' ? 'GESTO CONFIRMADO' : 'JUGADA MANUAL';
  $('botState').textContent = 'Q-BOT THINKING / MEASURING…';
  setNotice(source === 'camera' ? 'Gesto estable reconocido. Ejecutando la decisión cuántica…' : 'Jugada confirmada. Ejecutando la decisión cuántica…');
  try {
    const result = await api('/api/game/round', { session_id: sessionId, player_move: move });
    playRoundSound();
    renderState(result.state);
    const r = result.record;
    $('circuitNote').textContent = `Ronda ${r.round}: medida |${r.quantum_measurement}⟩ → ${SYMBOLS[r.machine_move]?.label}. Estado del jugador: ${r.player_state}; estado de Q-BOT: ${r.machine_state}.`;
    document.querySelectorAll('[data-m]').forEach((button) => button.classList.toggle('selected', button.dataset.m === move));
    if (!result.state.game_over) setNotice(`${source === 'camera' ? 'CÁMARA' : 'MOUSE'} · ${r.player_state.toUpperCase()} · ${r.winner.toUpperCase()} · +${r.player_points} / +${r.machine_points}. Retira la mano y vuelve a mostrarla para otra jugada.`);
  } catch (error) {
    setNotice(`No se pudo resolver la ronda: ${error.message}. Los botones manuales siguen disponibles si el servicio responde.`);
  } finally {
    roundBusy = false;
    if (gameState) renderState(gameState);
  }
}

async function guessPersonality(guess) {
  document.querySelectorAll('[data-personality]').forEach((button) => { button.disabled = true; });
  try {
    const result = await api('/api/game/guess-personality', { session_id: sessionId, guess });
    renderState(result.state);
    updateEnding(result.state);
  } catch (error) {
    setNotice(`No se pudo registrar la respuesta: ${error.message}`);
    document.querySelectorAll('[data-personality]').forEach((button) => { button.disabled = false; });
  }
}

function calculateAngle(p1, p2, p3) {
  const v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z];
  const v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z];
  const dot = v1.reduce((sum, value, i) => sum + value * v2[i], 0);
  const n1 = Math.hypot(...v1); const n2 = Math.hypot(...v2);
  if (n1 < 1e-9 || n2 < 1e-9) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (n1 * n2))));
}
function extractAngles(landmarks) {
  const triplets = [[1,2,3],[5,6,7],[9,10,11],[13,14,15],[17,18,19]];
  return triplets.map(([a,b,c]) => calculateAngle(landmarks[a], landmarks[b], landmarks[c]));
}
function majorityPrediction() {
  const counts = new Map();
  predictionHistory.forEach((move) => counts.set(move, (counts.get(move) || 0) + 1));
  let best = null, votes = 0;
  for (const [move, count] of counts.entries()) if (count > votes) { best = move; votes = count; }
  return { move: best, votes };
}
async function requestVqcPrediction(angles) {
  predictionPending = true;
  try {
    const result = await api('/api/predict', { angles });
    const move = result.gesture;
    if (!MOVES[move]) throw new Error('El backend devolvió una clase inválida.');
    predictionHistory.push(move);
    if (predictionHistory.length > 9) predictionHistory.shift();
    const { move: stable, votes } = majorityPrediction();
    $('cameraResult').textContent = stable && votes >= 5 ? `${MOVES[stable].label} · ESTABLE ${votes}/9` : `${MOVES[move].label} · ESTABILIZANDO`;
    setCameraStatus(`Cámara activa · VQC: ${MOVES[move].label} · historial ${predictionHistory.length}/9`);
    if (stable && votes >= 5 && gestureArmed && $('game').classList.contains('active') && !roundBusy && !gameState?.game_over) {
      gestureArmed = false;
      await submitMove(stable, 'camera');
    }
  } catch (error) {
    inferenceEnabled = false;
    $('cameraResult').textContent = 'VQC NO DISPONIBLE';
    setCameraStatus(`No se pudo consultar el VQC: ${error.message}. Usa los botones manuales.`);
    setNotice(`La cámara funciona localmente, pero la API de clasificación no está disponible. Los controles manuales siguen activos.`);
  } finally {
    predictionPending = false;
  }
}
function detectionLoop() {
  if (!cameraOn || !handLandmarker) return;
  const video = $('gestureVideo');
  const loopNow = performance.now();
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime && loopNow - lastDetectionAt >= 66) {
    lastVideoTime = video.currentTime;
    const now = loopNow;
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
        $('cameraResult').textContent = 'NO SE DETECTA MANO';
        if (noHandFrames >= 8) gestureArmed = true;
      }
    } catch (error) {
      setCameraStatus(`Error de MediaPipe: ${error.message}. Los controles manuales siguen disponibles.`);
    }
  }
  rafId = requestAnimationFrame(detectionLoop);
}
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus('Este navegador no permite cámara aquí. Abre el sitio HTTPS y usa los controles manuales si falla.');
    return;
  }
  $('cameraToggle').disabled = true;
  setCameraStatus('Solicitando permiso de cámara…');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    $('gestureVideo').srcObject = cameraStream;
    await $('gestureVideo').play();
    setCameraStatus('Cargando MediaPipe en el navegador…');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO', numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    cameraOn = true; inferenceEnabled = vqcAvailable; cameraErrorShown = false;
    $('cameraPlaceholder').hidden = true;
    $('cameraToggle').textContent = 'CAMERA: ON';
    $('cameraResult').textContent = vqcAvailable ? 'ESPERANDO GESTO' : 'VQC NO DISPONIBLE';
    setCameraStatus(vqcAvailable
      ? 'Cámara activa. Mantén un gesto estable; retira la mano para preparar la siguiente jugada.'
      : 'Cámara activa, pero el modelo VQC no está cargado. Puedes jugar con los botones manuales.');
    lastVideoTime = -1; lastDetectionAt = 0; lastPredictionAt = 0; predictionHistory = []; gestureArmed = true; noHandFrames = 0;
    cancelAnimationFrame(rafId); rafId = requestAnimationFrame(detectionLoop);
  } catch (error) {
    stopCamera();
    $('cameraPlaceholder').hidden = false;
    $('cameraPlaceholder').innerHTML = `◎<br><span>CÁMARA NO DISPONIBLE</span><small>${escapeHtml(error.message || 'No se pudo iniciar la cámara.')}</small>`;
    setCameraStatus(`No se pudo activar la cámara/MediaPipe: ${error.message || error}. Puedes jugar con el mouse.`);
  } finally {
    $('cameraToggle').disabled = false;
  }
}
function escapeHtml(text) { return String(text).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c])); }
function stopCamera() {
  cameraOn = false;
  cancelAnimationFrame(rafId); rafId = 0;
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  if ($('gestureVideo')) $('gestureVideo').srcObject = null;
  if (handLandmarker) { try { handLandmarker.close(); } catch (_) {} }
  handLandmarker = null;
  if ($('cameraPlaceholder')) $('cameraPlaceholder').hidden = false;
  if ($('cameraToggle')) $('cameraToggle').textContent = 'CAMERA: OFF';
  if ($('cameraResult')) $('cameraResult').textContent = 'ESPERANDO GESTO';
}

function openLab() { $('labModal').hidden = false; $('closeLab').focus(); }
function closeLab() { $('labModal').hidden = true; $('vqcLab').focus(); }

$('start').addEventListener('click', startNewGame);
$('restart').addEventListener('click', startNewGame);
$('newMatch').addEventListener('click', startNewGame);
document.querySelectorAll('[data-m]').forEach((button) => button.addEventListener('click', () => submitMove(button.dataset.m, 'buttons')));
document.querySelectorAll('[data-personality]').forEach((button) => button.addEventListener('click', () => guessPersonality(button.dataset.personality)));
$('cameraToggle').addEventListener('click', () => cameraOn ? stopCamera() : startCamera());
$('vqcLab').addEventListener('click', openLab);
$('closeLab').addEventListener('click', closeLab);
$('labModal').addEventListener('click', (event) => { if (event.target === $('labModal')) closeLab(); });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !$('labModal').hidden) closeLab(); });
window.addEventListener('beforeunload', stopCamera);

async function boot() {
  try {
    const health = await api('/health', undefined, 'GET');
    vqcAvailable = Boolean(health.vqc_loaded);
    $('backendStatus').textContent = health.vqc_loaded ? 'QISKIT / VQC READY' : 'QISKIT READY · VQC NO CARGADO';
    if (!health.vqc_loaded) {
      setCameraStatus('El juego manual funciona; para usar cámara/VQC, despliega el archivo .dill compatible en Cloud Run.');
      inferenceEnabled = false;
    } else {
      setCameraStatus('Backend conectado · cámara apagada · controles manuales disponibles.');
    }
  } catch (error) {
    vqcAvailable = false;
    inferenceEnabled = false;
    $('backendStatus').textContent = 'API OFFLINE';
    setCameraStatus(`API no conectada: ${error.message}. Comprueba VITE_API_BASE_URL en Cloudflare Pages.`);
  }
}
boot();

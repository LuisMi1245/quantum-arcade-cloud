import './style.css';
// MediaPipe se carga bajo demanda (ver startCamera) para que un fallo de esa librería nunca rompa el botón START.

const RAW_API_BASE = String(import.meta.env.VITE_API_BASE_URL ?? '').trim();
const API_BASE = (RAW_API_BASE || 'http://127.0.0.1:8080').replace(/\/$/, '');
// En producción, sin VITE_API_BASE_URL la web intentaría hablar con 127.0.0.1 (el PC de quien juega).
const API_MISCONFIGURED = !RAW_API_BASE && !import.meta.env.DEV;
const REQUEST_TIMEOUT_MS = 60000; // En frío, Cloud Run carga Qiskit y el modelo VQC antes de responder
const MODEL_URL = `${import.meta.env.BASE_URL}models/hand_landmarker.task`;
const WASM_URL = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const TOTAL_ROUNDS = 15;
// Ruta que el backend debe exponer para registrar la adivinanza final (+15 pts si acierta).
const GUESS_ENDPOINT = '/api/game/guess-personality';

const MOVES = {
  rock: { label: 'ROCK', emoji: '✊', spanish: 'piedra' },
  paper: { label: 'PAPER', emoji: '✋', spanish: 'papel' },
  scissors: { label: 'SCISSORS', emoji: '✌️', spanish: 'tijeras' },
};
const SPANISH_TO_MOVE = { piedra: 'rock', papel: 'paper', tijeras: 'scissors', tijera: 'scissors' };
const PERSONALITIES = { indiferente: 'INDIFFERENT', reciproco: 'RECIPROCAL', rencoroso: 'VENGEFUL' };
const $ = (id) => document.getElementById(id);

// Estos indicadores traducen los valores de estado enviados por el backend a
// etiquetas consistentes, con un emoji visible en tarjetas e historial.
const DECISIONS = {
  cooperation: { label: 'COOPERATION', emoji: '🤝', className: 'coop' },
  betrayal: { label: 'BETRAYAL', emoji: '⚔️', className: 'betray' },
  neutral: { label: 'NEUTRAL', emoji: '⚖️', className: 'neutral' },
  unknown: { label: 'UNCLASSIFIED', emoji: '❔', className: 'unknown' },
};

// crypto.randomUUID solo existe en contextos seguros y sessionStorage puede lanzar error si el
// navegador bloquea el almacenamiento: cualquiera de los dos rompía todo el script al cargar.
function generateSessionId() {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.floor(Math.random() * 16);
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

let sessionId = null;
try { sessionId = sessionStorage.getItem('qaSessionId'); } catch (_) { /* almacenamiento bloqueado */ }
if (!sessionId) {
  sessionId = generateSessionId();
  try { sessionStorage.setItem('qaSessionId', sessionId); } catch (_) { /* dura mientras la página siga abierta */ }
}

// --- Nombre del jugador: vive en sessionStorage, igual que el id de sesión. ---
const NAME_KEY = 'qaPlayerName';
const DEFAULT_PLAYER_NAME = 'PLAYER 1';
const MAX_NAME_LENGTH = 12;
let nameModalMode = 'start';

function sanitizeName(raw) {
  return String(raw ?? '')
    .replace(/[^\p{L}\p{N} _-]/gu, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_NAME_LENGTH);
}

function loadPlayerName() {
  try { return sanitizeName(sessionStorage.getItem(NAME_KEY)); } catch (_) { return ''; }
}

function savePlayerName(name) {
  try { sessionStorage.setItem(NAME_KEY, name); } catch (_) { /* sin almacenamiento: dura mientras la página siga abierta */ }
}

let playerName = loadPlayerName();

function displayName() {
  return (playerName || DEFAULT_PLAYER_NAME).toUpperCase();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]
  ));
}

let gameState = null;
let roundBusy = false;
let guessBusy = false;
let endSoundPlayed = false;
let endBus = null;
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

// Mensajes visibles en la pantalla de inicio (el panel de integración vive en #game, oculto al inicio).
function setIntroStatus(message, kind = 'info', detail = '') {
  const node = $('introStatus');
  if (!node) return;
  if (!message) {
    node.hidden = true;
    node.textContent = '';
    node.className = 'intro-status';
    return;
  }
  node.className = `intro-status ${kind}`;
  node.textContent = message;
  if (detail) {
    const small = document.createElement('small');
    small.textContent = detail;
    node.appendChild(small);
  }
  node.hidden = false;
}

function apiHint() {
  if (API_MISCONFIGURED) {
    return 'VITE_API_BASE_URL is missing in this deployment: the site is pointing to http://127.0.0.1:8080. Set it to the Cloud Run URL and deploy again.';
  }
  if (location.protocol === 'https:' && API_BASE.startsWith('http:') && !/^http:\/\/(127\.0\.0\.1|localhost)/.test(API_BASE)) {
    return `The site uses HTTPS but the API uses HTTP (${API_BASE}); the browser blocks it. Use an https URL.`;
  }
  return `API: ${API_BASE}`;
}

let startBusy = false;
const START_LABEL = 'START ▶';
function setStartBusy(busy) {
  startBusy = busy;
  const button = $('start');
  if (!button) return;
  button.disabled = busy;
  button.textContent = busy ? 'CONNECTING…' : START_LABEL;
}

function setIntegrationStatus(message) {
  const node = $('integrationStatus');
  if (node) node.textContent = message;
}

function setIntegrationDetails(message) {
  const node = $('integrationDetails');
  if (node) node.textContent = message;
}

function applyPlayerName() {
  const name = displayName();
  ['playerNameHead', 'playerNameSection', 'playerNameStatus', 'playerNameTh'].forEach((id) => {
    const node = $(id);
    if (node) node.textContent = name;
  });
  const chip = $('introPlayer');
  if (chip) chip.hidden = !playerName;
  const chipName = $('introPlayerName');
  if (chipName) chipName.textContent = name;
  // Si ya hay una partida en pantalla, refresca marcador, historial y pantalla final.
  if (gameState) renderState(gameState);
}

function openNameModal(mode) {
  nameModalMode = mode;
  const input = $('nameInput');
  input.value = playerName;
  $('nameTitle').textContent = mode === 'start' ? 'NEW PLAYER' : 'CHANGE NAME';
  $('nameConfirm').textContent = mode === 'start' ? 'PLAY ▶' : 'SAVE';
  $('nameModal').hidden = false;
  input.focus();
  input.select();
}

function closeNameModal() {
  $('nameModal').hidden = true;
  $((nameModalMode === 'start') ? 'start' : 'changeName')?.focus();
}

function confirmName() {
  playerName = sanitizeName($('nameInput').value) || DEFAULT_PLAYER_NAME;
  savePlayerName(playerName);
  $('nameModal').hidden = true;
  applyPlayerName();
  if (nameModalMode === 'start') startNewGame();
}

// --- Botón de cámara (integrado en la tarjeta del jugador) ---
const CAMERA_UI = {
  off: { icon: '📷', label: 'TURN ON CAMERA', hint: 'Play with your hand! Show ✊ ✋ or ✌ to the camera. Your video stays in your browser.' },
  loading: { icon: '⏳', label: 'OPENING CAMERA…', hint: 'Accept the camera permission in your browser.' },
  on: { icon: '🟢', label: 'CAMERA ON', hint: 'Show your move and hold it still; remove your hand for the next round. Tap the button to turn it off.' },
};

function setCameraUi(state, hintOverride) {
  const ui = CAMERA_UI[state] || CAMERA_UI.off;
  $('cameraCta').dataset.state = state;
  $('cameraIcon').textContent = ui.icon;
  $('cameraToggleLabel').textContent = ui.label;
  $('cameraHint').textContent = hintOverride || ui.hint;
  $('cameraToggle').setAttribute('aria-pressed', String(state === 'on'));
}

// --- Diálogo de reglas ---
let rulesReturnFocus = null;
const rulesAreOpen = () => !$('rulesModal').hidden;

function openRules() {
  rulesReturnFocus = document.activeElement;
  $('rulesModal').hidden = false;
  document.body.classList.add('modal-open');
  $('rulesBody').scrollTop = 0;
  $('rulesClose').focus();
}

function closeRules() {
  if (!rulesAreOpen()) return;
  $('rulesModal').hidden = true;
  document.body.classList.remove('modal-open');
  // Si la mano sigue a la vista al cerrar, exige retirarla antes de poder jugar por gesto.
  if (cameraOn) gestureArmed = false;
  (rulesReturnFocus && rulesReturnFocus.isConnected ? rulesReturnFocus : $('rulesBtn')).focus();
}

function trapRulesFocus(event) {
  if (!rulesAreOpen()) return;
  if (event.key === 'Escape') { closeRules(); return; }
  if (event.key !== 'Tab') return;
  const items = [...$('rulesDialog').querySelectorAll('button, [tabindex="0"]')];
  const first = items[0];
  const last = items[items.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
}

// --- Diálogo de créditos ---
let creditsReturnFocus = null;
const creditsAreOpen = () => !$('creditsModal').hidden;

function openCredits() {
  creditsReturnFocus = document.activeElement;
  $('creditsModal').hidden = false;
  document.body.classList.add('modal-open');
  $('creditsBody').scrollTop = 0;
  $('creditsClose').focus();
}

function closeCredits() {
  if (!creditsAreOpen()) return;
  $('creditsModal').hidden = true;
  if (!rulesAreOpen()) document.body.classList.remove('modal-open');
  // Si la mano sigue a la vista al cerrar, exige retirarla antes de poder jugar por gesto.
  if (cameraOn) gestureArmed = false;
  const fallback = document.querySelector('[data-open-credits]');
  const target = creditsReturnFocus && creditsReturnFocus.isConnected ? creditsReturnFocus : fallback;
  if (target) target.focus();
}

function trapCreditsFocus(event) {
  if (!creditsAreOpen()) return;
  if (event.key === 'Escape') { closeCredits(); return; }
  if (event.key !== 'Tab') return;
  const items = [...$('creditsDialog').querySelectorAll('button, [tabindex="0"]')];
  const first = items[0];
  const last = items[items.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
}

function setScreen(id) {
  document.querySelectorAll('.screen').forEach((screen) => screen.classList.remove('active'));
  $(id).classList.add('active');
  window.scrollTo(0, 0);
}

async function api(path, payload = undefined, method = 'POST', timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const options = { method, headers: { 'Content-Type': 'application/json' }, cache: 'no-store', signal: controller.signal };
  if (payload !== undefined) options.body = JSON.stringify(payload);
  try {
    const response = await fetch(`${API_BASE}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  } catch (error) {
    if (error?.name === 'AbortError') throw new Error(`the server did not answer in ${Math.round(timeoutMs / 1000)} s`);
    if (error instanceof TypeError) throw new Error('could not connect to the server (network, CORS or wrong API URL)');
    throw error;
  } finally {
    clearTimeout(timer);
  }
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

// Ejecuta `callback` cuando el AudioContext está realmente activo (cubre navegadores que lo dejan suspendido).
function withAudio(callback) {
  unlockAudio();
  if (!audioContext) return;
  if (audioContext.state === 'running') {
    callback(audioContext);
  } else {
    audioContext.resume()
      .then(() => { if (audioContext.state === 'running') callback(audioContext); })
      .catch(() => {});
  }
}

function playTone(ctx, out, { freq, start, duration, type = 'square', volume = 0.05, glideTo = null, filterHz = null, vibrato = false }) {
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(freq, start);
  if (glideTo) oscillator.frequency.exponentialRampToValueAtTime(glideTo, start + duration);

  // Ataque rápido, volumen sostenido y liberación suave (patrón ADSR estándar).
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.linearRampToValueAtTime(volume, start + 0.015);
  gain.gain.setTargetAtTime(0.0001, start + Math.max(0.02, duration - 0.1), 0.025);

  if (filterHz) {
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = filterHz;
    oscillator.connect(filter);
    filter.connect(gain);
  } else {
    oscillator.connect(gain);
  }
  gain.connect(out);

  if (vibrato) {
    const lfo = ctx.createOscillator();
    const lfoDepth = ctx.createGain();
    lfo.frequency.value = 5.5;
    lfoDepth.gain.value = freq * 0.012;
    lfo.connect(lfoDepth);
    lfoDepth.connect(oscillator.frequency);
    lfo.start(start);
    lfo.stop(start + duration + 0.03);
  }
  oscillator.start(start);
  oscillator.stop(start + duration + 0.03);
}

function stopEndSound() {
  if (!endBus || !audioContext) { endBus = null; return; }
  try {
    endBus.gain.cancelScheduledValues(audioContext.currentTime);
    endBus.gain.setTargetAtTime(0.0001, audioContext.currentTime, 0.03);
  } catch (_) { /* el sonido ya terminó */ }
  endBus = null;
}

// 'win' → fanfarria ascendente; 'lose' → "trombón triste" descendente; empate → silencio.
function playEndSound(outcome) {
  if (outcome !== 'win' && outcome !== 'lose') return;
  withAudio((ctx) => {
    stopEndSound();
    const bus = ctx.createGain();
    bus.gain.value = 1;
    bus.connect(ctx.destination);
    endBus = bus;
    const t0 = ctx.currentTime + 0.05;

    if (outcome === 'win') {
      const chord = [523.25, 659.25, 783.99, 1046.5]; // Do mayor: C5 E5 G5 C6
      chord.forEach((freq, i) => playTone(ctx, bus, { freq, start: t0 + i * 0.13, duration: 0.14, type: 'square', volume: 0.05 }));
      const chordStart = t0 + 0.55;
      chord.forEach((freq) => playTone(ctx, bus, { freq, start: chordStart, duration: 0.9, type: 'triangle', volume: 0.035 }));
      playTone(ctx, bus, { freq: 1568, start: chordStart + 0.15, duration: 0.7, type: 'sine', volume: 0.02 });
    } else {
      [311.13, 293.66, 277.18].forEach((freq, i) => playTone(ctx, bus, {
        freq, start: t0 + i * 0.42, duration: 0.38, type: 'sawtooth', volume: 0.05, filterHz: 900,
      }));
      playTone(ctx, bus, {
        freq: 261.63, start: t0 + 3 * 0.42, duration: 1.1, type: 'sawtooth', volume: 0.055,
        glideTo: 233.08, filterHz: 800, vibrato: true,
      });
    }
  });
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

function decisionInfo(value) {
  const normalized = String(value ?? '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[_/-]+/g, ' ')
    .trim();

  if (normalized.includes('cooper') || normalized === 'coop' || normalized === 'cooperate') {
    return DECISIONS.cooperation;
  }
  if (normalized.includes('traicion') || normalized.includes('betray') || normalized.includes('defect')) {
    return DECISIONS.betrayal;
  }
  if (normalized.includes('neutral') || normalized.includes('no decisoria') || normalized.includes('indiferente') || normalized.includes('indifferent')) {
    return DECISIONS.neutral;
  }
  return DECISIONS.unknown;
}

function recordDecision(row, participant) {
  if (!row) return undefined;
  const candidates = participant === 'player'
    ? [row.player_state, row.player_decision, row.player_action, row.player_behavior, row.player_strategy]
    : [row.machine_state, row.qbot_state, row.bot_state, row.machine_decision, row.machine_action, row.machine_behavior, row.machine_strategy];
  return candidates.find((value) => value !== undefined && value !== null && String(value).trim() !== '');
}

function renderDecisionCard(participant, move, decision) {
  const isBot = participant === 'bot';
  const prefix = isBot ? 'bot' : 'player';
  const info = decisionInfo(decision);
  const key = moveKey(move);
  const card = $(`${prefix}DecisionCard`);
  if (card) {
    card.classList.remove('coop', 'betray', 'neutral', 'unknown');
    card.classList.add(info.className);
  }
  $(`${prefix}DecisionEmoji`).textContent = info.emoji;
  $(`${prefix}DecisionLabel`).textContent = info.label;
  $(`${prefix}MoveEmoji`).textContent = key ? MOVES[key].emoji : '—';
  $(`${prefix}MoveLabel`).textContent = key ? MOVES[key].label : 'NO MOVE';
}

function historyParticipantCell(move, decision) {
  const key = moveKey(move);
  const info = decisionInfo(decision);
  const emoji = key ? MOVES[key].emoji : '—';
  const label = key ? MOVES[key].label : 'NO MOVE';
  return `<div class="history-participant">
    <div class="history-move"><span class="history-move-emoji">${emoji}</span><span class="history-move-label">${label}</span></div>
    <div class="decision-pill ${info.className}"><span class="decision-pill-emoji">${info.emoji}</span><span>${info.label}</span></div>
  </div>`;
}

function renderHistory(state) {
  const history = [...(state.history || [])].reverse();
  $('hist').innerHTML = history.length ? history.map((row, index) => {
    const agreement = moveKey(row.agreement ?? row.pact ?? row.current_agreement);
    const pactEmoji = agreement ? MOVES[agreement].emoji : '🎲';
    const pactLabel = agreement ? MOVES[agreement].label : '—';
    const playerDecision = recordDecision(row, 'player');
    const botDecision = recordDecision(row, 'bot');
    const roundNumber = Number(row.round ?? (history.length - index));
    const playerPoints = row.player_points ?? row.player_score ?? '—';
    const machinePoints = row.machine_points ?? row.machine_score ?? row.bot_points ?? '—';
    return `<tr>
      <td class="round-cell" data-label="ROUND">${String(roundNumber).padStart(2, '0')}</td>
      <td data-label="PACT"><div class="history-pact"><span class="history-pact-emoji">${pactEmoji}</span><span>${pactLabel}</span></div></td>
      <td data-label="${escapeHtml(displayName())}">${historyParticipantCell(row.player_move, playerDecision)}</td>
      <td data-label="Q-BOT">${historyParticipantCell(row.machine_move, botDecision)}</td>
      <td class="points-cell" data-label="POINTS P / Q"><span>${playerPoints}</span><i>/</i><span>${machinePoints}</span></td>
    </tr>`;
  }).join('') : '';

  const dotsFor = (forBot) => Array.from({ length: state.total_rounds || TOTAL_ROUNDS }, (_, i) => {
    const row = state.history?.[i];
    if (!row) return '<span class="dot" title="Round not played"></span>';
    const cls = winnerClass(row.winner, forBot);
    const label = row.winner === 'Gana jugador' ? (forBot ? 'Q-Bot loss' : 'Player win')
      : row.winner === 'Gana máquina' ? (forBot ? 'Q-Bot win' : 'Player loss') : 'Tie';
    return `<span class="dot ${cls === 'won' ? 'win' : cls === 'lost' ? 'loss' : 'tie'}" title="Round ${i + 1}: ${label}"></span>`;
  }).join('');
  $('pdots').innerHTML = dotsFor(false);
  $('bdots').innerHTML = dotsFor(true);
}

function updateEnding(state) {
  const playerScore = Number(state?.scores?.player || 0);
  const botScore = Number(state?.scores?.machine || 0);
  const completed = Number(state?.rounds_completed ?? state?.history?.length ?? 0);
  const awaitingGuess = Boolean(state?.game_over) && !state?.guess_submitted;

  if (awaitingGuess) {
    // Las 15 rondas terminaron, pero el resultado final depende del bono de +15 puntos.
    $('endtitle').innerHTML = `${completed} ROUNDS COMPLETE<br>GUESS Q-BOT!`;
    $('endscore').innerHTML = `${escapeHtml(displayName())}: ${playerScore}<br>Q-BOT: ${botScore}<br><small>GUESS THE PERSONALITY FOR THE FINAL BONUS</small>`;
  } else {
    const title = playerScore > botScore ? 'CONGRATULATIONS!<br>YOU WIN!'
      : playerScore < botScore ? 'BETTER LUCK<br>NEXT TIME' : 'IT\'S A TIE!<br>GOOD GAME!';
    $('endtitle').innerHTML = title;
    $('endscore').innerHTML = `${escapeHtml(displayName())}: ${playerScore}<br>Q-BOT: ${botScore}<br><small>${completed} ROUNDS COMPLETE</small>`;
    if (!endSoundPlayed) {
      endSoundPlayed = true;
      playEndSound(playerScore > botScore ? 'win' : playerScore < botScore ? 'lose' : 'tie');
    }
  }
  renderGuessPanel(state);
}

function renderGuessPanel(state) {
  const panel = $('guessPanel');
  if (!panel) return;
  const result = $('guessResult');
  const buttons = panel.querySelectorAll('[data-personality]');

  // Solo se puede adivinar cuando se completaron todas las rondas (el backend lo exige).
  if (!state?.game_over) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  if (state.guess_submitted) {
    const right = Boolean(state.guess_correct);
    buttons.forEach((button) => {
      button.disabled = true;
      button.classList.toggle('picked', button.dataset.personality === state.guessed_personality);
    });
    result.className = `guess-result ${right ? 'right' : 'wrong'}`;
    result.textContent = `${right ? 'YOU GOT IT! +15 POINTS.' : 'WRONG GUESS.'} Real personality: ${PERSONALITIES[state.personality] || String(state.personality || '—').toUpperCase()}.`;
  } else {
    buttons.forEach((button) => {
      button.disabled = guessBusy;
      button.classList.remove('picked');
    });
    result.className = 'guess-result';
    result.textContent = 'The personality is secret. Choose one option.';
  }
}

async function submitPersonalityGuess(guess) {
  if (guessBusy || !gameState?.game_over || gameState.guess_submitted) return;
  guessBusy = true;
  unlockAudio();
  const panel = $('guessPanel');
  panel.querySelectorAll('[data-personality]').forEach((button) => { button.disabled = true; });
  const result = $('guessResult');
  result.className = 'guess-result';
  result.textContent = "Revealing Q-BOT's personality…";

  try {
    const response = await api(GUESS_ENDPOINT, { session_id: sessionId, guess });
    renderState(response.state);
    setIntegrationStatus(response.correct
      ? 'Personality guessed · +15 points added to your score.'
      : 'Personality revealed · no bonus this time.');
  } catch (error) {
    panel.querySelectorAll('[data-personality]').forEach((button) => { button.disabled = false; });
    result.className = 'guess-result wrong';
    result.textContent = `Could not save your answer: ${error.message}`;
  } finally {
    guessBusy = false;
  }
}

function renderAgreementPanel(state) {
  const total = state.total_rounds || TOTAL_ROUNDS;
  const round = state.round_number || total;
  const key = moveKey(state.current_agreement);
  const emoji = $('agreementEmoji');
  const name = $('agreementName');
  const detail = $('agreementRound');
  const actions = $('agreementActions');
  if (!emoji || !name || !detail || !actions) return;

  if (state.game_over || !key) {
    emoji.textContent = '✓';
    name.textContent = 'PACTS COMPLETED';
    detail.textContent = 'The game is over: there is no active pact.';
    actions.innerHTML = `
      <div class="action-chip coop"><strong>COOPERATE</strong><span>Match the pact symbol.</span></div>
      <div class="action-chip betray"><strong>BETRAY</strong><span>Choose the symbol that beats the pact.</span></div>
      <div class="action-chip neutral"><strong>NEUTRAL</strong><span>Choose the symbol that loses to the pact.</span></div>`;
    return;
  }

  const beatsAgreement = { rock: 'paper', paper: 'scissors', scissors: 'rock' };
  const losesToAgreement = { rock: 'scissors', paper: 'rock', scissors: 'paper' };
  const coop = MOVES[key];
  const betray = MOVES[beatsAgreement[key]];
  const neutral = MOVES[losesToAgreement[key]];
  emoji.textContent = coop.emoji;
  name.textContent = coop.label;
  detail.textContent = `ROUND ${round}/${total} · random pact for this round`;
  actions.innerHTML = `
    <div class="action-chip coop"><strong>COOPERATE</strong><span>${coop.emoji} ${coop.label}<br>Match the pact</span></div>
    <div class="action-chip betray"><strong>BETRAY</strong><span>${betray.emoji} ${betray.label}<br>Beat the pact</span></div>
    <div class="action-chip neutral"><strong>NEUTRAL</strong><span>${neutral.emoji} ${neutral.label}<br>Lose to the pact</span></div>`;
}

function renderState(state) {
  gameState = state;
  $('ps').textContent = state.scores?.player ?? 0;
  $('bs').textContent = state.scores?.machine ?? 0;

  const total = state.total_rounds || TOTAL_ROUNDS;
  const agreement = state.current_agreement;
  renderAgreementPanel(state);
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
    renderDecisionCard('player', last.player_move, recordDecision(last, 'player'));
    renderDecisionCard('bot', last.machine_move, recordDecision(last, 'bot'));
    $('state').textContent = `${playerKey ? MOVES[playerKey].label : 'MOVE'} · RECORDED`;
    $('botstate').textContent = 'ROUND RESOLVED';
    document.querySelectorAll('[data-m]').forEach((button) => {
      button.classList.toggle('selected', button.dataset.m === playerKey);
    });
    setIntegrationDetails(
      `Pact this round: ${moveLabel(last.agreement)}. Last quantum measurement: |${last.quantum_measurement}⟩. ` +
      `Q-BOT move: ${botKey ? MOVES[botKey].label : '—'}. Round points: ${last.player_points} / ${last.machine_points}.`
    );
  } else {
    renderDecisionCard('player', null, null);
    renderDecisionCard('bot', null, null);
    $('state').textContent = 'READY';
    $('botstate').textContent = 'ACTIVE / THINKING';
    setIntegrationDetails(agreement
      ? `Current round pact: ${moveLabel(agreement)}. Remove your hand after a move to enable the next one.`
      : 'The camera is processed locally. The five angles are sent to the backend for VQC classification.');
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

function resetGuessPanel() {
  const panel = $('guessPanel');
  if (!panel) return;
  panel.hidden = true;
  panel.querySelectorAll('[data-personality]').forEach((button) => {
    button.disabled = false;
    button.classList.remove('picked');
  });
  const result = $('guessResult');
  if (result) { result.className = 'guess-result'; result.textContent = ''; }
}

function reportStartError(error) {
  console.error('[START] Could not start the game:', error);
  setIntroStatus(`Could not start the game: ${error.message}`, 'error', `${apiHint()} Press START to try again.`);
  setIntegrationStatus(`Could not start the game: ${error.message}`);
  setIntegrationDetails('Check the Cloud Run API and press START again.');
}

async function startNewGame() {
  if (startBusy) return;
  setStartBusy(true);
  setIntroStatus('Connecting to the quantum server… if it was asleep, it can take up to one minute.', 'info');
  unlockAudio();
  if (cameraOn || cameraStream) stopCamera();
  roundBusy = false;
  guessBusy = false;
  endSoundPlayed = false;
  stopEndSound();
  resetGuessPanel();
  gestureArmed = true;
  predictionHistory = [];
  predictionPending = false;
  setMoveProbabilities(null);
  document.querySelectorAll('[data-m]').forEach((button) => {
    button.classList.remove('selected');
    button.disabled = false;
  });
  let result;
  try {
    result = await api('/api/game/reset', { session_id: sessionId });
  } catch (error) {
    reportStartError(error);
    setStartBusy(false);
    return;
  }

  try {
    renderState(result.state);
    setScreen('game');
    setIntroStatus('');
    setIntegrationStatus(vqcAvailable
      ? 'Gesture model: VQC loaded · Qiskit backend: connected.'
      : 'Gesture model: unavailable · Qiskit backend: connected; manual play is available.');
  } catch (error) {
    reportStartError(new Error(`the /api/game/reset response does not have the expected format (${error.message})`));
  } finally {
    setStartBusy(false);
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
    : 'Manual move confirmed; resolving the quantum round…');

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
    setIntegrationStatus(`Could not resolve the round: ${error.message}`);
    setIntegrationDetails('You can keep using the manual buttons when the connection is back.');
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
    if (!MOVES[move]) throw new Error('The backend returned an invalid class.');
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

    if (stable && votes >= 5 && gestureArmed && !rulesAreOpen() && !creditsAreOpen() && $('game').classList.contains('active') && !roundBusy && !gameState?.game_over) {
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
    setCameraUi('off', 'This browser cannot use the camera here (HTTPS is needed). Play with the buttons ✊ ✋ ✌.');
    return;
  }
  $('cameraToggle').disabled = true;
  setCameraUi('loading');
  setIntegrationStatus('Requesting camera permission…');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    const video = $('gestureVideo');
    video.srcObject = cameraStream;
    video.style.display = 'block';
    $('cameraStage').classList.add('camera-active');
    await video.play();

    setIntegrationStatus('Loading MediaPipe hand detector in the browser…');
    const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
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
    setCameraUi('on');
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
    const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError';
    setCameraUi('off', denied
      ? 'Camera permission denied. Turn it on in your browser or play with the buttons ✊ ✋ ✌.'
      : 'Could not turn on the camera. You can play with the buttons ✊ ✋ ✌.');
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
  $('cameraStage')?.classList.remove('camera-active');
  if (handLandmarker) {
    try { handLandmarker.close(); } catch (_) { /* detector is already closed */ }
  }
  handLandmarker = null;
  setCameraUi('off');
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
  guessBusy = false;
  endSoundPlayed = false;
  stopEndSound();
  resetGuessPanel();
  predictionHistory = [];
  setMoveProbabilities(null);
  $('ps').textContent = '0';
  $('bs').textContent = '0';
  $('state').textContent = 'READY';
  $('botstate').textContent = 'ACTIVE / THINKING';
  renderDecisionCard('player', null, null);
  renderDecisionCard('bot', null, null);
  $('roundlabel').textContent = 'ROUND 1 / 15 — Choose your move';
  renderAgreementPanel({ total_rounds: TOTAL_ROUNDS, round_number: 1, current_agreement: null, game_over: false });
  $('agreementName').textContent = 'PRESS START';
  $('agreementRound').textContent = 'One random pact is drawn each round.';
  $('agreementEmoji').textContent = '🎲';
  $('agreementActions').innerHTML = '<div class="action-chip coop"><strong>COOPERATE</strong><span>Match the pact</span></div><div class="action-chip betray"><strong>BETRAY</strong><span>Beat the pact</span></div><div class="action-chip neutral"><strong>NEUTRAL</strong><span>Lose to the pact</span></div>';
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
  setIntegrationDetails('The camera is processed locally. The angles are sent to the backend for VQC classification.');
  setIntroStatus('');
  setStartBusy(false);
  setScreen('intro');
}

// START pide el nombre solo la primera vez de la sesión; después se reutiliza.
$('start').addEventListener('click', () => {
  if (playerName) startNewGame();
  else openNameModal('start');
});
$('changeName').addEventListener('click', () => openNameModal('rename'));
$('nameForm').addEventListener('submit', (event) => { event.preventDefault(); confirmName(); });
$('nameCancel').addEventListener('click', closeNameModal);
$('nameModal').addEventListener('click', (event) => { if (event.target === $('nameModal')) closeNameModal(); });
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !$('nameModal').hidden) closeNameModal();
});
document.querySelectorAll('[data-personality]').forEach((button) => {
  button.addEventListener('click', () => submitPersonalityGuess(button.dataset.personality));
});
$('restart').addEventListener('click', returnToIntro);
document.querySelectorAll('[data-m]').forEach((button) => {
  button.addEventListener('click', () => submitMove(button.dataset.m, 'buttons'));
});
$('cameraToggle').addEventListener('click', () => cameraOn ? stopCamera() : startCamera());
$('rulesBtn').addEventListener('click', openRules);
$('rulesClose').addEventListener('click', closeRules);
$('rulesOk').addEventListener('click', closeRules);
$('rulesModal').addEventListener('click', (event) => { if (event.target === $('rulesModal')) closeRules(); });
document.addEventListener('keydown', trapRulesFocus);
document.querySelectorAll('[data-open-credits]').forEach((button) => button.addEventListener('click', openCredits));
$('creditsClose').addEventListener('click', closeCredits);
$('creditsOk').addEventListener('click', closeCredits);
$('creditsModal').addEventListener('click', (event) => { if (event.target === $('creditsModal')) closeCredits(); });
document.addEventListener('keydown', trapCreditsFocus);
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
    if (!startBusy) setIntroStatus('');
  } catch (error) {
    vqcAvailable = false;
    inferenceEnabled = false;
    setIntegrationStatus(`API offline: ${error.message}. Check VITE_API_BASE_URL in Cloudflare Pages.`);
    setIntegrationDetails('The game requires the Cloud Run API; check the service URL and its availability.');
    if (!startBusy) setIntroStatus(`Could not reach the API: ${error.message}`, 'error', `${apiHint()} You can press START to try again.`);
  }
}

setMoveProbabilities(null);
applyPlayerName();
boot();

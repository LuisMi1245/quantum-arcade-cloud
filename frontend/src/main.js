import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './style.css';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080').replace(/\/$/, '');
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
const PERSONALITIES = { indiferente: 'INDIFERENTE', reciproco: 'RECÍPROCA', rencoroso: 'RENCOROSA' };
const $ = (id) => document.getElementById(id);

// Estos indicadores traducen los valores de estado enviados por el backend a
// etiquetas consistentes, con un emoji visible en tarjetas e historial.
const DECISIONS = {
  cooperation: { label: 'COOPERACIÓN', emoji: '🤝', className: 'coop' },
  betrayal: { label: 'TRAICIÓN', emoji: '⚔️', className: 'betray' },
  neutral: { label: 'NO DECISORIA', emoji: '⚖️', className: 'neutral' },
  unknown: { label: 'SIN CLASIFICAR', emoji: '❔', className: 'unknown' },
};

let sessionId = sessionStorage.getItem('qaSessionId');
if (!sessionId) {
  sessionId = crypto.randomUUID();
  sessionStorage.setItem('qaSessionId', sessionId);
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
  $('nameTitle').textContent = mode === 'start' ? 'NUEVO JUGADOR' : 'CAMBIAR NOMBRE';
  $('nameConfirm').textContent = mode === 'start' ? 'JUGAR ▶' : 'GUARDAR';
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
  off: { icon: '📷', label: 'ACTIVAR CÁMARA', hint: '¡Juega con tu mano! Muestra ✊ ✋ o ✌ a la cámara. Tu vídeo se queda en tu navegador.' },
  loading: { icon: '⏳', label: 'ABRIENDO CÁMARA…', hint: 'Acepta el permiso de cámara en tu navegador.' },
  on: { icon: '🟢', label: 'CÁMARA ACTIVA', hint: 'Muestra tu jugada y mantenla quieta; retira la mano para la siguiente ronda. Toca el botón para apagarla.' },
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
  $(`${prefix}MoveLabel`).textContent = key ? MOVES[key].label : 'SIN JUGADA';
}

function historyParticipantCell(move, decision) {
  const key = moveKey(move);
  const info = decisionInfo(decision);
  const emoji = key ? MOVES[key].emoji : '—';
  const label = key ? MOVES[key].label : 'SIN JUGADA';
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
      <td class="round-cell" data-label="RONDA">${String(roundNumber).padStart(2, '0')}</td>
      <td data-label="PACTO"><div class="history-pact"><span class="history-pact-emoji">${pactEmoji}</span><span>${pactLabel}</span></div></td>
      <td data-label="${escapeHtml(displayName())}">${historyParticipantCell(row.player_move, playerDecision)}</td>
      <td data-label="Q-BOT">${historyParticipantCell(row.machine_move, botDecision)}</td>
      <td class="points-cell" data-label="PUNTOS J / Q"><span>${playerPoints}</span><i>/</i><span>${machinePoints}</span></td>
    </tr>`;
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
  const completed = Number(state?.rounds_completed ?? state?.history?.length ?? 0);
  const awaitingGuess = Boolean(state?.game_over) && !state?.guess_submitted;

  if (awaitingGuess) {
    // Las 15 rondas terminaron, pero el resultado final depende del bono de +15 puntos.
    $('endtitle').innerHTML = `${completed} ROUNDS COMPLETE<br>GUESS Q-BOT!`;
    $('endscore').innerHTML = `${escapeHtml(displayName())}: ${playerScore}<br>Q-BOT: ${botScore}<br><small>ADIVINA LA PERSONALIDAD PARA EL BONO FINAL</small>`;
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
    result.textContent = `${right ? '¡ACERTASTE! +15 PUNTOS.' : 'NO ACERTASTE.'} Personalidad real: ${PERSONALITIES[state.personality] || String(state.personality || '—').toUpperCase()}.`;
  } else {
    buttons.forEach((button) => {
      button.disabled = guessBusy;
      button.classList.remove('picked');
    });
    result.className = 'guess-result';
    result.textContent = 'La personalidad es secreta. Elige una opción.';
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
  result.textContent = 'Revelando la personalidad de Q-BOT…';

  try {
    const response = await api(GUESS_ENDPOINT, { session_id: sessionId, guess });
    renderState(response.state);
    setIntegrationStatus(response.correct
      ? 'Personalidad acertada · +15 puntos añadidos al marcador.'
      : 'Personalidad revelada · sin bono esta vez.');
  } catch (error) {
    panel.querySelectorAll('[data-personality]').forEach((button) => { button.disabled = false; });
    result.className = 'guess-result wrong';
    result.textContent = `No se pudo registrar la respuesta: ${error.message}`;
  } finally {
    guessBusy = false;
  }
}

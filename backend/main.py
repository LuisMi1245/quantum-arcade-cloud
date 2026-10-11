"""Public API for Quantum Arcade (Cloud Run).

- MediaPipe runs in each visitor's browser; images never reach this service.
- The five angle features are sent to /api/predict and classified by the saved VQC.
- Game sessions are stored in Firestore so requests can reach any Cloud Run instance.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

from quantum_prisoners_dilemma import QuantumPrisonersDilemma

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "VQC_MODEL_PATH",
    str(BASE_DIR / "models" / "vqc_piedra_papel_tijera_seed_0.dill"),
))
TOTAL_ROUNDS = int(os.environ.get("TOTAL_ROUNDS", "15"))
USE_FIRESTORE = os.environ.get("USE_FIRESTORE", "true").strip().lower() in {"1", "true", "yes"}
ALLOWED_ORIGINS = {
    item.strip().rstrip("/")
    for item in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if item.strip()
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

# Local-only fallback is for development. Cloud Run should use Firestore.
_memory_games: dict[str, dict[str, Any]] = {}
_memory_lock = threading.RLock()
_firestore_client = None
_firestore_lock = threading.Lock()

def interpretar_tres_clases(indice):
    """Keep the same interpret helper name as the training script for dill loads."""
    return indice % 3


# Some dill snapshots resolve this top-level helper through __main__. Make that
# reference available under Gunicorn, where this file is imported as module "main".
if not hasattr(sys.modules["__main__"], "interpretar_tres_clases"):
    setattr(sys.modules["__main__"], "interpretar_tres_clases", interpretar_tres_clases)

VQC_MODEL = None
VQC_LOAD_ERROR: str | None = None
try:
    from qiskit_machine_learning.algorithms import VQC

    if MODEL_PATH.is_file():
        VQC_MODEL = VQC.from_dill(str(MODEL_PATH))
        print(f"VQC loaded from {MODEL_PATH}", flush=True)
    else:
        VQC_LOAD_ERROR = f"Model file not found: {MODEL_PATH}"
        print(VQC_LOAD_ERROR, flush=True)
except Exception as exc:  # keep manual game accessible if VQC loading fails
    VQC_LOAD_ERROR = f"Could not load VQC: {type(exc).__name__}: {exc}"
    print(VQC_LOAD_ERROR, flush=True)


def get_firestore():
    global _firestore_client
    if not USE_FIRESTORE:
        return None
    with _firestore_lock:
        if _firestore_client is None:
            from google.cloud import firestore
            _firestore_client = firestore.Client()
        return _firestore_client


def _cors_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    normalized = origin.rstrip("/")
    if normalized in ALLOWED_ORIGINS:
        return normalized
    # Development convenience only: when no explicit allowlist is set, permit
    # Pages deployments. In production set ALLOWED_ORIGINS to the exact site URL.
    if not ALLOWED_ORIGINS and normalized.endswith(".pages.dev"):
        return normalized
    if normalized in {"http://localhost:5173", "http://127.0.0.1:5173"}:
        return normalized
    return None


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return ("", 204)
    return None


@app.after_request
def add_cors_headers(response):
    origin = _cors_origin(request.headers.get("Origin"))
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "600"
    response.headers["Cache-Control"] = "no-store"
    return response


def json_error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def get_session_id(data: dict[str, Any] | None = None) -> str:
    raw = (data or {}).get("session_id") or request.args.get("session_id") or ""
    session_id = str(raw)
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", session_id):
        raise ValueError("session_id no válido; crea una partida nueva desde el navegador")
    return session_id


def serialize_game(game: QuantumPrisonersDilemma) -> dict[str, Any]:
    """Save only JSON-compatible gameplay state; simulator objects stay in memory."""
    return {
        "total_rounds": int(game.total_rounds),
        "personality": str(game.maquina.personalidad),
        "pc": float(game.maquina.pc),
        "pt": float(game.maquina.pt),
        "pn": float(game.maquina.pn),
        "history": list(game.historial),
        "player_points": int(game.puntos_jugador),
        "machine_points": int(game.puntos_maquina),
        "current_agreement": game.current_agreement,
        "guess_submitted": bool(game.guess_submitted),
        "guess_correct": game.guess_correct,
        "guessed_personality": game.guessed_personality,
        "last_record": game._last_record,
        "updated_at_unix": time.time(),
    }


def restore_game(data: dict[str, Any]) -> QuantumPrisonersDilemma:
    """Recreate a game engine from a Firestore document."""
    game = QuantumPrisonersDilemma(total_rounds=int(data.get("total_rounds", TOTAL_ROUNDS)))
    game.maquina.personalidad = str(data["personality"])
    game.maquina.pc = float(data["pc"])
    game.maquina.pt = float(data["pt"])
    game.maquina.pn = float(data["pn"])
    game.historial = list(data.get("history", []))
    game.maquina.historial = list(game.historial)
    game.puntos_jugador = int(data.get("player_points", 0))
    game.puntos_maquina = int(data.get("machine_points", 0))
    game.current_agreement = data.get("current_agreement")
    game.guess_submitted = bool(data.get("guess_submitted", False))
    game.guess_correct = data.get("guess_correct")
    game.guessed_personality = data.get("guessed_personality")
    game._last_record = data.get("last_record")
    return game


def save_game(session_id: str, game: QuantumPrisonersDilemma) -> None:
    payload = serialize_game(game)
    if USE_FIRESTORE:
        get_firestore().collection("quantum_arcade_games").document(session_id).set(payload)
    else:
        with _memory_lock:
            _memory_games[session_id] = payload


def load_game(session_id: str) -> QuantumPrisonersDilemma | None:
    if USE_FIRESTORE:
        snapshot = get_firestore().collection("quantum_arcade_games").document(session_id).get()
        if not snapshot.exists:
            return None
        return restore_game(snapshot.to_dict() or {})
    with _memory_lock:
        payload = _memory_games.get(session_id)
        return restore_game(payload) if payload else None


def require_game(session_id: str) -> QuantumPrisonersDilemma:
    game = load_game(session_id)
    if game is None:
        raise LookupError("No existe esta partida. Pulsa START para crear una nueva.")
    return game


def validate_angles(value: Any) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("angles debe ser una lista de exactamente cinco ángulos en radianes")
    angles = np.asarray(value, dtype=float)
    if not np.isfinite(angles).all():
        raise ValueError("Los ángulos contienen valores no finitos")
    if np.any(angles < 0.0) or np.any(angles > np.pi + 1e-6):
        raise ValueError("Cada ángulo debe estar entre 0 y pi radianes")
    return angles.reshape(1, 5)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "quantum-arcade-api",
        "vqc_loaded": VQC_MODEL is not None,
        "vqc_error": VQC_LOAD_ERROR,
        "firestore_enabled": USE_FIRESTORE,
    })


@app.post("/api/predict")
def predict_gesture():
    if VQC_MODEL is None:
        return json_error(
            "El VQC no está cargado. Comprueba que el archivo .dill esté en backend/models y que Qiskit Machine Learning sea compatible.",
            503,
        )
    payload = request.get_json(silent=True) or {}
    try:
        x = validate_angles(payload.get("angles"))
        raw = np.asarray(VQC_MODEL.predict(x))
        if raw.size == 1:
            class_id = int(raw.reshape(-1)[0])
        elif raw.ndim == 2 and raw.shape[0] == 1 and raw.shape[1] == 3:
            class_id = int(np.argmax(raw[0]))
        else:
            return json_error(f"Formato de salida VQC inesperado: {raw.shape}", 500)
        moves = {0: "rock", 1: "paper", 2: "scissors"}
        if class_id not in moves:
            return json_error(f"El VQC devolvió una clase desconocida: {class_id}", 500)
        return jsonify({"ok": True, "class_id": class_id, "gesture": moves[class_id]})
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("VQC inference failed")
        return json_error(f"Falló la inferencia VQC: {type(exc).__name__}", 500)


@app.get("/api/game/state")
def game_state():
    try:
        session_id = get_session_id()
        game = load_game(session_id)
        if game is None:
            return json_error("No existe una partida activa", 404)
        return jsonify({"ok": True, "state": game.state()})
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Could not load game state")
        return json_error("No se pudo leer la partida. Revisa la configuración de Firestore.", 500)


@app.post("/api/game/reset")
def reset_game():
    payload = request.get_json(silent=True) or {}
    try:
        session_id = get_session_id(payload)
        game = QuantumPrisonersDilemma(total_rounds=TOTAL_ROUNDS)
        save_game(session_id, game)
        return jsonify({"ok": True, "state": game.state()})
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception:
        app.logger.exception("Could not reset game")
        return json_error("No se pudo crear la partida. Comprueba Firestore y sus permisos.", 500)


@app.post("/api/game/round")
def play_round():
    payload = request.get_json(silent=True) or {}
    try:
        session_id = get_session_id(payload)
        move = str(payload.get("player_move", ""))
        game = require_game(session_id)
        result = game.play_round(move)
        save_game(session_id, game)
        return jsonify(result)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except LookupError as exc:
        return json_error(str(exc), 404)
    except Exception:
        app.logger.exception("Quantum round failed")
        return json_error("No se pudo resolver la ronda cuántica. Revisa los logs de Cloud Run.", 500)


@app.post("/api/game/guess-personality")
def guess_personality():
    payload = request.get_json(silent=True) or {}
    try:
        session_id = get_session_id(payload)
        guess = str(payload.get("guess", ""))
        game = require_game(session_id)
        result = game.guess_personality(guess)
        save_game(session_id, game)
        return jsonify(result)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except LookupError as exc:
        return json_error(str(exc), 404)
    except Exception:
        app.logger.exception("Personality guess failed")
        return json_error("No se pudo registrar la personalidad.", 500)


if __name__ == "__main__":
    # Local development only. Cloud Run starts Gunicorn through the Dockerfile.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)

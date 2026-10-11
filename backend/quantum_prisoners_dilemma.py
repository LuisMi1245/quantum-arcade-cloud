"""Motor local del juego de acuerdos/dilema del prisionero con rival cuántico.

Este módulo adapta las reglas proporcionadas para que la interfaz HTML pueda
pedir una ronda al servidor local. No depende de matplotlib ni solicita datos
por consola. El rival, sus probabilidades y su personalidad se mantienen en
memoria durante una partida.

Requiere qiskit y qiskit-aer en el entorno del proyecto.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except ImportError as exc:  # error explícito y fácil de diagnosticar al iniciar
    raise ImportError(
        "El rival cuántico requiere Qiskit y Qiskit Aer. Instala ambos en el "
        "mismo entorno Python que ejecuta quantum_arcade_main.py."
    ) from exc


SIMBOLOS = ("piedra", "papel", "tijeras")
PERSONALIDADES = ("indiferente", "reciproco", "rencoroso")
ESTADOS_CUANTICOS = {"00": "piedra", "01": "papel", "10": "tijeras"}

ETIQUETAS_UI = {
    "piedra": "ROCK",
    "papel": "PAPER",
    "tijeras": "SCISSORS",
}
MOVIMIENTO_UI_A_SIMBOLO = {
    "rock": "piedra",
    "paper": "papel",
    "scissors": "tijeras",
    "piedra": "piedra",
    "papel": "papel",
    "tijera": "tijeras",
    "tijeras": "tijeras",
}
SIMBOLO_A_UI = {v: k for k, v in {
    "rock": "piedra", "paper": "papel", "scissors": "tijeras"
}.items()}


def simbolo_que_gana(simbolo: str) -> str:
    """Devuelve el símbolo que vence al recibido."""
    return {"piedra": "papel", "papel": "tijeras", "tijeras": "piedra"}[simbolo]


def simbolo_que_pierde(simbolo: str) -> str:
    """Devuelve el símbolo que pierde contra el recibido."""
    return {"piedra": "tijeras", "papel": "piedra", "tijeras": "papel"}[simbolo]


def piedra_papel_tijeras(jugador: str, maquina: str) -> str:
    if jugador not in SIMBOLOS or maquina not in SIMBOLOS:
        raise ValueError("Símbolo no válido")
    if jugador == maquina:
        return "Empate"
    if simbolo_que_gana(jugador) == maquina:
        return "Gana máquina"
    return "Gana jugador"


def clasificar_decision(decision: str, acuerdo: str) -> str:
    if decision == acuerdo:
        return "Cooperación"
    if decision == simbolo_que_gana(acuerdo):
        return "Traición"
    if decision == simbolo_que_pierde(acuerdo):
        return "No decisoria"
    raise ValueError("Decisión no válida")


def calcular_puntos(estado_jugador: str, estado_maquina: str, resultado: str) -> tuple[int, int]:
    # Primero se atienden los pactos mutuos, incluso si el resultado de R/P/T empata.
    if estado_jugador == "Cooperación" and estado_maquina == "Cooperación":
        return 3, 3
    if estado_jugador == "Traición" and estado_maquina == "Traición":
        return 1, 1
    if estado_jugador == "Traición" or estado_maquina == "Traición":
        if resultado == "Gana jugador":
            return 5, 0
        if resultado == "Gana máquina":
            return 0, 5
    return 0, 0


def elegir_acuerdo(rng: np.random.Generator) -> str:
    return str(rng.choice(SIMBOLOS))


class MaquinaCuantica:
    """Rival que muestrea sus decisiones de un circuito de dos qubits."""

    def __init__(
        self,
        personalidad: str | None = None,
        *,
        rng: np.random.Generator | None = None,
        simulador: Any | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        if personalidad is None:
            personalidad = str(self.rng.choice(PERSONALIDADES))
        if personalidad not in PERSONALIDADES:
            raise ValueError("Personalidad no válida")
        self.personalidad = personalidad
        self.pc = self.pt = self.pn = 1 / 3
        self.historial: list[dict[str, Any]] = []
        self.puntos_jugador = 0
        self.puntos_maquina = 0
        self.simulador = simulador or AerSimulator()

    def probabilidades(self, acuerdo: str) -> dict[str, float]:
        return {
            acuerdo: float(self.pc),
            simbolo_que_gana(acuerdo): float(self.pt),
            simbolo_que_pierde(acuerdo): float(self.pn),
        }

    def crear_circuito(self, probabilidades: dict[str, float]):
        p00 = float(probabilidades["piedra"])
        p01 = float(probabilidades["papel"])
        p10 = float(probabilidades["tijeras"])
        total = p00 + p01 + p10
        if total <= 0:
            raise ValueError("La distribución cuántica debe tener suma positiva")
        p00, p01, p10 = p00 / total, p01 / total, p10 / total

        circuito = QuantumCircuit(2, 2)
        theta = 2 * np.arcsin(np.sqrt(np.clip(p10, 0, 1)))
        circuito.ry(theta, 1)
        if p00 + p01 > 1e-12:
            phi = 2 * np.arctan2(np.sqrt(p01), np.sqrt(p00))
            circuito.x(1)
            circuito.ry(phi / 2, 0)
            circuito.cx(1, 0)
            circuito.ry(-phi / 2, 0)
            circuito.cx(1, 0)
            circuito.x(1)
        circuito.measure([0, 1], [0, 1])
        return circuito

    def decidir(self, acuerdo: str) -> tuple[str, Any, str, dict[str, float]]:
        probabilidades = self.probabilidades(acuerdo)
        circuito = self.crear_circuito(probabilidades)
        resultado = self.simulador.run(circuito, shots=1).result()
        conteos = resultado.get_counts()
        estado = next(iter(conteos))
        if estado not in ESTADOS_CUANTICOS:
            raise RuntimeError(f"El simulador devolvió un estado inesperado: {estado}")
        return ESTADOS_CUANTICOS[estado], circuito, estado, probabilidades

    def actualizar_estrategia(self, estado_jugador: str) -> None:
        if self.personalidad == "indiferente" or estado_jugador == "No decisoria":
            return
        actual = np.array([self.pc, self.pt, self.pn], dtype=float)
        if self.personalidad == "reciproco":
            if estado_jugador == "Cooperación":
                alpha, objetivo = 0.35, np.array([0.90, 0.05, 0.05])
            elif estado_jugador == "Traición":
                alpha, objetivo = 0.05, np.array([0.15, 0.75, 0.10])
            else:
                return
        elif self.personalidad == "rencoroso":
            if estado_jugador == "Cooperación":
                alpha, objetivo = 0.03, np.array([0.75, 0.15, 0.10])
            elif estado_jugador == "Traición":
                alpha, objetivo = 0.35, np.array([0.025, 0.95, 0.025])
            else:
                return
        else:
            return
        nueva = (1 - alpha) * actual + alpha * objetivo
        nueva = np.clip(nueva, 0, 1)
        nueva /= nueva.sum()
        self.pc, self.pt, self.pn = (float(v) for v in nueva)


class QuantumPrisonersDilemma:
    """Coordina partidas, acuerdos por ronda, puntuación y adivinanza final."""

    def __init__(self, total_rounds: int = 15, seed: int | None = None) -> None:
        if total_rounds < 1:
            raise ValueError("total_rounds debe ser al menos 1")
        self.total_rounds = int(total_rounds)
        self.rng = np.random.default_rng(seed)
        self.simulador = AerSimulator()
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.maquina = MaquinaCuantica(rng=self.rng, simulador=self.simulador)
            self.historial: list[dict[str, Any]] = []
            self.puntos_jugador = 0
            self.puntos_maquina = 0
            self.current_agreement = elegir_acuerdo(self.rng)
            self.guess_submitted = False
            self.guess_correct: bool | None = None
            self.guessed_personality: str | None = None
            self._last_record: dict[str, Any] | None = None
            return self.state()

    def _strategy_state(self, agreement: str | None) -> dict[str, float] | None:
        if agreement is None:
            return None
        return {
            "cooperation": round(float(self.maquina.pc), 6),
            "betrayal": round(float(self.maquina.pt), 6),
            "no_decision": round(float(self.maquina.pn), 6),
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            round_number = len(self.historial) + 1
            game_over = len(self.historial) >= self.total_rounds
            agreement = None if game_over else self.current_agreement
            revealed = self.guess_submitted
            state = {
                "ok": True,
                "total_rounds": self.total_rounds,
                "rounds_completed": len(self.historial),
                "round_number": min(round_number, self.total_rounds),
                "game_over": game_over,
                "current_agreement": agreement,
                "scores": {"player": self.puntos_jugador, "machine": self.puntos_maquina},
                "history": list(self.historial),
                "strategy_probabilities": self._strategy_state(agreement),
                "last_record": self._last_record,
                "personality_revealed": revealed,
                "personality": self.maquina.personalidad if revealed else None,
                "guess_submitted": self.guess_submitted,
                "guess_correct": self.guess_correct,
                "guessed_personality": self.guessed_personality,
            }
            return state

    def play_round(self, player_move: str) -> dict[str, Any]:
        with self._lock:
            if len(self.historial) >= self.total_rounds:
                raise ValueError("La partida ya terminó")
            try:
                decision_jugador = MOVIMIENTO_UI_A_SIMBOLO[str(player_move).strip().lower()]
            except KeyError as exc:
                raise ValueError("Movimiento del jugador no válido") from exc

            agreement = self.current_agreement
            probs_before = self._strategy_state(agreement)
            decision_maquina, circuito, estado_cuantico, distribucion_simbolos = self.maquina.decidir(agreement)
            estado_jugador = clasificar_decision(decision_jugador, agreement)
            estado_maquina = clasificar_decision(decision_maquina, agreement)
            resultado = piedra_papel_tijeras(decision_jugador, decision_maquina)
            puntos_jugador, puntos_maquina = calcular_puntos(estado_jugador, estado_maquina, resultado)
            self.puntos_jugador += puntos_jugador
            self.puntos_maquina += puntos_maquina

            # Probabilidades expresadas por categoría respecto al acuerdo antes de medir.
            record = {
                "round": len(self.historial) + 1,
                "agreement": agreement,
                "player_move": decision_jugador,
                "machine_move": decision_maquina,
                "player_state": estado_jugador,
                "machine_state": estado_maquina,
                "winner": resultado,
                "player_points": puntos_jugador,
                "machine_points": puntos_maquina,
                "player_total": self.puntos_jugador,
                "machine_total": self.puntos_maquina,
                "quantum_measurement": estado_cuantico,
                "quantum_operations": int(circuito.size()),
                "probabilities_before": probs_before,
                "symbol_probabilities_before": {
                    key: round(float(value), 6)
                    for key, value in distribucion_simbolos.items()
                },
            }
            self.historial.append(record)
            self._last_record = record
            self.maquina.historial.append(record)
            self.maquina.actualizar_estrategia(estado_jugador)

            if len(self.historial) < self.total_rounds:
                self.current_agreement = elegir_acuerdo(self.rng)
            return {"ok": True, "record": record, "state": self.state()}

    def guess_personality(self, guess: str) -> dict[str, Any]:
        with self._lock:
            if len(self.historial) < self.total_rounds:
                raise ValueError("Completa todas las rondas antes de adivinar la personalidad")
            if self.guess_submitted:
                return {"ok": True, "state": self.state(), "already_submitted": True}
            guess = str(guess).strip().lower()
            if guess not in PERSONALIDADES:
                raise ValueError("Personalidad no válida")
            self.guessed_personality = guess
            self.guess_correct = guess == self.maquina.personalidad
            if self.guess_correct:
                self.puntos_jugador += 15
            self.guess_submitted = True
            return {"ok": True, "correct": self.guess_correct, "bonus": 15 if self.guess_correct else 0, "state": self.state()}


__all__ = [
    "QuantumPrisonersDilemma", "MaquinaCuantica", "SIMBOLOS", "PERSONALIDADES",
    "simbolo_que_gana", "simbolo_que_pierde", "clasificar_decision",
    "calcular_puntos", "piedra_papel_tijeras", "elegir_acuerdo",
]
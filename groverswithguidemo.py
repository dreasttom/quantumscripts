#!/usr/bin/env python3
"""
grover_gui_demo.py

A student-friendly GUI demo of Grover's algorithm using Tkinter + Matplotlib.

Start with n=3 or 4, target like 101 or 1010.

Leave iterations blank first (auto), then try +1 and -1 to show overshoot.

Increase shots (e.g., 2048) to make the histogram more stable.

What students will see:
- Input boxes for:
    - number of qubits n
    - marked (target) state (bitstring or integer)
    - number of Grover iterations
    - number of measurement shots
- A step-by-step log of what the algorithm does
- A live bar chart that shows probability mass concentrating on the target state

Two modes:
1) If Qiskit is installed:
   - uses a Qiskit statevector computation after each iteration (ideal probabilities)
   - and a final shots-based measurement simulation
2) If Qiskit is not installed:
   - uses a pure-Python amplitude simulation (toy model) that is ideal for teaching

Recommended for class:
- n = 2..8 (small enough to visualize)
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

# Matplotlib is used to draw the probability bar chart inside the Tkinter window.
# Install with: pip install matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# Utility helpers (parsing, formatting)
# =============================================================================

def is_bitstring(s: str) -> bool:
    """True if s is a non-empty string containing only '0' and '1'."""
    return len(s) > 0 and all(ch in "01" for ch in s)


def int_to_bitstring(x: int, n: int) -> str:
    """Convert integer x into an n-bit binary string."""
    return format(x, f"0{n}b")


def bitstring_to_int(b: str) -> int:
    """Convert a bitstring (like '1011') to an integer."""
    return int(b, 2)


def choose_default_iterations(n: int) -> int:
    """
    Rule of thumb for ONE marked item:
        k ≈ floor((pi/4) * sqrt(N)), where N=2^n.
    """
    N = 2 ** n
    return max(1, int(math.floor((math.pi / 4.0) * math.sqrt(N))))


def top_k_probs(probs: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k outcomes by probability, sorted descending."""
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]


# =============================================================================
# Pure Python "toy" Grover simulation (always available)
# =============================================================================

@dataclass
class ToyGroverState:
    """
    A simple representation of Grover's algorithm state for teaching.

    - We store a real-valued amplitude vector for N=2^n states.
    - This is enough for Grover, since the toy oracle is just a sign flip,
      and the diffuser keeps amplitudes real (in this simplified setting).
    """
    n: int
    target_index: int
    amplitudes: List[float]  # length N

    @property
    def N(self) -> int:
        return 2 ** self.n

    def prob_dict(self) -> Dict[str, float]:
        """Return a probability distribution {bitstring: probability}."""
        out: Dict[str, float] = {}
        for i, a in enumerate(self.amplitudes):
            out[int_to_bitstring(i, self.n)] = a * a
        return out

    def oracle(self) -> None:
        """
        Oracle step:
            |x> -> -|x> if x == target
        i.e., flip the sign of the amplitude on the marked state.
        """
        self.amplitudes[self.target_index] *= -1.0

    def diffuser(self) -> None:
        """
        Diffuser (inversion about the mean):
            a_i <- 2*avg(a) - a_i
        """
        avg = sum(self.amplitudes) / self.N
        for i in range(self.N):
            self.amplitudes[i] = 2.0 * avg - self.amplitudes[i]


def toy_initialize(n: int, target_index: int) -> ToyGroverState:
    """Initialize to a uniform superposition state (all amplitudes equal)."""
    N = 2 ** n
    amp0 = 1.0 / math.sqrt(N)
    return ToyGroverState(n=n, target_index=target_index, amplitudes=[amp0] * N)


def toy_sample_measurements(probs: Dict[str, float], shots: int) -> Dict[str, int]:
    """Sample measurement outcomes according to the given distribution."""
    outcomes = list(probs.keys())
    weights = [probs[o] for o in outcomes]
    counts: Dict[str, int] = {}
    for _ in range(shots):
        pick = random.choices(outcomes, weights=weights, k=1)[0]
        counts[pick] = counts.get(pick, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# =============================================================================
# Optional Qiskit-backed probability computation (if installed)
# =============================================================================

def qiskit_available() -> bool:
    """Return True if Qiskit + Aer appear installed."""
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        return True
    except Exception:
        return False


def qiskit_build_oracle(n: int, target_bits: str):
    """
    Build a phase-flip oracle for a specific marked bitstring.

    Method:
    - Apply X on qubits where target bit is 0 (map |target> -> |11..1>)
    - Apply multi-controlled Z phase flip on |11..1>
      (implemented as H, MCX, H on last qubit)
    - Undo the X gates
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name="Oracle")

    # Qiskit qubit 0 is usually the least significant bit.
    # We'll iterate over reversed bitstring so bits map naturally.
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            qc.x(i)

    # Multi-controlled Z via H + MCX + H on last qubit
    qc.h(n - 1)
    if n == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            qc.x(i)

    return qc


def qiskit_build_diffuser(n: int):
    """
    Build the Grover diffuser (inversion about the mean):
      H^n -> X^n -> phase flip on |00..0> -> X^n -> H^n
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))

    qc.h(n - 1)
    if n == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))
    return qc


def qiskit_probs_after_each_iteration(n: int, target_bits: str, iterations: int) -> List[Dict[str, float]]:
    """
    Return a list of probability distributions:
      probs_list[0] = after initial H...H
      probs_list[1] = after 1 Grover iteration
      ...
      probs_list[k] = after k Grover iterations

    This uses Statevector simulation (ideal, no noise).
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    oracle = qiskit_build_oracle(n, target_bits)
    diffuser = qiskit_build_diffuser(n)

    step_circ = QuantumCircuit(n)
    step_circ.h(range(n))

    probs_list: List[Dict[str, float]] = []
    sv = Statevector.from_instruction(step_circ)
    probs0 = {k.zfill(n): float(v) for k, v in sv.probabilities_dict().items()}
    probs_list.append(probs0)

    for _ in range(iterations):
        step_circ.compose(oracle, inplace=True)
        step_circ.compose(diffuser, inplace=True)
        sv = Statevector.from_instruction(step_circ)
        probsi = {k.zfill(n): float(v) for k, v in sv.probabilities_dict().items()}
        probs_list.append(probsi)

    return probs_list


def qiskit_measure_counts(n: int, target_bits: str, iterations: int, shots: int) -> Dict[str, int]:
    """Run a shots-based measurement simulation of Grover circuit and return counts."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import Aer

    oracle = qiskit_build_oracle(n, target_bits)
    diffuser = qiskit_build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n), range(n))

    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=shots).result()
    counts_raw = result.get_counts()

    counts = {k.zfill(n): int(v) for k, v in counts_raw.items()}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# =============================================================================
# GUI Application
# =============================================================================

class GroverGUIDemo(tk.Tk):
    """
    A Tkinter window that:
    - collects inputs
    - runs Grover step-by-step
    - updates a plot and log area
    """

    def __init__(self) -> None:
        super().__init__()

        self.title("Grover's Algorithm GUI Demo (Student Version)")
        self.geometry("1050x650")

        # A flag to prevent multiple runs at once
        self._running = False

        # Build the UI
        self._build_controls()
        self._build_plot()
        self._build_log()

        # Default: show whether Qiskit is available
        self._set_status()

    # -------------------------
    # UI Construction
    # -------------------------

    def _build_controls(self) -> None:
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        # Inputs
        ttk.Label(frame, text="n qubits:").grid(row=0, column=0, sticky="w")
        self.n_var = tk.StringVar(value="4")
        ttk.Entry(frame, textvariable=self.n_var, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(frame, text="target (bitstring or int):").grid(row=0, column=2, sticky="w")
        self.target_var = tk.StringVar(value="1010")
        ttk.Entry(frame, textvariable=self.target_var, width=16).grid(row=0, column=3, padx=6)

        ttk.Label(frame, text="iterations (blank = auto):").grid(row=0, column=4, sticky="w")
        self.iter_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.iter_var, width=10).grid(row=0, column=5, padx=6)

        ttk.Label(frame, text="shots:").grid(row=0, column=6, sticky="w")
        self.shots_var = tk.StringVar(value="512")
        ttk.Entry(frame, textvariable=self.shots_var, width=10).grid(row=0, column=7, padx=6)

        # Buttons
        self.run_btn = ttk.Button(frame, text="Run Demo", command=self.on_run_clicked)
        self.run_btn.grid(row=0, column=8, padx=10)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self.on_stop_clicked, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=9, padx=5)

        self.reset_btn = ttk.Button(frame, text="Reset", command=self.on_reset_clicked)
        self.reset_btn.grid(row=0, column=10, padx=5)

        # Status line
        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var).pack(side=tk.TOP, fill=tk.X, padx=10)

    def _build_plot(self) -> None:
        """
        Create a Matplotlib bar chart embedded in Tkinter.
        We'll display the top outcomes and their probabilities.
        """
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Top Outcome Probabilities (updates each iteration)")
        self.ax.set_xlabel("State (bitstring)")
        self.ax.set_ylabel("Probability")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_log(self) -> None:
        """
        A scrollable text box to show step-by-step messages for students.
        """
        log_frame = ttk.Frame(self)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=8)

        ttk.Label(log_frame, text="Step-by-step log").pack(anchor="w")

        self.log_text = tk.Text(log_frame, width=48, height=32, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scroll.set)

    # -------------------------
    # Status and logging helpers
    # -------------------------

    def _set_status(self) -> None:
        mode = "Qiskit mode ✅" if qiskit_available() else "Toy mode (no Qiskit) ✅"
        self.status_var.set(
            f"Mode: {mode}    |    Tip: Use n=2..8 for visualization. Too many iterations can 'overshoot'."
        )

    def log(self, msg: str) -> None:
        """Append a message to the log box."""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    # -------------------------
    # Plotting helper
    # -------------------------

    def update_plot(self, probs: Dict[str, float], target_bits: str, top_k: int = 10) -> None:
        """
        Update bar chart with the top_k most likely outcomes.
        Highlight target label by appending a marker to its x-tick label.
        """
        self.ax.clear()
        self.ax.set_title("Top Outcome Probabilities (updates each iteration)")
        self.ax.set_xlabel("State (bitstring)")
        self.ax.set_ylabel("Probability")

        top = top_k_probs(probs, k=top_k)
        labels = []
        values = []
        for b, p in top:
            if b == target_bits:
                labels.append(b + "  ← target")
            else:
                labels.append(b)
            values.append(p)

        # Plot bars (we do not set explicit colors to keep things simple/neutral)
        self.ax.bar(range(len(values)), values)

        self.ax.set_xticks(range(len(values)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_ylim(0, 1.0)

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------------
    # Input parsing
    # -------------------------

    def parse_inputs(self) -> Optional[Tuple[int, int, str, int, int]]:
        """
        Parse GUI inputs into:
          n, target_index, target_bits, iterations, shots
        """
        try:
            n = int(self.n_var.get().strip())
            if n < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "n must be an integer >= 1.")
            return None

        N = 2 ** n

        raw_target = self.target_var.get().strip()
        if is_bitstring(raw_target):
            if len(raw_target) != n:
                messagebox.showerror("Input error", f"Target bitstring must have length {n}.")
                return None
            target_bits = raw_target
            target_index = bitstring_to_int(target_bits)
        else:
            try:
                target_index = int(raw_target)
            except Exception:
                messagebox.showerror("Input error", "Target must be a bitstring (e.g., 1010) or an integer.")
                return None
            if not (0 <= target_index < N):
                messagebox.showerror("Input error", f"Target integer must be in [0, {N-1}].")
                return None
            target_bits = int_to_bitstring(target_index, n)

        raw_iters = self.iter_var.get().strip()
        if raw_iters == "":
            iterations = choose_default_iterations(n)
        else:
            try:
                iterations = int(raw_iters)
            except Exception:
                messagebox.showerror("Input error", "Iterations must be an integer, or blank for auto.")
                return None
            if iterations < 1:
                messagebox.showerror("Input error", "Iterations must be >= 1.")
                return None

        try:
            shots = int(self.shots_var.get().strip())
            if shots < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "Shots must be an integer >= 1.")
            return None

        return (n, target_index, target_bits, iterations, shots)

    # -------------------------
    # Button handlers
    # -------------------------

    def on_reset_clicked(self) -> None:
        """Reset the log and plot."""
        if self._running:
            messagebox.showinfo("Busy", "Stop the demo before resetting.")
            return

        self.clear_log()
        self.ax.clear()
        self.ax.set_title("Top Outcome Probabilities (updates each iteration)")
        self.ax.set_xlabel("State (bitstring)")
        self.ax.set_ylabel("Probability")
        self.canvas.draw()
        self._set_status()

    def on_stop_clicked(self) -> None:
        """
        Stop is implemented by setting a flag that the run loop checks.
        """
        self._running = False
        self.log("🛑 Stop requested. (Will stop after the current step.)")

    def on_run_clicked(self) -> None:
        """Start the demo in a background thread so the GUI doesn't freeze."""
        if self._running:
            messagebox.showinfo("Busy", "The demo is already running.")
            return

        parsed = self.parse_inputs()
        if parsed is None:
            return

        n, target_index, target_bits, iterations, shots = parsed

        self.clear_log()
        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Run in a thread to keep GUI responsive
        t = threading.Thread(
            target=self.run_demo,
            args=(n, target_index, target_bits, iterations, shots),
            daemon=True,
        )
        t.start()

    # -------------------------
    # The main demo logic
    # -------------------------

    def run_demo(self, n: int, target_index: int, target_bits: str, iterations: int, shots: int) -> None:
        """
        Executes Grover step-by-step, updating log and plot.

        We add small delays so students can visually observe changes.
        """
        try:
            N = 2 ** n
            default_iters = choose_default_iterations(n)

            self.log("=== Grover's Algorithm Demo ===")
            self.log(f"n = {n} qubits  ->  N = 2^n = {N} states")
            self.log(f"Marked (target) state = {target_bits} (decimal {target_index})")
            self.log(f"Iterations = {iterations}  (default suggestion would be ~{default_iters})")
            self.log(f"Shots (measurement samples) = {shots}")
            self.log("")
            self.log("Algorithm outline:")
            self.log("  1) Create uniform superposition with H on all qubits")
            self.log("  2) Repeat k times:")
            self.log("       - Oracle: phase-flip the target state")
            self.log("       - Diffuser: invert amplitudes about the mean")
            self.log("  3) Measure (many shots) -> target should appear most often")
            self.log("")

            use_qiskit = qiskit_available()
            self.log(f"Mode selected: {'Qiskit (statevector + shots)' if use_qiskit else 'Toy (pure Python)'}")
            self.log("")

            # STEP-BY-STEP probabilities
            if use_qiskit:
                probs_list = qiskit_probs_after_each_iteration(n, target_bits, iterations)

                # Step 0: after Hadamards
                for step, probs in enumerate(probs_list):
                    if not self._running:
                        break

                    p_target = probs.get(target_bits, 0.0)
                    if step == 0:
                        self.log(f"Step 0: after initialization (H...H). P(target) = {p_target:.6f}")
                    else:
                        self.log(f"Step {step}: after oracle+diffuser. P(target) = {p_target:.6f}")

                    # Update plot in GUI thread
                    self.after(0, self.update_plot, probs, target_bits, 10)
                    time.sleep(0.75)

                if self._running:
                    self.log("\nNow sampling measurements (shots) from the final circuit...")
                    counts = qiskit_measure_counts(n, target_bits, iterations, shots)
                else:
                    counts = {}

                final_probs = probs_list[-1] if probs_list else {}

            else:
                # Toy mode: we evolve amplitudes and compute probabilities per step
                state = toy_initialize(n, target_index)

                # Step 0
                probs0 = state.prob_dict()
                self.log(f"Step 0: after initialization (uniform). P(target) = {probs0[target_bits]:.6f}")
                self.after(0, self.update_plot, probs0, target_bits, 10)
                time.sleep(0.75)

                for step in range(1, iterations + 1):
                    if not self._running:
                        break

                    # One Grover iteration = oracle + diffuser
                    state.oracle()
                    state.diffuser()

                    probsi = state.prob_dict()
                    self.log(f"Step {step}: after oracle+diffuser. P(target) = {probsi[target_bits]:.6f}")
                    self.after(0, self.update_plot, probsi, target_bits, 10)
                    time.sleep(0.75)

                final_probs = state.prob_dict()
                if self._running:
                    self.log("\nNow sampling measurements (shots) from the final probability distribution...")
                    counts = toy_sample_measurements(final_probs, shots)
                else:
                    counts = {}

            # FINAL OUTPUT
            if self._running:
                self.log("\n=== Final results ===")
                self.log("Top ideal probabilities (final state):")
                for b, p in top_k_probs(final_probs, k=10):
                    mark = "  <-- target" if b == target_bits else ""
                    self.log(f"  {b}: {p:.6f}{mark}")

                self.log("\nMeasurement counts (top outcomes):")
                top_counts = list(counts.items())[:10]
                for b, c in top_counts:
                    mark = "  <-- target" if b == target_bits else ""
                    self.log(f"  {b}: {c}{mark}")

                if counts:
                    best = max(counts, key=counts.get)
                    self.log(f"\nMost frequent measured outcome: {best} (decimal {bitstring_to_int(best)})")
                    if best == target_bits:
                        self.log("✅ Success: the target is the most frequent outcome!")
                    else:
                        self.log("ℹ️ Not the top outcome this time. Try:")
                        self.log("   - using the default iteration count, or")
                        self.log("   - increasing shots, or")
                        self.log("   - reducing/adjusting iterations (overshoot can happen).")

                self.log("\nTeaching note: Grover amplification can overshoot if you do too many iterations.")
                self.log("Try a few iteration values and watch P(target) rise and then fall.")

        except Exception as e:
            # If something unexpected happens, show a friendly error
            self.after(0, messagebox.showerror, "Runtime error", f"Something went wrong:\n{e}")

        finally:
            # Reset UI state
            self._running = False
            self.after(0, self.run_btn.config, {"state": tk.NORMAL})
            self.after(0, self.stop_btn.config, {"state": tk.DISABLED})


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    app = GroverGUIDemo()
    app.mainloop()


if __name__ == "__main__":
    main()

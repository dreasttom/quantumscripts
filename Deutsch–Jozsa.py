#!/usr/bin/env python3
"""
deutsch_jozsa_visual_demo.py
============================

A heavily-commented, student-friendly implementation of the Deutsch–Jozsa algorithm
WITH a visual demo.

Deutsch–Jozsa problem (promise problem):
- You are given an oracle for a Boolean function f:{0,1}^n -> {0,1}
- PROMISE: f is either
    (A) CONSTANT: f(x)=0 for all x OR f(x)=1 for all x
    (B) BALANCED: f(x)=0 for exactly half the inputs and 1 for the other half
- Goal: determine whether f is constant or balanced.

Classically:
- In the worst case, you need many queries.
Quantumly (Deutsch–Jozsa):
- You can decide with 1 oracle query (under the promise).

This script:
- Prompts for n (number of input qubits).
- Lets students choose an oracle type:
    1) constant-0
    2) constant-1
    3) balanced (a standard DJ balanced oracle based on a hidden bitmask "a")
- Builds the Deutsch–Jozsa circuit
- Simulates it (Qiskit Aer if installed)
- Shows intermediate steps:
    - circuit drawing
    - statevector probabilities after key stages (optional)
- Displays a GUI with a bar chart of measurement results + a clear conclusion.

Requirements:
- Python 3.x
- matplotlib
- tkinter (usually included with Python)
- Qiskit + qiskit-aer (recommended)

Install:
    pip install matplotlib
    pip install qiskit qiskit-aer

Run:
    python deutsch_jozsa_visual_demo.py
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# 1) Qiskit import helper
# =============================================================================

def try_import_qiskit():
    """
    Try importing the pieces of Qiskit we need.
    If not installed, we return None and show a helpful message.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import Aer
        from qiskit.quantum_info import Statevector
        return QuantumCircuit, transpile, Aer, Statevector
    except Exception:
        return None


# =============================================================================
# 2) Build Deutsch–Jozsa oracles (as quantum circuits)
# =============================================================================

def build_constant_oracle(QuantumCircuit, n: int, value: int):
    """
    Build a CONSTANT oracle U_f for f(x)=value for all x.

    Oracle definition:
      U_f |x>|y> = |x>|y XOR f(x)>

    If f(x)=0 always:
      U_f does nothing to y (identity).

    If f(x)=1 always:
      U_f flips y for every input x (so just apply an X gate to y).
    """
    qc = QuantumCircuit(n + 1, name=f"Const({value})")

    # The last qubit (index n) is the output/ancilla y
    if value == 1:
        qc.x(n)  # y <- y XOR 1

    return qc


def build_balanced_oracle_bitmask(QuantumCircuit, n: int, a_bits: str):
    """
    Build a standard BALANCED Deutsch–Jozsa oracle:
        f(x) = a · x (mod 2)
    where:
      - a is a non-zero n-bit string (bitmask)
      - "·" is bitwise dot product mod 2 (XOR of ANDs)

    This function is guaranteed BALANCED when a != 0...0:
      Half of inputs give 0, half give 1.

    How to implement U_f:
      U_f|x>|y> = |x>|y XOR (a·x)>

    We can implement y XOR (a·x) using CNOTs:
      For each i where a_i = 1, do CNOT(control=x_i, target=y).

    NOTE on bit order:
      We'll interpret a_bits as a string like "1011" for n=4
      where the LEFTMOST character is the highest-order bit.
      Qiskit qubit 0 is typically the least significant. We'll map carefully.
    """
    if len(a_bits) != n:
        raise ValueError("a_bits must have length n")
    if all(b == "0" for b in a_bits):
        raise ValueError("a_bits must not be all zeros for a balanced oracle")

    qc = QuantumCircuit(n + 1, name=f"Balanced(a={a_bits})")
    y = n  # output qubit index

    # Map bits to qubits: a_bits[-1] corresponds to qubit 0 (LSB) in typical convention
    for qubit_index, bit in enumerate(reversed(a_bits)):
        if bit == "1":
            qc.cx(qubit_index, y)

    return qc


# =============================================================================
# 3) Build the full Deutsch–Jozsa algorithm circuit
# =============================================================================

def build_deutsch_jozsa_circuit(QuantumCircuit, n: int, oracle_circuit):
    """
    Build the Deutsch–Jozsa circuit:

    Registers:
      - n input qubits: |0...0>
      - 1 output qubit: |1> (we will prepare it as |->)

    Steps:
      1) Prepare output qubit in |1>, then apply H to make |-> = (|0>-|1>)/sqrt(2)
      2) Apply H to all input qubits to create uniform superposition over x
      3) Apply oracle U_f
      4) Apply H to input qubits again
      5) Measure input qubits:
           - If result is all zeros => f is CONSTANT
           - Otherwise => f is BALANCED
    """
    qc = QuantumCircuit(n + 1, n)

    # Output qubit is the last qubit
    y = n

    # Step 1: put output in |1>
    qc.x(y)
    # Then H to get |-> (phase kickback works nicely with |->)
    qc.h(y)

    # Step 2: Hadamards on input register
    qc.h(range(n))

    # Step 3: apply the oracle
    qc.compose(oracle_circuit, inplace=True)

    # Step 4: Hadamards again on input register
    qc.h(range(n))

    # Step 5: measure ONLY the input qubits
    qc.measure(range(n), range(n))

    return qc


# =============================================================================
# 4) Simulation helpers: get counts and (optionally) intermediate statevectors
# =============================================================================

def simulate_counts(transpile, Aer, qc, shots: int = 1024) -> Dict[str, int]:
    """Run the circuit with measurements and return a counts dictionary."""
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Ensure keys are strings and counts are ints
    return {str(k): int(v) for k, v in counts.items()}


def statevector_after_stages(QuantumCircuit, Statevector, n: int, oracle):
    """
    OPTIONAL: show intermediate ideal state probabilities (no measurement).

    We'll build three circuits:
      - After initialization (H on input, |-> on output)
      - After oracle
      - After final H on input (right before measurement)

    This is great for teaching, because students can see that:
      - For constant oracles: input register ends as |0...0> (prob 1)
      - For balanced oracles: probability of |0...0> becomes 0 (ideally)
    """
    y = n

    # 1) After init
    c1 = QuantumCircuit(n + 1)
    c1.x(y)
    c1.h(y)
    c1.h(range(n))
    sv1 = Statevector.from_instruction(c1)
    p1 = sv1.probabilities_dict()

    # 2) After oracle
    c2 = c1.copy()
    c2.compose(oracle, inplace=True)
    sv2 = Statevector.from_instruction(c2)
    p2 = sv2.probabilities_dict()

    # 3) After final H
    c3 = c2.copy()
    c3.h(range(n))
    sv3 = Statevector.from_instruction(c3)
    p3 = sv3.probabilities_dict()

    # The full state includes the output qubit, so bitstrings are length n+1.
    # We'll return them for completeness.
    return p1, p2, p3


# =============================================================================
# 5) A simple GUI to visualize the results
# =============================================================================

@dataclass
class DJRunResult:
    n: int
    oracle_name: str
    shots: int
    counts: Dict[str, int]
    conclusion: str
    p_stage0: Optional[Dict[str, float]] = None
    p_stage1: Optional[Dict[str, float]] = None
    p_stage2: Optional[Dict[str, float]] = None


class DeutschJozsaGUI(tk.Tk):
    """
    A GUI that:
    - Lets students choose n, oracle type, and (for balanced) a-bitmask
    - Runs the algorithm
    - Shows:
        - Circuit text
        - Bar chart of measurement counts
        - Clear conclusion (constant vs balanced)
        - Optional intermediate stage notes
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("Deutsch–Jozsa Algorithm Visual Demo")
        self.geometry("1100x700")

        self.qiskit = try_import_qiskit()
        self._build_ui()

        if self.qiskit is None:
            self._disable_run_with_message()

    def _disable_run_with_message(self) -> None:
        self.run_btn.configure(state=tk.DISABLED)
        messagebox.showwarning(
            "Qiskit not found",
            "This demo needs Qiskit + qiskit-aer.\n\n"
            "Install with:\n"
            "  pip install qiskit qiskit-aer\n\n"
            "Then rerun this script."
        )

    def _build_ui(self) -> None:
        # Top controls
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(controls, text="n (input qubits):").grid(row=0, column=0, sticky="w")
        self.n_var = tk.StringVar(value="3")
        ttk.Entry(controls, textvariable=self.n_var, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(controls, text="shots:").grid(row=0, column=2, sticky="w")
        self.shots_var = tk.StringVar(value="1024")
        ttk.Entry(controls, textvariable=self.shots_var, width=10).grid(row=0, column=3, padx=6)

        ttk.Label(controls, text="oracle type:").grid(row=0, column=4, sticky="w")
        self.oracle_var = tk.StringVar(value="constant-0")
        oracle_menu = ttk.Combobox(
            controls,
            textvariable=self.oracle_var,
            values=["constant-0", "constant-1", "balanced (a·x mod 2)"],
            width=22,
            state="readonly"
        )
        oracle_menu.grid(row=0, column=5, padx=6)

        ttk.Label(controls, text="balanced mask a (bitstring):").grid(row=0, column=6, sticky="w")
        self.a_var = tk.StringVar(value="101")  # default for n=3
        ttk.Entry(controls, textvariable=self.a_var, width=12).grid(row=0, column=7, padx=6)

        self.show_intermediate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="show intermediate stages", variable=self.show_intermediate_var)\
            .grid(row=0, column=8, padx=10)

        self.run_btn = ttk.Button(controls, text="Run Deutsch–Jozsa", command=self.run_demo)
        self.run_btn.grid(row=0, column=9, padx=6)

        # Main split: left text/circuit, right plot
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(left, text="Circuit + Explanation (student notes)", font=("Arial", 12, "bold")).pack(anchor="w")
        self.text = tk.Text(left, wrap="word", height=32)
        self.text.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Measurement Results (bar chart)", font=("Arial", 12, "bold")).pack(anchor="w")
        self.fig = Figure(figsize=(5.2, 4.5), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Counts")
        self.ax.set_xlabel("Measured input bitstring")
        self.ax.set_ylabel("Counts")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _write(self, s: str) -> None:
        self.text.insert(tk.END, s + "\n")
        self.text.see(tk.END)

    def _clear_text(self) -> None:
        self.text.delete("1.0", tk.END)

    def run_demo(self) -> None:
        if self.qiskit is None:
            return

        QuantumCircuit, transpile, Aer, Statevector = self.qiskit

        # Parse inputs
        try:
            n = int(self.n_var.get().strip())
            if n < 1 or n > 10:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "n must be an integer between 1 and 10 (try 2..6).")
            return

        try:
            shots = int(self.shots_var.get().strip())
            if shots < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "shots must be a positive integer.")
            return

        oracle_type = self.oracle_var.get().strip()

        # Build oracle
        oracle = None
        oracle_name = ""

        if oracle_type == "constant-0":
            oracle = build_constant_oracle(QuantumCircuit, n, value=0)
            oracle_name = "constant-0 (f(x)=0 for all x)"
        elif oracle_type == "constant-1":
            oracle = build_constant_oracle(QuantumCircuit, n, value=1)
            oracle_name = "constant-1 (f(x)=1 for all x)"
        else:
            a_bits = self.a_var.get().strip()
            if not (len(a_bits) == n and all(c in "01" for c in a_bits)):
                messagebox.showerror("Input error", f"a must be a bitstring of length {n}.")
                return
            if all(c == "0" for c in a_bits):
                messagebox.showerror("Input error", "a must not be all zeros (that would be constant, not balanced).")
                return
            oracle = build_balanced_oracle_bitmask(QuantumCircuit, n, a_bits=a_bits)
            oracle_name = f"balanced (f(x)=a·x mod 2), a={a_bits}"

        # Build full Deutsch–Jozsa circuit
        dj = build_deutsch_jozsa_circuit(QuantumCircuit, n, oracle)

        # Run simulation and get counts
        counts = simulate_counts(transpile, Aer, dj, shots=shots)

        # Decision rule:
        # - In ideal DJ, constant => measure all zeros with probability 1
        # - balanced => measure NOT all zeros with probability 1
        all_zeros = "0" * n
        zeros_count = counts.get(all_zeros, 0)
        conclusion = "CONSTANT" if zeros_count == shots else "BALANCED"

        # Optional intermediate stage probabilities
        p0 = p1 = p2 = None
        if self.show_intermediate_var.get():
            p0, p1, p2 = statevector_after_stages(QuantumCircuit, Statevector, n, oracle)

        # Update GUI: text + plot
        self._clear_text()
        self._write("=== Deutsch–Jozsa Algorithm Demo ===")
        self._write(f"n = {n} input qubits  ->  N = 2^n = {2**n} inputs")
        self._write(f"Oracle type: {oracle_name}")
        self._write(f"Shots: {shots}")
        self._write("")
        self._write("PROMISE PROBLEM:")
        self._write("  f is either CONSTANT or BALANCED. DJ decides which with ONE oracle call.")
        self._write("")
        self._write("Circuit steps (high-level):")
        self._write("  1) Prepare |0...0>|1> then H on all qubits -> uniform superposition + |->")
        self._write("  2) Apply oracle U_f (phase kickback happens because output is |->)")
        self._write("  3) Apply H to input register")
        self._write("  4) Measure inputs:")
        self._write("      - if measurement is 00...0 => CONSTANT")
        self._write("      - else => BALANCED")
        self._write("")
        self._write("=== Circuit (ASCII) ===")
        self._write(str(dj.draw(output="text")))
        self._write("")
        self._write("=== Measurement results ===")
        self._write(f"Counts for all-zeros ({all_zeros}): {zeros_count} / {shots}")
        self._write(f"Conclusion: {conclusion}")
        self._write("")

        if p0 is not None and p1 is not None and p2 is not None:
            self._write("=== Intermediate (ideal) notes ===")
            self._write("These are IDEAL statevector probabilities over ALL qubits (including output).")
            self._write("We mainly care about what happens to the input register after the final H.")
            self._write("")
            self._write("Stage 0: after initialization (H on inputs, output in |->)")
            self._write(f"  (Showing a few largest probabilities)")
            self._write(self._top_probs_text(p0, top=6))
            self._write("")
            self._write("Stage 1: after applying the oracle")
            self._write(self._top_probs_text(p1, top=6))
            self._write("")
            self._write("Stage 2: after final H on inputs (right before measurement)")
            self._write("  Key idea:")
            self._write("    - constant -> input collapses to |00..0> with prob ~1")
            self._write("    - balanced -> probability of |00..0> ideally becomes 0")
            self._write(self._top_probs_text(p2, top=6))
            self._write("")

        self._update_plot(counts, n)

    def _top_probs_text(self, probs: Dict[str, float], top: int = 6) -> str:
        # Show the top 'top' probabilities for readability
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top]
        lines = []
        for bitstr, p in items:
            lines.append(f"  |{bitstr}> : {p:.6f}")
        return "\n".join(lines)

    def _update_plot(self, counts: Dict[str, int], n: int) -> None:
        self.ax.clear()
        self.ax.set_title("Measured input register outcomes")
        self.ax.set_xlabel("bitstring (input)")
        self.ax.set_ylabel("counts")

        # Sort keys by count (descending)
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        # Only show top outcomes for clarity
        items = items[:min(12, len(items))]
        labels = [k for k, _ in items]
        values = [v for _, v in items]

        self.ax.bar(range(len(values)), values)
        self.ax.set_xticks(range(len(values)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right")

        self.fig.tight_layout()
        self.canvas.draw()


# =============================================================================
# 6) Main entry point
# =============================================================================

def main() -> None:
    app = DeutschJozsaGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
entanglement_tutorial_gui.py

A student-friendly, heavily-commented Python script that displays an interactive
tutorial on quantum entanglement WITH graphics.

What you get:
- A simple GUI (Tkinter) with multiple "pages" of a tutorial
- Explanations in plain language
- Graphics made with Matplotlib:
    * Bloch-sphere-style sketches (simplified)
    * Probability bar charts
    * Correlation plots for entangled vs non-entangled states
    * A Bell-state “measurement outcomes” visualization

No quantum SDK required. (No Qiskit needed.)
Everything is computed using small, exact linear algebra (NumPy).

Requirements:
- Python 3.x
- numpy
- matplotlib
- tkinter (usually included with Python on Windows/macOS; on some Linux distros
  you may need to install it separately)

Install deps:
    pip install numpy matplotlib

Run:
    python entanglement_tutorial_gui.py
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# 1) Quantum basics we will use (small and explicit)
# =============================================================================

# We represent quantum states as column vectors.
# For a single qubit, the computational basis vectors are:
ket0 = np.array([[1.0], [0.0]], dtype=complex)  # |0>
ket1 = np.array([[0.0], [1.0]], dtype=complex)  # |1>

# For two qubits, the basis states are tensor products:
# |00> = |0> ⊗ |0>, |01> = |0> ⊗ |1>, |10> = |1> ⊗ |0>, |11> = |1> ⊗ |1>
def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience wrapper for Kronecker product (tensor product)."""
    return np.kron(a, b)


# Single-qubit gates we need: Hadamard (H) and Pauli matrices (optional)
H = (1 / math.sqrt(2)) * np.array([[1, 1],
                                  [1, -1]], dtype=complex)

I = np.eye(2, dtype=complex)

# Two-qubit CNOT gate in the basis order |00>,|01>,|10>,|11>
# Control = first qubit, target = second qubit
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


# =============================================================================
# 2) Helper functions for measurement probabilities and correlations
# =============================================================================

def normalize(state: np.ndarray) -> np.ndarray:
    """Return a normalized copy of 'state'."""
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("State has zero norm.")
    return state / norm


def probabilities_two_qubit(state: np.ndarray) -> Dict[str, float]:
    """
    Return measurement probabilities in the computational basis for a 2-qubit state.

    The order is:
      index 0 -> |00>
      index 1 -> |01>
      index 2 -> |10>
      index 3 -> |11>
    """
    state = state.reshape((4, 1))
    probs = np.abs(state[:, 0]) ** 2
    labels = ["00", "01", "10", "11"]
    return {labels[i]: float(probs[i].real) for i in range(4)}


def measure_first_qubit_conditional_probs(state: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Show conditional probabilities that illustrate entanglement nicely:

    - If we measure the FIRST qubit and get 0 or 1, what's the probability
      distribution of the SECOND qubit?

    This is a simplified “collapse” view in the computational basis.

    Returns:
      {
        "A=0": {"B=0": p00/(p00+p01), "B=1": p01/(p00+p01)},
        "A=1": {"B=0": p10/(p10+p11), "B=1": p11/(p10+p11)},
      }
    """
    p = probabilities_two_qubit(state)
    # probabilities for A=0 are p00+p01; for A=1 are p10+p11
    pA0 = p["00"] + p["01"]
    pA1 = p["10"] + p["11"]

    # Avoid division by zero (can happen for some states)
    out: Dict[str, Dict[str, float]] = {}
    if pA0 > 0:
        out["A=0"] = {"B=0": p["00"] / pA0, "B=1": p["01"] / pA0}
    else:
        out["A=0"] = {"B=0": 0.0, "B=1": 0.0}

    if pA1 > 0:
        out["A=1"] = {"B=0": p["10"] / pA1, "B=1": p["11"] / pA1}
    else:
        out["A=1"] = {"B=0": 0.0, "B=1": 0.0}

    return out


def correlation_ZZ(state: np.ndarray) -> float:
    """
    Compute the expectation value <Z ⊗ Z> for a two-qubit state.

    In the computational basis:
      Z = diag(1, -1)

    The correlation value lies in [-1, 1].
    For the Bell state |Φ+> = (|00> + |11>)/sqrt(2), <Z⊗Z> = +1.
    For |Ψ+> = (|01> + |10>)/sqrt(2), <Z⊗Z> = -1.
    """
    Z = np.array([[1, 0],
                  [0, -1]], dtype=complex)
    ZZ = kron(Z, Z)  # 4x4
    state = state.reshape((4, 1))
    return float((state.conj().T @ ZZ @ state).real.item())


# =============================================================================
# 3) Build example states (product vs entangled)
# =============================================================================

def product_state_example() -> np.ndarray:
    """
    A simple product state:
      |+> ⊗ |0>
    where |+> = (|0> + |1>)/sqrt(2).
    This is NOT entangled.
    """
    plus = normalize(ket0 + ket1)
    return kron(plus, ket0)


def bell_phi_plus() -> np.ndarray:
    """
    One of the four Bell states:
      |Φ+> = (|00> + |11>)/sqrt(2)
    This is maximally entangled.
    """
    return normalize(kron(ket0, ket0) + kron(ket1, ket1))


def make_bell_state_via_circuit() -> np.ndarray:
    """
    Show how entanglement is created by a short circuit:

      Start: |00>
      Apply H on qubit A (first qubit)
      Apply CNOT (A controls B)

    Result is |Φ+>.
    """
    # Start in |00>
    state = kron(ket0, ket0)  # shape (4,1)

    # Apply H on first qubit: (H ⊗ I) |00>
    HI = kron(H, I)  # 4x4
    state = HI @ state

    # Apply CNOT
    state = CNOT @ state

    return normalize(state)


# =============================================================================
# 4) GUI "pages" (each page has text + a plot)
# =============================================================================

@dataclass
class TutorialPage:
    title: str
    text: str


PAGES: List[TutorialPage] = [
    TutorialPage(
        title="What is entanglement?",
        text=(
            "Entanglement is a quantum connection between particles.\n\n"
            "In a product (non-entangled) state, each qubit has its own state:\n"
            "    |ψ> = |a> ⊗ |b>\n\n"
            "In an entangled state, you cannot write the two-qubit state that way.\n"
            "The state belongs to the pair as a whole.\n\n"
            "A famous entangled state is the Bell state:\n"
            "    |Φ+> = (|00> + |11>)/√2\n\n"
            "If you measure qubit A and get 0, qubit B is guaranteed to be 0.\n"
            "If you measure qubit A and get 1, qubit B is guaranteed to be 1.\n"
        ),
    ),
    TutorialPage(
        title="Product vs entangled (probabilities)",
        text=(
            "Let’s compare two states:\n\n"
            "1) Product state: |+> ⊗ |0>\n"
            "   where |+> = (|0> + |1>)/√2\n\n"
            "2) Entangled state: |Φ+> = (|00> + |11>)/√2\n\n"
            "Both are valid quantum states. But their measurement patterns differ.\n"
            "We will plot measurement probabilities for outcomes 00, 01, 10, 11.\n"
        ),
    ),
    TutorialPage(
        title="Collapse and conditional probabilities",
        text=(
            "Entanglement becomes very clear when we look at conditional probabilities.\n\n"
            "We measure the FIRST qubit (A). Then we ask:\n"
            "  - If A=0, what is the distribution of B?\n"
            "  - If A=1, what is the distribution of B?\n\n"
            "For the Bell state |Φ+>, the second qubit becomes perfectly determined.\n"
            "For the product state, it does not.\n"
        ),
    ),
    TutorialPage(
        title="Creating entanglement with a simple circuit",
        text=(
            "A standard way to create entanglement:\n\n"
            "  Start with |00>\n"
            "  Apply H (Hadamard) to qubit A:\n"
            "      |00> -> (|00> + |10>)/√2\n"
            "  Apply CNOT (A controls B):\n"
            "      (|00> + |10>)/√2 -> (|00> + |11>)/√2\n\n"
            "That final state is |Φ+>, a maximally entangled Bell state.\n"
        ),
    ),
    TutorialPage(
        title="Correlation: ⟨Z ⊗ Z⟩",
        text=(
            "A quick numeric way to see correlation is ⟨Z ⊗ Z⟩.\n\n"
            "Z is the Pauli-Z operator:\n"
            "    Z|0> = +|0>\n"
            "    Z|1> = -|1>\n\n"
            "For |Φ+>, the qubits match in the Z basis, giving ⟨Z⊗Z⟩ ≈ +1.\n"
            "For some other entangled states, it could be -1.\n"
            "For product states, it can be anywhere depending on the state.\n"
        ),
    ),
]


# =============================================================================
# 5) The GUI App
# =============================================================================

class EntanglementTutorialApp(tk.Tk):
    """
    Tkinter app that shows a multi-page tutorial.
    Each page updates both the explanation text and the graphic.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("Quantum Entanglement Tutorial (with graphics)")
        self.geometry("1050x680")

        # Current page index
        self.page_idx = 0

        # Precompute example states we will show repeatedly
        self.state_product = product_state_example()
        self.state_bell = bell_phi_plus()
        self.state_bell_from_circuit = make_bell_state_via_circuit()

        # Build UI components
        self._build_layout()
        self._render_page()

    def _build_layout(self) -> None:
        """
        Build the main layout:
        - left: text area
        - right: matplotlib plot
        - bottom: navigation buttons
        """
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left: text
        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.title_label = ttk.Label(left, text="", font=("Arial", 16, "bold"))
        self.title_label.pack(anchor="w", pady=(0, 8))

        self.text_box = tk.Text(left, wrap="word", height=28)
        self.text_box.pack(fill=tk.BOTH, expand=True)
        self.text_box.configure(state=tk.DISABLED)

        # Right: plot
        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5.6, 4.5), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom navigation
        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.prev_btn = ttk.Button(nav, text="◀ Previous", command=self.prev_page)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(nav, text="Next ▶", command=self.next_page)
        self.next_btn.pack(side=tk.RIGHT)

        self.page_label = ttk.Label(nav, text="")
        self.page_label.pack(side=tk.LEFT, padx=15)

    def _set_text(self, title: str, body: str) -> None:
        """Update the page title + body text."""
        self.title_label.configure(text=title)
        self.text_box.configure(state=tk.NORMAL)
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, body)
        self.text_box.configure(state=tk.DISABLED)

    def _render_page(self) -> None:
        """Render the current page (text + plot)."""
        page = PAGES[self.page_idx]
        self._set_text(page.title, page.text)
        self.page_label.configure(text=f"Page {self.page_idx + 1} / {len(PAGES)}")

        # Enable/disable nav buttons at the ends
        self.prev_btn.configure(state=tk.NORMAL if self.page_idx > 0 else tk.DISABLED)
        self.next_btn.configure(state=tk.NORMAL if self.page_idx < len(PAGES) - 1 else tk.DISABLED)

        # Update the plot depending on the page
        self.ax.clear()
        if self.page_idx == 0:
            self._plot_intro()
        elif self.page_idx == 1:
            self._plot_probabilities_compare()
        elif self.page_idx == 2:
            self._plot_conditionals()
        elif self.page_idx == 3:
            self._plot_circuit_probs()
        elif self.page_idx == 4:
            self._plot_correlations()
        else:
            self.ax.set_title("")

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------------------------------------------------------------
    # Plot functions (graphics)
    # -------------------------------------------------------------------------

    def _plot_intro(self) -> None:
        """
        A simple “cartoon” plot: show two states and label them.
        We also show the probability distribution of |Φ+> quickly.
        """
        probs = probabilities_two_qubit(self.state_bell)
        labels = list(probs.keys())
        values = [probs[k] for k in labels]

        self.ax.set_title("Bell state |Φ+> measurement probabilities")
        self.ax.bar(labels, values)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_ylabel("Probability")

        # Add text annotation on the plot
        self.ax.text(0.02, 0.92,
                     "Entangled Bell state:\n|Φ+> = (|00> + |11>)/√2",
                     transform=self.ax.transAxes)

    def _plot_probabilities_compare(self) -> None:
        """
        Compare measurement probabilities of:
        - a product state |+> ⊗ |0>
        - the Bell state |Φ+>

        We do a grouped bar chart.
        """
        p_prod = probabilities_two_qubit(self.state_product)
        p_bell = probabilities_two_qubit(self.state_bell)

        labels = ["00", "01", "10", "11"]
        prod_vals = [p_prod[l] for l in labels]
        bell_vals = [p_bell[l] for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        self.ax.set_title("Product vs Entangled: measurement probabilities")
        self.ax.bar(x - width/2, prod_vals, width, label="Product |+>⊗|0>")
        self.ax.bar(x + width/2, bell_vals, width, label="Bell |Φ+>")

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels)
        self.ax.set_ylabel("Probability")
        self.ax.set_ylim(0, 1.0)
        self.ax.legend(loc="upper right")

    def _plot_conditionals(self) -> None:
        """
        Show conditional distributions for B given measurement of A.

        We'll plot two small “bars in one chart” by stacking categories:
          A=0: P(B=0|A=0), P(B=1|A=0)
          A=1: P(B=0|A=1), P(B=1|A=1)

        And show product vs Bell side-by-side in a compact way.
        """
        c_prod = measure_first_qubit_conditional_probs(self.state_product)
        c_bell = measure_first_qubit_conditional_probs(self.state_bell)

        # We'll label these four conditional cases:
        cases = ["A=0,B=0", "A=0,B=1", "A=1,B=0", "A=1,B=1"]

        prod_vals = [
            c_prod["A=0"]["B=0"], c_prod["A=0"]["B=1"],
            c_prod["A=1"]["B=0"], c_prod["A=1"]["B=1"]
        ]
        bell_vals = [
            c_bell["A=0"]["B=0"], c_bell["A=0"]["B=1"],
            c_bell["A=1"]["B=0"], c_bell["A=1"]["B=1"]
        ]

        x = np.arange(len(cases))
        width = 0.35

        self.ax.set_title("Conditional probabilities: what happens to B after measuring A?")
        self.ax.bar(x - width/2, prod_vals, width, label="Product |+>⊗|0>")
        self.ax.bar(x + width/2, bell_vals, width, label="Bell |Φ+>")

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(cases, rotation=25, ha="right")
        self.ax.set_ylabel("Conditional probability")
        self.ax.set_ylim(0, 1.0)
        self.ax.legend(loc="upper right")

        # Add a helpful callout
        self.ax.text(
            0.02, 0.88,
            "For |Φ+>:\nIf A=0 => B=0 with prob 1\nIf A=1 => B=1 with prob 1",
            transform=self.ax.transAxes
        )

    def _plot_circuit_probs(self) -> None:
        """
        Show that the Bell state from the circuit matches the analytic Bell state.

        We'll plot:
        - probabilities of the state produced by H on A then CNOT
        - and annotate it as |Φ+>.
        """
        probs = probabilities_two_qubit(self.state_bell_from_circuit)
        labels = ["00", "01", "10", "11"]
        values = [probs[l] for l in labels]

        self.ax.set_title("Probabilities after circuit: H on A, then CNOT(A→B)")
        self.ax.bar(labels, values)
        self.ax.set_ylabel("Probability")
        self.ax.set_ylim(0, 1.0)

        # Show a tiny “circuit description” text
        self.ax.text(
            0.02, 0.90,
            "Circuit:\n|00> —H—●—\n         | \n|00> ——X—\n\nResult: |Φ+>",
            transform=self.ax.transAxes,
            family="monospace"
        )

    def _plot_correlations(self) -> None:
        """
        Plot the correlation <Z⊗Z> for product vs entangled state.
        """
        corr_prod = correlation_ZZ(self.state_product)
        corr_bell = correlation_ZZ(self.state_bell)

        labels = ["Product |+>⊗|0>", "Bell |Φ+>"]
        values = [corr_prod, corr_bell]

        self.ax.set_title("Correlation measure: ⟨Z ⊗ Z⟩")
        self.ax.bar(labels, values)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_ylabel("Expectation value (correlation)")

        # Add horizontal reference lines at -1, 0, +1
        self.ax.axhline(0.0, linewidth=1)
        self.ax.axhline(1.0, linewidth=0.8, linestyle="--")
        self.ax.axhline(-1.0, linewidth=0.8, linestyle="--")

        self.ax.text(
            0.02, 0.86,
            f"Product ⟨Z⊗Z⟩ = {corr_prod:.3f}\nBell ⟨Z⊗Z⟩ = {corr_bell:.3f}",
            transform=self.ax.transAxes
        )

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    def next_page(self) -> None:
        if self.page_idx < len(PAGES) - 1:
            self.page_idx += 1
            self._render_page()

    def prev_page(self) -> None:
        if self.page_idx > 0:
            self.page_idx -= 1
            self._render_page()


# =============================================================================
# 6) Main entry point
# =============================================================================

def main() -> None:
    app = EntanglementTutorialApp()
    app.mainloop()


if __name__ == "__main__":
    main()

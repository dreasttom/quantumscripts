#!/usr/bin/env python3
"""
hhl_student_demo.py
===================

A heavily-commented, student-friendly demo of the HHL algorithm
(Harrow–Hassidim–Lloyd) for solving a linear system:

    A x = b

⚠️ Very important teaching notes (tell your students up front):
- HHL does NOT output the full vector x as classical numbers.
  It outputs a QUANTUM STATE |x> proportional to the solution vector x.
- Reading out all entries of x would generally take exponentially many measurements.
  HHL is useful when you only need certain expectation values / overlaps.

This script focuses on:
- a small 2x2 example (so students can understand and verify)
- printing intermediate “classical” diagnostics (eigenvalues, condition number)
- showing intermediate “quantum” information when possible (circuits, success probability)
- comparing the final quantum state to the classical solution direction

Dependencies:
- numpy (required)
- Qiskit (recommended) + qiskit-aer (recommended)
  If Qiskit isn't installed, the script still runs the classical solve and explains next steps.

Install:
    pip install numpy
    pip install qiskit qiskit-aer

Run:
    python hhl_student_demo.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# 1) Helper math utilities
# =============================================================================

def is_hermitian(A: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if A is Hermitian: A == A^†."""
    return np.allclose(A, A.conj().T, atol=tol)


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Check positive definiteness (PD) using eigenvalues for small matrices.
    PD => all eigenvalues > 0.
    """
    eigvals = np.linalg.eigvalsh(A)
    return np.all(eigvals > 0)


def normalize_vector(v: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return v normalized to unit length + its original norm."""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Vector has zero norm.")
    return v / norm, float(norm)


def condition_number_from_eigs(eigs: np.ndarray) -> float:
    """Condition number κ = |λ_max| / |λ_min| for PD matrices."""
    lam_min = float(np.min(np.abs(eigs)))
    lam_max = float(np.max(np.abs(eigs)))
    if lam_min == 0:
        return float("inf")
    return lam_max / lam_min


# =============================================================================
# 2) Input prompting (student-friendly)
# =============================================================================

def prompt_float(prompt: str, default: float) -> float:
    """
    Prompt a float with a default.
    Example: press Enter to accept the default.
    """
    s = input(f"{prompt} [default {default}]: ").strip()
    if s == "":
        return float(default)
    return float(s)


def prompt_example_system() -> Tuple[np.ndarray, np.ndarray]:
    """
    Prompt for a 2x2 real symmetric (Hermitian) matrix A and a 2-vector b.

    We use a *default* matrix that is symmetric and positive definite:
        A = [[1.0, 0.5],
             [0.5, 1.0]]
        b = [1.0, 0.0]

    For teaching: start with the defaults, then try small variations.
    """
    print("\nEnter a 2x2 (real) symmetric matrix A and vector b for A x = b.")
    print("Tip: Use defaults first. HHL requires A to be (effectively) Hermitian, and typically PD.\n")

    a11 = prompt_float("A[0,0]", 1.0)
    a12 = prompt_float("A[0,1] (= A[1,0])", 0.5)
    a22 = prompt_float("A[1,1]", 1.0)

    b1 = prompt_float("b[0]", 1.0)
    b2 = prompt_float("b[1]", 0.0)

    A = np.array([[a11, a12],
                  [a12, a22]], dtype=float)

    b = np.array([b1, b2], dtype=float)

    return A, b


# =============================================================================
# 3) Qiskit availability + thin wrapper for multiple API versions
# =============================================================================

@dataclass
class QiskitHandles:
    """A small container for whichever Qiskit classes we successfully imported."""
    Aer: object
    transpile: object
    QuantumCircuit: object
    Statevector: object
    HHL: object
    LinearSolverResult: Optional[object] = None


def try_import_qiskit() -> Optional[QiskitHandles]:
    """
    Qiskit has had API shifts across versions. We try a few common imports.

    If imports fail, return None and the script will run in "classical-only" mode.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector
    except Exception:
        return None

    # Aer simulator is often in qiskit_aer
    try:
        from qiskit_aer import Aer
    except Exception:
        Aer = None  # type: ignore

    # HHL class has moved around between Qiskit versions.
    HHL = None
    # Newer split packages sometimes use qiskit_algorithms
    try:
        from qiskit_algorithms.linear_solvers.hhl import HHL as HHL_cls  # type: ignore
        HHL = HHL_cls
    except Exception:
        pass

    # Older monolithic qiskit.algorithms
    if HHL is None:
        try:
            from qiskit.algorithms.linear_solvers import HHL as HHL_cls  # type: ignore
            HHL = HHL_cls
        except Exception:
            pass

    # If we couldn't find HHL, treat as unavailable
    if HHL is None or Aer is None:
        return None

    return QiskitHandles(
        Aer=Aer,
        transpile=transpile,
        QuantumCircuit=QuantumCircuit,
        Statevector=Statevector,
        HHL=HHL,
    )


# =============================================================================
# 4) Classical solution + comparison helpers
# =============================================================================

def classical_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b classically."""
    return np.linalg.solve(A, b)


def vector_to_state(v: np.ndarray) -> np.ndarray:
    """
    Convert a real/complex vector to a normalized quantum state vector.
    HHL output is a quantum state |x> proportional to x.
    """
    v = v.astype(complex).reshape((-1, 1))
    v_normed, _ = normalize_vector(v)
    return v_normed


def fidelity_between_states(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Fidelity for pure states: |<psi|phi>|^2
    """
    psi = psi.reshape((-1, 1))
    phi = phi.reshape((-1, 1))
    inner = (psi.conj().T @ phi).item()
    return float(np.abs(inner) ** 2)


# =============================================================================
# 5) HHL run (high-level, educational)
# =============================================================================

def run_hhl_demo(A: np.ndarray, b: np.ndarray, handles: QiskitHandles) -> None:
    """
    Run HHL on a 2x2 system using Qiskit, and print intermediate steps.

    Because Qiskit versions differ, we keep things conservative:
    - Use HHL().solve(A, b) if available
    - Otherwise try HHL().construct_circuit(A, b) + simulate

    We'll also:
    - Print the quantum circuit (so students see the structure)
    - Simulate the final statevector of the solution register when possible
    - Compare the quantum |x> direction to the classical normalized x
    """
    print("\n==================== QUANTUM (HHL) SECTION ====================")

    # Normalize b for state preparation (HHL expects a quantum state |b>)
    b_normed, b_norm = normalize_vector(b.astype(complex))
    print(f"Step Q0: Normalize b for state preparation")
    print(f"  b = {b}")
    print(f"  ||b|| = {b_norm:.6f}")
    print(f"  |b> = b / ||b|| = {b_normed}")

    # Create HHL solver instance
    hhl = handles.HHL()

    # Try the common high-level API first: result = hhl.solve(A, b)
    result = None
    solve_ok = False

    # Many Qiskit versions implement solve(matrix, vector)
    try:
        result = hhl.solve(A, b)  # type: ignore
        solve_ok = True
    except Exception:
        solve_ok = False

    # If solve() is available, print what we can:
    if solve_ok and result is not None:
        # Different versions name fields differently.
        # We'll attempt to extract:
        # - solution statevector (quantum state)
        # - circuit
        # - classical post-processed solution if provided
        print("\nStep Q1: HHL solver ran via hhl.solve(A, b).")
        print("  (Note: HHL outputs a quantum state |x>, not the full classical x vector.)")

        # Circuit (if exposed)
        circ = getattr(result, "circuit", None) or getattr(hhl, "circuit", None)
        if circ is not None:
            print("\n--- HHL circuit (text) ---")
            print(circ)

        # Some versions expose a "state" or "solution" quantum object
        sol_state = None
        for attr in ["state", "solution", "statevector", "eigenstate"]:
            if hasattr(result, attr):
                sol_state = getattr(result, attr)
                break

        # If it's already a Statevector
        sv = None
        try:
            if sol_state is not None:
                # Qiskit Statevector can often be constructed from raw data
                if isinstance(sol_state, handles.Statevector):
                    sv = sol_state
                else:
                    # Some versions store numpy arrays or QuantumCircuit objects
                    # We'll attempt to convert if possible:
                    if isinstance(sol_state, np.ndarray):
                        sv = handles.Statevector(sol_state)
        except Exception:
            sv = None

        # If we can get a statevector, compare with classical direction
        if sv is not None:
            # For 2x2 system, the solution register ideally is 1 qubit (2 amplitudes).
            # But HHL implementations can include ancillas; extraction can vary.
            #
            # In many Qiskit HHL results, the returned state already corresponds
            # to the solution register (post-selected).
            probs = sv.probabilities_dict()
            print("\nStep Q2: Extracted solution state probabilities (ideal):")
            for k, v in sorted(probs.items()):
                print(f"  |{k}> : {v:.6f}")

            # Compare against classical normalized solution (direction comparison)
            x_classical = classical_solve(A, b.astype(float))
            x_state = vector_to_state(x_classical)
            print("\nStep Q3: Classical solution for comparison")
            print(f"  x (classical) = {x_classical}")
            print(f"  |x_classical> (normalized) = {x_state.ravel()}")

            # Map Qiskit bitstring ordering carefully:
            # For a single qubit solution, probabilities_dict uses '0'/'1' keys.
            # We'll build a 2-vector state from amplitudes if possible.
            try:
                amps = np.array([sv.data[0], sv.data[1]], dtype=complex).reshape((-1, 1))
                amps = normalize(amps)
                fid = fidelity_between_states(amps, x_state)
                print("\nStep Q4: Fidelity between quantum |x> and classical normalized solution direction")
                print(f"  Fidelity = {fid:.6f}  (1.0 means perfect match up to global phase)")
            except Exception:
                print("\nStep Q4: Could not compute fidelity (state format not as expected).")

        else:
            print("\nCould not extract a clean solution statevector from this Qiskit version's result.")
            print("You can still use the printed circuit and the classical comparison.")

        print("===============================================================")
        return

    # -------------------------------------------------------------------------
    # Fallback path: attempt to build a circuit and simulate it.
    # Different Qiskit versions may have construct_circuit or similar.
    # -------------------------------------------------------------------------
    print("\nStep Q1: hhl.solve(A, b) not available in this Qiskit install.")
    print("Trying to construct an HHL circuit and simulate it...")

    circ = None
    try:
        # Some versions: hhl.construct_circuit(matrix, vector)
        circ = hhl.construct_circuit(A, b)  # type: ignore
    except Exception:
        pass

    if circ is None:
        print("\n❌ Could not build an HHL circuit with this Qiskit version.")
        print("If you want this to run reliably, install a Qiskit version that includes HHL:")
        print("  pip install 'qiskit>=0.45' qiskit-aer  (example)")
        print("Or adapt the script to your specific Qiskit version's HHL API.")
        return

    print("\n--- Constructed HHL circuit (text) ---")
    print(circ)

    # Simulate the circuit statevector (ideal, no noise)
    try:
        sv = handles.Statevector.from_instruction(circ.remove_final_measurements(inplace=False))
        print("\nStep Q2: Simulated full circuit statevector (includes ancillas).")
        print("  Note: Extracting ONLY the solution register depends on how the circuit is laid out.")
        print("  We'll show the top measurement probabilities over all qubits as a teaching aid.\n")

        probs = sv.probabilities_dict()
        top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for bitstr, p in top:
            print(f"  |{bitstr}> : {p:.6f}")

        print("\n(For a deeper lab exercise: identify which qubits correspond to the solution register,")
        print(" and trace out ancillas / post-select as required by the specific HHL implementation.)")

    except Exception as e:
        print("\n❌ Statevector simulation failed:", e)

    print("===============================================================")


# =============================================================================
# 6) Main program
# =============================================================================

def main() -> None:
    print("HHL Algorithm Demo (Student Version)")
    print("-----------------------------------")
    print("We will solve A x = b for a 2x2 system.\n")

    A, b = prompt_example_system()

    print("\n==================== INPUT CHECKS ====================")
    print("A =\n", A)
    print("b =", b)

    # HHL typically assumes A is Hermitian (or can be embedded into a Hermitian form).
    herm = is_hermitian(A.astype(complex))
    print(f"\nCheck: Is A Hermitian? {herm}")

    # Many textbook HHL demos assume positive definite (helps interpretation).
    pd = is_positive_definite(A.astype(float))
    print(f"Check: Is A positive definite? {pd}")

    eigvals = np.linalg.eigvalsh(A.astype(float))
    kappa = condition_number_from_eigs(eigvals)

    print("\nStep C1: Eigenvalues and condition number (classical diagnostics)")
    print(f"  eigenvalues(A) = {eigvals}")
    print(f"  condition number κ ≈ {kappa:.6f}")
    print("(Large κ generally makes the problem harder for both classical numerics and HHL.)")

    # Classical solution for verification
    print("\n==================== CLASSICAL SOLUTION ====================")
    try:
        x = classical_solve(A.astype(float), b.astype(float))
        print("Step C2: Solve A x = b classically")
        print("  x =", x)

        x_state = vector_to_state(x)
        print("\nStep C3: What HHL aims to output is a quantum state |x> proportional to x:")
        print("  |x_classical> (normalized) =", x_state.ravel())
    except Exception as e:
        print("❌ Classical solve failed:", e)
        return

    # If A isn't Hermitian/PD, HHL may not apply directly; warn students.
    if not herm:
        print("\n⚠️ Warning: This A is not Hermitian. Basic HHL requires Hermitian (or an embedding trick).")
        print("For classroom use, stick to symmetric (real) or Hermitian matrices.")
    if not pd:
        print("\n⚠️ Warning: This A is not positive definite. Some HHL demos assume PD.")
        print("It might still be possible, but interpretation can get more complicated.")

    # Try quantum part (if Qiskit present)
    handles = try_import_qiskit()
    if handles is None:
        print("\n==================== QUANTUM (HHL) SECTION ====================")
        print("Qiskit HHL + Aer not found on this system.")
        print("To run the quantum circuit portion, install:")
        print("  pip install qiskit qiskit-aer")
        print("\nWhat would happen next (conceptually):")
        print("  1) Prepare |b> on a 'solution' register")
        print("  2) Run Quantum Phase Estimation on e^{iAt} to estimate eigenvalues")
        print("  3) Do a controlled rotation to apply 1/λ scaling (probabilistic step)")
        print("  4) Uncompute phase estimation")
        print("  5) Post-select an ancilla measurement to obtain |x>")
        print("===============================================================")
        return

    # Run HHL using Qiskit
    run_hhl_demo(A.astype(float), b.astype(float), handles)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)

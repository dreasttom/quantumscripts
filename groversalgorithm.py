#!/usr/bin/env python3
"""
grover_demo_interactive.py

Student Friendly Grover's Algorithm demo.

What this script does:
- Prompts for:
    1) number of qubits n (search space size = 2^n)
    2) the "marked" item (target) as either a bitstring (like 1011) or an integer
    3) number of Grover iterations (or press Enter to auto-choose)
    4) number of measurement shots (how many times to run and measure)
- Builds Grover's algorithm:
    - Initialize uniform superposition with Hadamards
    - Apply the ORACLE (phase flip on the marked state)
    - Apply the DIFFUSER (inversion about the mean)
    - Repeat for k iterations
- Shows "steps" by printing:
    - key choices (n, target, iterations)
    - ideal probabilities after each iteration (statevector simulation) if Qiskit is available
    - final measurement results (histogram counts) from a simulator

Two modes:
1) If Qiskit is installed, uses Qiskit Aer simulator:
   - statevector simulation per iteration (to show amplitude amplification)
   - measurement shots at the end (to show sampling)
2) If Qiskit is NOT installed, uses a pure-Python "toy" amplitude simulation:
   - still demonstrates Grover’s amplification steps for small n (recommended n <= 12)
   - prints probabilities per iteration
   - simulates measurement by sampling

Install Qiskit (recommended):
    pip install qiskit qiskit-aer

Run:
    python grover_demo_interactive.py
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Helper functions: parsing and formatting
# =============================================================================

def is_bitstring(s: str) -> bool:
    """Return True if s is a non-empty string containing only 0/1."""
    return len(s) > 0 and all(ch in "01" for ch in s)


def int_to_bitstring(x: int, n: int) -> str:
    """Convert integer x to an n-bit binary string."""
    return format(x, f"0{n}b")


def bitstring_to_int(b: str) -> int:
    """Convert bitstring b to integer."""
    return int(b, 2)


def pretty_top_probs(probs: Dict[str, float], top_k: int = 8) -> str:
    """
    Pretty-print the top_k most likely outcomes from a probability dictionary.
    probs maps bitstring -> probability.
    """
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    lines = []
    for b, p in items:
        lines.append(f"  {b} : {p:.6f}")
    return "\n".join(lines)


def choose_grover_iterations(n: int) -> int:
    """
    Standard rule-of-thumb for 1 marked item:
        k ≈ floor( (pi/4) * sqrt(N) )
    where N = 2^n.
    """
    N = 2 ** n
    return max(1, int(math.floor((math.pi / 4.0) * math.sqrt(N))))


# =============================================================================
# PURE PYTHON TOY SIMULATION (works without Qiskit)
# =============================================================================

def toy_grover_simulation(n: int, target: int, iterations: int, verbose: bool = True) -> Dict[str, float]:
    """
    Pure Python amplitude simulation of Grover's algorithm (single marked item).

    Representation:
      - A state vector of length N = 2^n, complex amplitudes (here they stay real).
      - Start with uniform amplitudes 1/sqrt(N)
      - Oracle: flip sign of amplitude at index 'target'
      - Diffuser: inversion about the mean:
            a_i <- 2*avg(a) - a_i

    Returns:
      - Probability distribution as dict bitstring -> probability
    """
    N = 2 ** n
    amp = [1.0 / math.sqrt(N)] * N  # start uniform

    def oracle(a: List[float]) -> None:
        # Phase flip only on the marked state
        a[target] *= -1.0

    def diffuser(a: List[float]) -> None:
        # Inversion about the mean
        avg = sum(a) / N
        for i in range(N):
            a[i] = 2.0 * avg - a[i]

    if verbose:
        print("\n[TOY MODE] Qiskit not found. Using pure-Python amplitude simulation.")
        print("This is great for learning, but only practical for small n (try n <= 12).")
        print(f"Search space size N = 2^{n} = {N}")
        print(f"Marked target index = {target} (bitstring {int_to_bitstring(target, n)})")
        print(f"Iterations = {iterations}\n")

    # Show initial probability of target
    if verbose:
        p0 = amp[target] ** 2
        print(f"Step 0 (initial uniform superposition):")
        print(f"  P(target) = {p0:.6f}")

    # Apply Grover iterations
    for k in range(1, iterations + 1):
        oracle(amp)
        diffuser(amp)
        if verbose:
            pk = amp[target] ** 2
            print(f"Step {k} (after oracle + diffuser):")
            print(f"  P(target) = {pk:.6f}")

    # Build probability distribution
    probs: Dict[str, float] = {}
    for i, a in enumerate(amp):
        probs[int_to_bitstring(i, n)] = a * a  # amplitudes remain real in this toy model

    return probs


def toy_sample_measurements(probs: Dict[str, float], shots: int) -> Dict[str, int]:
    """
    Sample measurement outcomes from a probability distribution.
    """
    outcomes = list(probs.keys())
    weights = [probs[o] for o in outcomes]

    counts: Dict[str, int] = {o: 0 for o in outcomes}
    for _ in range(shots):
        pick = random.choices(outcomes, weights=weights, k=1)[0]
        counts[pick] += 1

    # Keep only nonzero entries (cleaner printing)
    counts = {k: v for k, v in counts.items() if v > 0}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# =============================================================================
# QISKIT IMPLEMENTATION (if installed)
# =============================================================================

def qiskit_available() -> bool:
    """Check whether Qiskit + Aer are importable."""
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        return True
    except Exception:
        return False


def build_oracle_phase_flip(n: int, target_bitstring: str):
    """
    Build a Grover oracle that applies a phase flip (-1) to the marked state |target>.

    How we do it (high-level idea):
    - We want: |x> -> -|x> if x == target, else |x>
    - Use X gates to map |target> to |11...1>
      (i.e., flip qubits where target has 0)
    - Apply a multi-controlled Z phase flip on |11...1>
      (implemented using an H + multi-controlled X + H on the last qubit)
    - Undo the X gates

    Returns a Qiskit QuantumCircuit containing the oracle as a gate.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name="Oracle")

    # 1) Map the target state to |11..1> using X on qubits where target bit is 0
    #    Example: target = 1010 means apply X to qubits (bit=0 positions).
    for i, bit in enumerate(reversed(target_bitstring)):
        # Note: Qiskit qubit 0 is the least-significant bit in the usual convention.
        if bit == "0":
            qc.x(i)

    # 2) Apply a phase flip to |11..1>:
    #    Multi-controlled Z can be done with:
    #      H on last qubit
    #      multi-controlled X onto last qubit
    #      H on last qubit
    qc.h(n - 1)
    if n == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n - 1)), n - 1)  # controls are 0..n-2, target is n-1
    qc.h(n - 1)

    # 3) Undo the X gates
    for i, bit in enumerate(reversed(target_bitstring)):
        if bit == "0":
            qc.x(i)

    return qc


def build_diffuser(n: int):
    """
    Build the Grover diffuser (inversion about the mean).

    Standard diffuser circuit:
      - Apply H to all qubits
      - Apply X to all qubits
      - Apply phase flip to |00..0> (equivalently to |11..1> after X)
      - Apply X to all qubits
      - Apply H to all qubits

    Returns a Qiskit QuantumCircuit named "Diffuser".
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name="Diffuser")

    qc.h(range(n))
    qc.x(range(n))

    # Phase flip on |00..0>:
    # After X gates, |00..0> becomes |11..1>, so we can do the same
    # multi-controlled Z trick as in the oracle.
    qc.h(n - 1)
    if n == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))

    return qc


def qiskit_grover_demo(n: int, target: int, iterations: int, shots: int, verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Run Grover's algorithm with Qiskit:
    - Show ideal probabilities after each iteration using statevector simulation
    - Then run a shots-based measurement to produce counts

    Returns:
      (ideal_probs, measurement_counts)
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_aer import Aer
    from qiskit import transpile

    target_bits = int_to_bitstring(target, n)

    if verbose:
        print("\n[QISKIT MODE] Using Qiskit + Aer simulator.")
        print(f"Search space size N = 2^{n} = {2**n}")
        print(f"Marked target index = {target} (bitstring {target_bits})")
        print(f"Iterations = {iterations}")
        print(f"Shots (measurements) = {shots}\n")

    # Build oracle and diffuser subcircuits
    oracle = build_oracle_phase_flip(n, target_bits)
    diffuser = build_diffuser(n)

    # Build the full Grover circuit for measurement (all iterations, then measure)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))  # uniform superposition

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n), range(n))

    # -------------------------------------------------------------------------
    # STEP-BY-STEP: show ideal probabilities after each iteration
    # We'll build a circuit incrementally and compute its Statevector.
    # -------------------------------------------------------------------------
    if verbose:
        print("Step-by-step ideal probabilities (statevector, no noise):")

        step_circ = QuantumCircuit(n)
        step_circ.h(range(n))

        # Step 0: after initial Hadamards
        sv = Statevector.from_instruction(step_circ)
        probs0 = sv.probabilities_dict()
        # Qiskit returns bitstrings with qubit-0 on the RIGHT (little endian).
        # We'll reformat to our standard n-bit strings (still okay for comparisons).
        # We’ll convert keys to fixed-width strings.
        probs0_fmt = {k.zfill(n): float(v) for k, v in probs0.items()}
        print("Step 0 (after H...H):")
        print(f"  P(target={target_bits}) = {probs0_fmt.get(target_bits, 0.0):.6f}")
        print(pretty_top_probs(probs0_fmt, top_k=6))

        # Each Grover iteration
        for i in range(1, iterations + 1):
            step_circ.compose(oracle, inplace=True)
            step_circ.compose(diffuser, inplace=True)
            sv = Statevector.from_instruction(step_circ)
            probsi = sv.probabilities_dict()
            probsi_fmt = {k.zfill(n): float(v) for k, v in probsi.items()}

            print(f"\nStep {i} (after oracle + diffuser):")
            print(f"  P(target={target_bits}) = {probsi_fmt.get(target_bits, 0.0):.6f}")
            print(pretty_top_probs(probsi_fmt, top_k=6))

    # Final ideal probabilities from the full (unmeasured) circuit:
    # We'll compute it by rebuilding the same logic without measurements.
    ideal_circ = QuantumCircuit(n)
    ideal_circ.h(range(n))
    for _ in range(iterations):
        ideal_circ.compose(oracle, inplace=True)
        ideal_circ.compose(diffuser, inplace=True)

    sv_final = Statevector.from_instruction(ideal_circ)
    ideal_probs_raw = sv_final.probabilities_dict()
    ideal_probs = {k.zfill(n): float(v) for k, v in ideal_probs_raw.items()}

    # -------------------------------------------------------------------------
    # Measurement sampling (shots)
    # -------------------------------------------------------------------------
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=shots).result()
    counts_raw = result.get_counts()

    # Qiskit counts keys are bitstrings; keep fixed width
    counts = {k.zfill(n): int(v) for k, v in counts_raw.items()}
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    return ideal_probs, counts


# =============================================================================
# Interactive main program
# =============================================================================

def prompt_int(prompt: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """Prompt repeatedly until the user enters a valid integer within optional bounds."""
    while True:
        s = input(prompt).strip()
        try:
            x = int(s)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if min_value is not None and x < min_value:
            print(f"Please enter an integer >= {min_value}.")
            continue
        if max_value is not None and x > max_value:
            print(f"Please enter an integer <= {max_value}.")
            continue

        return x


def main() -> None:
    print("Grover's Algorithm (Student Demo)")
    print("--------------------------------")
    print("Grover searches an unsorted list of size N=2^n for ONE marked item.")
    print("This demo assumes exactly one marked (target) state.\n")

    # 1) Choose number of qubits
    n = prompt_int("Enter number of qubits n (try 2..8 for easy viewing): ", min_value=1)
    N = 2 ** n
    print(f"Search space size N = 2^n = {N}\n")

    # 2) Choose target item
    raw_target = input(f"Enter target as bitstring of length {n} (e.g., 0101) OR an integer 0..{N-1}: ").strip()

    if is_bitstring(raw_target):
        if len(raw_target) != n:
            print(f"Bitstring length must be exactly {n}.")
            return
        target_bits = raw_target
        target = bitstring_to_int(target_bits)
    else:
        try:
            target = int(raw_target)
        except ValueError:
            print("Invalid target. Enter a bitstring or an integer.")
            return
        if not (0 <= target < N):
            print(f"Target integer must be between 0 and {N-1}.")
            return
        target_bits = int_to_bitstring(target, n)

    print(f"\nMarked target is: {target} (bitstring {target_bits})\n")

    # 3) Choose number of iterations
    default_iters = choose_grover_iterations(n)
    raw_iters = input(f"Enter number of Grover iterations (press Enter for default ~ pi/4*sqrt(N) = {default_iters}): ").strip()
    if raw_iters == "":
        iterations = default_iters
    else:
        try:
            iterations = int(raw_iters)
        except ValueError:
            print("Invalid iteration count.")
            return
        if iterations < 1:
            print("Iterations must be >= 1.")
            return

    # 4) Choose number of shots
    shots = prompt_int("Enter number of shots for measurement sampling (e.g., 256, 1024): ", min_value=1)

    # 5) Run demo
    if qiskit_available():
        ideal_probs, counts = qiskit_grover_demo(n=n, target=target, iterations=iterations, shots=shots, verbose=True)
    else:
        ideal_probs = toy_grover_simulation(n=n, target=target, iterations=iterations, verbose=True)
        counts = toy_sample_measurements(ideal_probs, shots=shots)

    # 6) Print final results
    print("\n==================== FINAL OUTPUT ====================")
    print(f"Target = {target_bits} (decimal {target})")
    print(f"Iterations used = {iterations}")
    print(f"Shots = {shots}\n")

    print("Ideal final probabilities (top outcomes):")
    print(pretty_top_probs(ideal_probs, top_k=10))

    print("\nSampled measurement counts (top outcomes):")
    top_counts = list(counts.items())[:10]
    for b, c in top_counts:
        print(f"  {b} : {c}")

    # Most frequent measured result
    best_bitstring = max(counts, key=counts.get)
    print("\nMost frequent measured outcome:")
    print(f"  {best_bitstring}  (decimal {bitstring_to_int(best_bitstring)})")

    print("\nDid we find the target?")
    print(f"  Target bitstring: {target_bits}")
    print(f"  Measured best   : {best_bitstring}")
    print("  " + ("YES ✅" if best_bitstring == target_bits else "Not this time (try adjusting iterations/shots)"))

    print("======================================================\n")

    # Teaching note: too many iterations can "overshoot" and reduce success probability.
    print("Teaching note:")
    print("- Grover success probability rises, then can fall if you do too many iterations (overshoot).")
    print(f"- The default iteration choice (~{default_iters}) is usually near-optimal for one marked item.\n")


if __name__ == "__main__":
    main()

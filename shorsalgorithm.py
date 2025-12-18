#!/usr/bin/env python3
"""
shor_demo_interactive.py

Student-friendly Shor's algorithm demo:
- Prompts the user for N (the integer to factor)
- Tries Shor's algorithm structure (with quantum period finding via Qiskit if available)
- Falls back to a classical period finder if Qiskit is not installed
- Prints intermediate values so students can trace what is happening

This is intended for SMALL N (e.g., 15, 21, 33, 35).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Optional, Tuple


# =============================================================================
# 1) Helper functions
# =============================================================================

def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    return math.gcd(a, b)


def is_perfect_power(n: int) -> bool:
    """
    Return True if n can be written as x^k for integers x>1, k>1.
    (Simple check good for small classroom examples.)
    """
    if n < 4:
        return False
    for k in range(2, int(math.log2(n)) + 1):
        x = round(n ** (1.0 / k))
        if x > 1 and x ** k == n:
            return True
    return False


def continued_fraction_denominator(measured: int, q: int, max_den: int) -> int:
    """
    Convert measured/q ≈ s/r into a candidate r using continued fractions.
    """
    frac = Fraction(measured, q).limit_denominator(max_den)
    return frac.denominator


# =============================================================================
# 2) Classical (toy) period finding (fallback)
# =============================================================================

def classical_find_period(a: int, N: int) -> Optional[int]:
    """
    Find smallest r>0 such that a^r mod N = 1 (order of a mod N).
    This is slow for large N but fine for small demos.
    """
    if gcd(a, N) != 1:
        return None

    value = 1
    for r in range(1, 2 * N + 1):
        value = (value * a) % N
        if value == 1:
            return r

    return None


# =============================================================================
# 3) Quantum period finding via Qiskit (small N only)
# =============================================================================

def qiskit_find_period(a: int, N: int, shots: int = 2048, verbose: bool = False) -> Optional[int]:
    """
    Attempt quantum period finding with Qiskit.
    For small N only: uses a full unitary permutation matrix (not scalable).

    Returns r if found/validated, else None.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile
        from qiskit_aer import Aer
        from qiskit.circuit.library import UnitaryGate
        import numpy as np
    except Exception:
        # Qiskit not installed
        return None

    n = N.bit_length()
    t = 2 * n
    q = 1 << t

    if verbose:
        print(f"  [QISKIT] Using n={n} work qubits, t={t} counting qubits (q=2^t={q}).")

    qc = QuantumCircuit(t + n, t)

    # Work register starts at |1>
    qc.x(t + 0)

    # Put counting register into superposition
    qc.h(range(t))

    # Build multiplication-by-a^(2^k) mod N unitaries as permutation matrices
    def mul_mod_unitary(a_pow: int) -> "np.ndarray":
        dim = 1 << n
        U = np.zeros((dim, dim), dtype=complex)

        for y in range(dim):
            if y < N:
                U[(a_pow * y) % N, y] = 1.0
            else:
                U[y, y] = 1.0

        return U

    for k in range(t):
        a_pow = pow(a, 1 << k, N)
        U = mul_mod_unitary(a_pow)
        gate = UnitaryGate(U, label=f"mul_{a_pow}_mod_{N}")
        qc.append(gate.control(1), [k] + list(range(t, t + n)))

    # Inverse QFT on counting register
    def inverse_qft(circ: QuantumCircuit, qubits: list[int]) -> None:
        m = len(qubits)
        for i in range(m // 2):
            circ.swap(qubits[i], qubits[m - i - 1])
        for j in range(m):
            for k2 in range(j):
                circ.cp(-math.pi / (1 << (j - k2)), qubits[j], qubits[k2])
            circ.h(qubits[j])

    inverse_qft(qc, list(range(t)))

    # Measure counting register
    qc.measure(range(t), range(t))

    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Most frequent bitstring
    measured_str = max(counts, key=counts.get)
    measured = int(measured_str, 2)

    if verbose:
        print(f"  [QISKIT] Most frequent measurement: {measured_str} (decimal {measured})")

    # Continued fractions to guess r
    r = continued_fraction_denominator(measured, q, max_den=N)

    if verbose:
        print(f"  [QISKIT] Continued fraction candidate r={r}")

    # Validate
    if r > 0 and pow(a, r, N) == 1:
        return r

    # Check small multiples (common heuristic)
    for mult in range(2, 6):
        rr = r * mult
        if pow(a, rr, N) == 1:
            if verbose:
                print(f"  [QISKIT] Candidate failed; using multiple {mult}*r = {rr}")
            return rr

    return None


# =============================================================================
# 4) Shor driver (prints intermediate steps)
# =============================================================================

def shor_factor_verbose(N: int, max_tries: int = 25, shots: int = 2048) -> Optional[Tuple[int, int]]:
    """
    Attempt to factor N using Shor's algorithm structure, printing intermediate values.

    Returns (p, q) on success or None on failure.
    """

    print("\n=== Shor's Algorithm Demo (Educational) ===")
    print(f"Target N = {N}")

    # Easy checks
    if N <= 1:
        print("N must be > 1.")
        return None

    if N % 2 == 0:
        print("N is even, so an easy factor is 2.")
        return (2, N // 2)

    if is_perfect_power(N):
        print("N looks like a perfect power (x^k). This script doesn't handle that case.")
        return None

    print(f"Will try up to {max_tries} random choices of a.\n")

    for attempt in range(1, max_tries + 1):
        print(f"--- Attempt {attempt} ---")

        # Choose a random a in [2, N-2]
        a = random.randrange(2, N - 1)
        print(f"Chosen a = {a}")

        # Step 1: gcd(a, N)
        g = gcd(a, N)
        print(f"gcd(a, N) = gcd({a}, {N}) = {g}")

        # If gcd > 1, we found a factor right away
        if g > 1:
            p, q = min(g, N // g), max(g, N // g)
            print("Non-trivial gcd found immediately!")
            print(f"Factors: {p} * {q} = {N}")
            return (p, q)

        # Step 2: Find period r of a^x mod N
        print("Finding period r such that a^r mod N = 1 ...")

        r = qiskit_find_period(a, N, shots=shots, verbose=True)

        if r is None:
            print("  [QISKIT] Not available or failed. Falling back to classical period finding.")
            r = classical_find_period(a, N)

        print(f"Candidate period r = {r}")

        if r is None:
            print("No period found; trying a new a.\n")
            continue

        if r % 2 == 1:
            print("r is odd. Shor needs even r, so try again.\n")
            continue

        # Step 3: Compute x = a^(r/2) mod N
        x = pow(a, r // 2, N)
        print(f"x = a^(r/2) mod N = {a}^({r//2}) mod {N} = {x}")

        # If x == -1 mod N, factors will be trivial
        if x == N - 1:
            print(f"x == N-1 (i.e., {x} == {N}-1). This gives trivial factors; try again.\n")
            continue
        if x == 1:
            print("x == 1. This usually leads to trivial factors; try again.\n")
            continue

        # Step 4: Compute gcd(x-1, N) and gcd(x+1, N)
        p = gcd(x - 1, N)
        q = gcd(x + 1, N)
        print(f"p = gcd(x - 1, N) = gcd({x}-1, {N}) = gcd({x-1}, {N}) = {p}")
        print(f"q = gcd(x + 1, N) = gcd({x}+1, {N}) = gcd({x+1}, {N}) = {q}")

        # Check if they are non-trivial
        if p not in (1, N) and q not in (1, N) and p * q == N:
            p, q = min(p, q), max(p, q)
            print("\nSUCCESS: Found non-trivial factors!")
            print(f"Final answer: {N} = {p} * {q}")
            return (p, q)

        print("Factors were trivial or incorrect; try again.\n")

    print("Gave up after max_tries attempts without finding factors.")
    return None


# =============================================================================
# 5) Interactive main()
# =============================================================================

def main() -> None:
    """
    Prompt the user for N, run the verbose Shor demo, print the final result.
    """
    print("Shor's Algorithm (Educational Demo)")
    print("Enter a SMALL composite integer N to factor (try 15, 21, 33, 35).")

    while True:
        raw = input("N = ").strip()
        try:
            N = int(raw)
            break
        except ValueError:
            print("Please enter a valid integer (e.g., 15).")

    result = shor_factor_verbose(N)

    print("\n=== DONE ===")
    if result is None:
        print(f"Could not factor N={N} in this run.")
    else:
        p, q = result
        print(f"Result: {N} = {p} * {q}")


if __name__ == "__main__":
    main()

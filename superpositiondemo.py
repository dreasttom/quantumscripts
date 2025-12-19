"""
Entanglement demo (graphical) for intro quantum computing students.

UPDATED:
  - Adds a panel that prints:
      * the full 2-qubit density matrix ρ (4x4)
      * the reduced density matrices ρ_A and ρ_B (2x2 each)

This is intentionally "framework-free" (no Qiskit), so students can see the linear algebra.

Dependencies:
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

# -----------------------------
# Basic quantum linear algebra
# -----------------------------

# Computational basis states |0> and |1>
ket0 = np.array([[1.0], [0.0]])
ket1 = np.array([[0.0], [1.0]])

# Pauli matrices (used for Bloch sphere / reduced state visualization)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(a, b):
    """Tensor product (Kronecker product) helper."""
    return np.kron(a, b)


def normalize(psi):
    """Normalize a statevector |psi>."""
    return psi / np.linalg.norm(psi)


# ----------------------------------------
# Build Bell state and a comparison "fake"
# ----------------------------------------

def bell_state_phi_plus():
    """
    |Φ+> = (|00> + |11>) / sqrt(2)

    Basis ordering in the 4D vector:
      |00>, |01>, |10>, |11>
    """
    psi = (kron(ket0, ket0) + kron(ket1, ket1))
    return normalize(psi)


def classical_correlated_mixture():
    """
    A classical mixture (NOT entangled):
      rho = 0.5 |00><00| + 0.5 |11><11|

    This produces perfect correlations in Z basis,
    but differs from an entangled Bell state in other measurement bases.
    """
    proj00 = kron(ket0, ket0) @ kron(ket0, ket0).conj().T
    proj11 = kron(ket1, ket1) @ kron(ket1, ket1).conj().T
    return 0.5 * proj00 + 0.5 * proj11


def pure_to_density(psi):
    """Convert pure statevector |psi> into density matrix rho = |psi><psi|."""
    return psi @ psi.conj().T


# -----------------------------
# Partial trace (2 qubits)
# -----------------------------

def partial_trace_two_qubits(rho, keep=0):
    """
    Partial trace of a 2-qubit density matrix rho (4x4).

    keep=0 -> return reduced density matrix for qubit A
    keep=1 -> return reduced density matrix for qubit B

    Assumes basis ordering |00>,|01>,|10>,|11>.

    Conceptually:
      rho[a,b; a',b']  (where a,b,a',b' ∈ {0,1})
    Then:
      ρ_A[a,a'] = Σ_b rho[a,b; a',b]
      ρ_B[b,b'] = Σ_a rho[a,b; a,b']
    """
    rho_reshaped = rho.reshape(2, 2, 2, 2)  # indices: a,b,a',b'

    if keep == 0:
        red = np.zeros((2, 2), dtype=complex)
        for a in range(2):
            for ap in range(2):
                s = 0.0 + 0.0j
                for b in range(2):
                    s += rho_reshaped[a, b, ap, b]
                red[a, ap] = s
        return red

    if keep == 1:
        red = np.zeros((2, 2), dtype=complex)
        for b in range(2):
            for bp in range(2):
                s = 0.0 + 0.0j
                for a in range(2):
                    s += rho_reshaped[a, b, a, bp]
                red[b, bp] = s
        return red

    raise ValueError("keep must be 0 or 1")


# -----------------------------
# Bloch vector from 1-qubit rho
# -----------------------------

def bloch_vector(rho1):
    """
    For a 1-qubit density matrix rho1, Bloch vector r:
      r_x = Tr(rho X)
      r_y = Tr(rho Y)
      r_z = Tr(rho Z)

    Pure state -> |r| = 1
    Maximally mixed I/2 -> r = (0,0,0)
    """
    rx = np.real(np.trace(rho1 @ X))
    ry = np.real(np.trace(rho1 @ Y))
    rz = np.real(np.trace(rho1 @ Z))
    return np.array([rx, ry, rz], dtype=float)


# -----------------------------------------
# Measurement in a rotated basis (1 qubit)
# -----------------------------------------

def basis_state(theta):
    """
    Measurement basis restricted to the X-Z plane (rotation about Y):

      |0_theta> = cos(theta/2)|0> + sin(theta/2)|1>
      |1_theta> = sin(theta/2)|0> - cos(theta/2)|1>

    theta=0       -> Z basis
    theta=pi/2    -> X basis (up to a global phase)
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    ket0t = c * ket0 + s * ket1
    ket1t = s * ket0 - c * ket1
    return normalize(ket0t), normalize(ket1t)


def projectors_for_theta(theta):
    """Return projectors (P0, P1) for measurement basis at angle theta."""
    k0, k1 = basis_state(theta)
    return k0 @ k0.conj().T, k1 @ k1.conj().T


# -----------------------------------------
# Joint measurement probabilities (2 qubits)
# -----------------------------------------

def joint_probs(rho, thetaA, thetaB):
    """
    Joint outcome probabilities for measuring:
      - qubit A in basis thetaA
      - qubit B in basis thetaB

    Using Born rule with density matrix:
      p(a,b) = Tr( rho * (P_a ⊗ P_b) )
    """
    PA0, PA1 = projectors_for_theta(thetaA)
    PB0, PB1 = projectors_for_theta(thetaB)
    projectorsA = [PA0, PA1]
    projectorsB = [PB0, PB1]

    probs = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            M = kron(projectorsA[a], projectorsB[b])
            probs[a, b] = max(0.0, np.real(np.trace(rho @ M)))

    # Normalize to 1 (protects against tiny numerical drift)
    probs /= probs.sum()
    return probs


def correlation_from_probs(probs):
    """
    Simple correlation score:

    Map measurement outcomes:
      0 -> +1
      1 -> -1

    Then compute:
      E = <A * B> = Σ_{a,b} val(a)val(b)p(a,b)
    """
    val = {0: +1, 1: -1}
    E = 0.0
    for a in (0, 1):
        for b in (0, 1):
            E += val[a] * val[b] * probs[a, b]
    return E


# -----------------------------------------
# Plotting helpers
# -----------------------------------------

def draw_bloch_sphere(ax, r, title):
    """Draw a simple Bloch sphere and the Bloch vector r."""
    ax.clear()

    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.5, alpha=0.4)

    # Coordinate axes
    ax.plot([-1, 1], [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], [-1, 1], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, 0], [-1, 1], linewidth=1)

    # Bloch vector arrow
    ax.quiver(0, 0, 0, r[0], r[1], r[2], length=1.0, normalize=False)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=35)


def draw_joint_bar(ax, probs, title):
    """Bar chart for joint probabilities: p00, p01, p10, p11."""
    ax.clear()
    labels = ["00", "01", "10", "11"]
    vals = [probs[0, 0], probs[0, 1], probs[1, 0], probs[1, 1]]
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")


def draw_correlation(ax, E, title):
    """Bar for correlation E in [-1, 1]."""
    ax.clear()
    ax.axhline(0, linewidth=1)
    ax.bar(["E"], [E])
    ax.set_ylim(-1, 1)
    ax.set_title(title)
    ax.set_ylabel("Correlation (E)")


# -----------------------------------------
# Density matrix formatting for display
# -----------------------------------------

def fmt_complex(z, decimals=3, tol=1e-12):
    """
    Format a complex number as a short string for printing.

    Examples:
      0.5+0.0j  -> 0.500
      0.0+0.5j  -> 0.500i
      0.5-0.5j  -> 0.500-0.500i

    tol is used to treat very small values as 0.
    """
    a = np.real(z)
    b = np.imag(z)
    if abs(a) < tol: a = 0.0
    if abs(b) < tol: b = 0.0

    if b == 0.0:
        return f"{a:.{decimals}f}"
    if a == 0.0:
        return f"{b:.{decimals}f}i"
    sign = "+" if b > 0 else "-"
    return f"{a:.{decimals}f}{sign}{abs(b):.{decimals}f}i"


def matrix_to_string(M, name="M", decimals=3):
    """
    Convert a small matrix (2x2 or 4x4) into a pretty monospace string.
    """
    lines = [f"{name} ="]
    for row in M:
        row_str = "  [" + "  ".join(fmt_complex(x, decimals=decimals) for x in row) + "]"
        lines.append(row_str)
    return "\n".join(lines)


# -----------------------------
# Main interactive demo
# -----------------------------

def main():
    # Prepare the two alternative 2-qubit states:
    #   - entangled Bell state
    #   - classical correlated mixture
    bell_rho = pure_to_density(bell_state_phi_plus())
    classical_rho = classical_correlated_mixture()

    # Create figure layout
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Graphical Entanglement Demo: Bell State vs Classical Correlation", fontsize=14)

    # Layout (2 rows x 4 columns):
    #  Row 1: Bloch A | Bloch B | Joint probs | Density panel
    #  Row 2: Circuit/explain | (empty) | Correlation | (spare)
    ax_bloch_A = fig.add_subplot(2, 4, 1, projection='3d')
    ax_bloch_B = fig.add_subplot(2, 4, 2, projection='3d')
    ax_joint   = fig.add_subplot(2, 4, 3)
    ax_density = fig.add_subplot(2, 4, 4)
    ax_corr    = fig.add_subplot(2, 4, 7)
    ax_text    = fig.add_subplot(2, 4, 5)

    ax_text.axis("off")
    ax_density.axis("off")

    circuit_text = (
        "State prep (conceptual):\n"
        "  |0> ──H──■──  => |Φ+> = (|00>+|11>)/√2\n"
        "  |0> ─────X──\n\n"
        "Key ideas:\n"
        "  • Reduced states of entangled qubits can be maximally mixed.\n"
        "  • Correlation appears in joint measurements.\n\n"
        "Controls:\n"
        "  • Angle sliders rotate measurement bases (in X–Z plane).\n"
        "  • Switch modes to compare entangled vs classical correlation.\n"
    )
    ax_text.text(0, 1, circuit_text, va="top", family="monospace", fontsize=10)

    # Slider axes for measurement angles (degrees are easier for students)
    ax_slider_A = fig.add_axes([0.10, 0.08, 0.35, 0.03])
    ax_slider_B = fig.add_axes([0.55, 0.08, 0.35, 0.03])

    sliderA = Slider(ax_slider_A, "Angle A (deg)", 0, 180, valinit=0, valstep=1)
    sliderB = Slider(ax_slider_B, "Angle B (deg)", 0, 180, valinit=0, valstep=1)

    # Mode selector (radio buttons)
    ax_radio = fig.add_axes([0.83, 0.55, 0.16, 0.20])
    radio = RadioButtons(ax_radio, ("Entangled (Bell |Φ+⟩)", "Classical correlated mixture"))

    # Reset button
    ax_reset = fig.add_axes([0.83, 0.48, 0.16, 0.05])
    btn_reset = Button(ax_reset, "Reset angles")

    def update(_=None):
        """
        Recompute the relevant quantities and redraw all plots/text.

        This is called whenever:
          - a slider changes
          - the mode changes (radio button)
          - reset button is pressed
        """
        thetaA = np.deg2rad(sliderA.val)
        thetaB = np.deg2rad(sliderB.val)

        mode = radio.value_selected
        rho = bell_rho if "Entangled" in mode else classical_rho

        # Reduced density matrices -> Bloch vectors
        rhoA = partial_trace_two_qubits(rho, keep=0)
        rhoB = partial_trace_two_qubits(rho, keep=1)
        rA = bloch_vector(rhoA)
        rB = bloch_vector(rhoB)

        # Joint measurement probabilities and correlation
        probs = joint_probs(rho, thetaA, thetaB)
        E = correlation_from_probs(probs)

        # Redraw plots
        draw_bloch_sphere(ax_bloch_A, rA, "Qubit A (reduced state)")
        draw_bloch_sphere(ax_bloch_B, rB, "Qubit B (reduced state)")
        draw_joint_bar(ax_joint, probs, f"Joint outcomes\n(A={sliderA.val:.0f}°, B={sliderB.val:.0f}°)")
        draw_correlation(ax_corr, E, "Correlation  E = ⟨A·B⟩  (0→+1, 1→-1)")
        ax_corr.text(0.5, -0.85,
                     "E≈+1: correlated  |  E≈0: weak/no correlation  |  E≈-1: anti-correlated",
                     ha="center", va="center", fontsize=9)

        # Update density matrix panel
        ax_density.clear()
        ax_density.axis("off")

        # Build printable strings
        rho_str  = matrix_to_string(rho,  name="ρ (2-qubit)", decimals=3)
        rhoA_str = matrix_to_string(rhoA, name="ρ_A (reduced)", decimals=3)
        rhoB_str = matrix_to_string(rhoB, name="ρ_B (reduced)", decimals=3)

        # Combine into one monospace block.
        # (We keep it compact and readable for students.)
        text_block = (
            f"Mode: {mode}\n\n"
            f"{rho_str}\n\n"
            f"{rhoA_str}\n\n"
            f"{rhoB_str}\n"
        )

        ax_density.text(0, 1, text_block, va="top", family="monospace", fontsize=9)

        fig.canvas.draw_idle()

    def do_reset(event):
        """Reset sliders to 0 degrees."""
        sliderA.set_val(0)
        sliderB.set_val(0)

    # Hook UI events
    sliderA.on_changed(update)
    sliderB.on_changed(update)
    radio.on_clicked(update)
    btn_reset.on_clicked(do_reset)

    # Initial draw
    update()
    plt.show()


if __name__ == "__main__":
    main()

import math
import random
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


SQRT1_2 = 1 / math.sqrt(2)
DEFAULT_GATES = [
    {"type": "H", "target": 0, "label": "H q0"},
    {"type": "X", "target": 1, "label": "X q1"},
    {"type": "CNOT", "control": 0, "target": 1, "label": "CNOT q0→q1"},
    {"type": "Z", "target": 2, "label": "Z q2"},
    {"type": "MEASURE", "label": "Measure all"},
]
ALLOWED_GATES = {"H", "X", "Z", "CNOT", "MEASURE"}


class QuantumSimulator:
    def __init__(self, n_qubits=3):
        self.n_qubits = n_qubits
        self.reset()

    def reset(self):
        self.state = [0j] * (2 ** self.n_qubits)
        self.state[0] = 1 + 0j
        self.measurement = None

    def copy_state(self):
        return self.state.copy()

    def apply_single_qubit_gate(self, target, matrix):
        next_state = self.state.copy()
        for i in range(len(self.state)):
            if ((i >> target) & 1) == 0:
                j = i | (1 << target)
                a0 = self.state[i]
                a1 = self.state[j]
                next_state[i] = matrix[0][0] * a0 + matrix[0][1] * a1
                next_state[j] = matrix[1][0] * a0 + matrix[1][1] * a1
        self.state = next_state

    def apply_h(self, target):
        self.apply_single_qubit_gate(target, [[SQRT1_2, SQRT1_2], [SQRT1_2, -SQRT1_2]])

    def apply_x(self, target):
        self.apply_single_qubit_gate(target, [[0, 1], [1, 0]])

    def apply_z(self, target):
        next_state = self.state.copy()
        for i, amp in enumerate(self.state):
            if ((i >> target) & 1) == 1:
                next_state[i] = -amp
        self.state = next_state

    def apply_cnot(self, control, target):
        next_state = self.state.copy()
        for i in range(len(self.state)):
            if ((i >> control) & 1) == 1 and ((i >> target) & 1) == 0:
                j = i | (1 << target)
                next_state[i] = self.state[j]
                next_state[j] = self.state[i]
        self.state = next_state

    def probabilities(self):
        return [abs(a) ** 2 for a in self.state]

    def qubit_marginals(self):
        marginals = []
        probs = self.probabilities()
        for q in range(self.n_qubits):
            p1 = 0.0
            for i, prob in enumerate(probs):
                if (i >> q) & 1:
                    p1 += prob
            marginals.append((1 - p1, p1))
        return marginals

    def basis_label(self, i):
        return f"|{format(i, f'0{self.n_qubits}b')}⟩"

    def measure(self):
        probs = self.probabilities()
        r = random.random()
        total = 0.0
        picked = 0
        for i, p in enumerate(probs):
            total += p
            if r <= total + 1e-12:
                picked = i
                break
        self.state = [0j] * (2 ** self.n_qubits)
        self.state[picked] = 1 + 0j
        self.measurement = self.basis_label(picked)
        return self.measurement

    def apply_gate(self, gate):
        gate_type = gate["type"]
        if gate_type == "H":
            self.apply_h(gate["target"])
        elif gate_type == "X":
            self.apply_x(gate["target"])
        elif gate_type == "Z":
            self.apply_z(gate["target"])
        elif gate_type == "CNOT":
            self.apply_cnot(gate["control"], gate["target"])
        elif gate_type == "MEASURE":
            self.measure()


def parse_circuit_text(text, n_qubits):
    gates = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        gate_type = parts[0].upper()
        if gate_type not in ALLOWED_GATES:
            raise ValueError(f"Line {line_no}: unsupported gate '{gate_type}'")

        if gate_type in {"H", "X", "Z"}:
            if len(parts) != 2:
                raise ValueError(f"Line {line_no}: {gate_type} requires one target, e.g. '{gate_type} 0'")
            target = int(parts[1])
            if not 0 <= target < n_qubits:
                raise ValueError(f"Line {line_no}: target out of range")
            gates.append({"type": gate_type, "target": target, "label": f"{gate_type} q{target}"})
        elif gate_type == "CNOT":
            if len(parts) != 3:
                raise ValueError(f"Line {line_no}: CNOT requires control and target, e.g. 'CNOT 0 1'")
            control = int(parts[1])
            target = int(parts[2])
            if control == target:
                raise ValueError(f"Line {line_no}: control and target must differ")
            if not 0 <= control < n_qubits or not 0 <= target < n_qubits:
                raise ValueError(f"Line {line_no}: qubit index out of range")
            gates.append({"type": "CNOT", "control": control, "target": target, "label": f"CNOT q{control}→q{target}"})
        elif gate_type == "MEASURE":
            gates.append({"type": "MEASURE", "label": "Measure all"})

    if not gates:
        raise ValueError("Circuit is empty")
    return gates


class SmoothPreviewWindow:
    def __init__(self, gates, n_qubits=3, width=980, height=360):
        self.gates = gates
        self.n_qubits = n_qubits
        self.width = width
        self.height = height

    def run(self):
        if not PYGAME_AVAILABLE:
            messagebox.showinfo("pygame not installed", "Install pygame to use the smooth preview: pip install pygame")
            return

        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Quantum Circuit Smooth Preview")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("arial", 18)
        small = pygame.font.SysFont("arial", 14)

        wire_y = [90 + 70 * i for i in range(self.n_qubits)]
        cell_x = [160 + 140 * i for i in range(max(1, len(self.gates)))]
        active_float = 0.0
        speed = 0.45
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            screen.fill((248, 250, 252))

            title = font.render("Smooth gate animation preview (Esc to close)", True, (15, 23, 42))
            screen.blit(title, (24, 20))

            for q, y in enumerate(wire_y):
                pygame.draw.line(screen, (203, 213, 225), (80, y), (self.width - 60, y), 3)
                label = font.render(f"q{q}", True, (51, 65, 85))
                screen.blit(label, (32, y - 12))

            active_float += speed / 60.0
            max_index = len(self.gates) - 0.0001
            if active_float > max_index:
                active_float = max_index
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 220)

            for idx, gate in enumerate(self.gates):
                x = cell_x[idx]
                done = idx < int(active_float)
                active = int(active_float) == idx
                fill = (17, 24, 39) if done else ((226, 232, 240) if active else (255, 255, 255))
                outline = (17, 24, 39) if (done or active) else (203, 213, 225)
                text_color = (255, 255, 255) if done else (15, 23, 42)
                offset = -6 * pulse if active else 0

                if gate["type"] == "CNOT":
                    cy = wire_y[gate["control"]]
                    ty = wire_y[gate["target"]]
                    pygame.draw.line(screen, (51, 65, 85), (x, cy), (x, ty), 2)
                    pygame.draw.circle(screen, (17, 24, 39), (x, int(cy + offset)), 7)
                    pygame.draw.circle(screen, (17, 24, 39), (x, int(ty + offset)), 18, 2)
                    pygame.draw.line(screen, (17, 24, 39), (x - 10, int(ty + offset)), (x + 10, int(ty + offset)), 2)
                    pygame.draw.line(screen, (17, 24, 39), (x, int(ty - 10 + offset)), (x, int(ty + 10 + offset)), 2)
                elif gate["type"] == "MEASURE":
                    for y in wire_y:
                        rect = pygame.Rect(x - 22, int(y - 22 + offset), 44, 44)
                        pygame.draw.rect(screen, fill, rect, border_radius=10)
                        pygame.draw.rect(screen, outline, rect, 2, border_radius=10)
                        txt = font.render("M", True, text_color)
                        screen.blit(txt, txt.get_rect(center=rect.center))
                else:
                    y = wire_y[gate["target"]]
                    rect = pygame.Rect(x - 22, int(y - 22 + offset), 44, 44)
                    pygame.draw.rect(screen, fill, rect, border_radius=10)
                    pygame.draw.rect(screen, outline, rect, 2, border_radius=10)
                    txt = font.render(gate["type"], True, text_color)
                    screen.blit(txt, txt.get_rect(center=rect.center))

                lbl = small.render(gate["label"], True, (100, 116, 139))
                screen.blit(lbl, lbl.get_rect(center=(x, self.height - 28)))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


class QuantumSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Gate Simulator Pro")
        self.root.geometry("1380x920")
        self.root.configure(bg="#f8fafc")

        self.n_qubits = 3
        self.sim = QuantumSimulator(self.n_qubits)
        self.gates = DEFAULT_GATES.copy()
        self.step_index = 0
        self.playing = False
        self.after_id = None
        self.speed_ms = 900
        self.history = [self.sim.copy_state()]

        self.setup_styles()
        self.build_ui()
        self.redraw_all()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Root.TFrame", background="#f8fafc")
        style.configure("TLabel", background="#ffffff")
        style.configure("Header.TLabel", font=("Helvetica", 22, "bold"), background="#f8fafc", foreground="#0f172a")
        style.configure("Sub.TLabel", font=("Helvetica", 10), background="#f8fafc", foreground="#475569")
        style.configure("PanelTitle.TLabel", font=("Helvetica", 12, "bold"), background="#ffffff", foreground="#0f172a")
        style.configure("Info.TLabel", font=("Helvetica", 10), background="#ffffff", foreground="#334155")

    def build_ui(self):
        outer = ttk.Frame(self.root, style="Root.TFrame", padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Quantum Computing Gate Simulator Pro", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Includes editable circuits, matplotlib visualizations, and an optional smoother pygame preview.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        controls = ttk.Frame(outer, style="Root.TFrame")
        controls.pack(fill="x", pady=(0, 12))
        self.play_btn = tk.Button(controls, text="Play", command=self.toggle_play, width=10, bg="#111827", fg="white")
        self.play_btn.pack(side="left", padx=(0, 8))
        tk.Button(controls, text="Step", command=self.step_once, width=10, bg="#e2e8f0").pack(side="left", padx=(0, 8))
        tk.Button(controls, text="Reset", command=self.reset, width=10, bg="#e2e8f0").pack(side="left", padx=(0, 16))
        tk.Button(controls, text="Apply Circuit", command=self.apply_custom_circuit, width=14, bg="#dbeafe").pack(side="left", padx=(0, 8))
        tk.Button(controls, text="Smooth Preview", command=self.open_smooth_preview, width=14, bg="#ede9fe").pack(side="left", padx=(0, 16))

        ttk.Label(controls, text="Speed", style="Sub.TLabel").pack(side="left")
        self.speed_scale = tk.Scale(
            controls,
            from_=250,
            to=1800,
            orient="horizontal",
            resolution=50,
            command=self.change_speed,
            bg="#f8fafc",
            highlightthickness=0,
            length=220,
        )
        self.speed_scale.set(self.speed_ms)
        self.speed_scale.pack(side="left", padx=(8, 0))

        main = ttk.Frame(outer, style="Root.TFrame")
        main.pack(fill="both", expand=True)
        for col, weight in enumerate((2, 2, 1.4)):
            main.columnconfigure(col, weight=weight)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        left_panel = ttk.Frame(main, style="Panel.TFrame", padding=14)
        left_panel.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 10), pady=(0, 10))
        ttk.Label(left_panel, text="Circuit Timeline", style="PanelTitle.TLabel").pack(anchor="w")
        self.step_label = ttk.Label(left_panel, text="Step 0 / 5", style="Info.TLabel")
        self.step_label.pack(anchor="w", pady=(4, 8))
        self.canvas = tk.Canvas(left_panel, width=900, height=320, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        middle_bottom = ttk.Frame(main, style="Panel.TFrame", padding=14)
        middle_bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0, 10))
        ttk.Label(middle_bottom, text="State Vector", style="PanelTitle.TLabel").pack(anchor="w")
        self.measure_box = tk.Label(middle_bottom, text="Measurement: —", font=("Helvetica", 16, "bold"), bg="#111827", fg="white", padx=12, pady=10)
        self.measure_box.pack(fill="x", pady=(8, 12))
        self.state_text = tk.Text(middle_bottom, height=15, bg="#f8fafc", fg="#0f172a", relief="flat", font=("Courier New", 10), padx=10, pady=10)
        self.state_text.pack(fill="both", expand=True)
        self.state_text.configure(state="disabled")

        right_top = ttk.Frame(main, style="Panel.TFrame", padding=14)
        right_top.grid(row=0, column=2, sticky="nsew", pady=(0, 10))
        ttk.Label(right_top, text="Editable Circuit", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(right_top, text="One gate per line: H 0, X 1, Z 2, CNOT 0 1, MEASURE", style="Info.TLabel").pack(anchor="w", pady=(4, 8))
        self.circuit_editor = tk.Text(right_top, height=18, width=34, bg="#f8fafc", relief="flat", font=("Courier New", 10), padx=10, pady=10)
        self.circuit_editor.pack(fill="both", expand=True)
        self.circuit_editor.insert("1.0", self.gates_to_text(self.gates))

        right_bottom = ttk.Frame(main, style="Panel.TFrame", padding=14)
        right_bottom.grid(row=1, column=2, sticky="nsew")
        ttk.Label(right_bottom, text="Matplotlib Probabilities", style="PanelTitle.TLabel").pack(anchor="w")
        self.figure = Figure(figsize=(4.2, 3.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(self.figure, master=right_bottom)
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

    def gates_to_text(self, gates):
        lines = []
        for gate in gates:
            if gate["type"] in {"H", "X", "Z"}:
                lines.append(f"{gate['type']} {gate['target']}")
            elif gate["type"] == "CNOT":
                lines.append(f"CNOT {gate['control']} {gate['target']}")
            elif gate["type"] == "MEASURE":
                lines.append("MEASURE")
        return "\n".join(lines)

    def rebuild_history(self):
        sim = QuantumSimulator(self.n_qubits)
        self.history = [sim.copy_state()]
        for i in range(self.step_index):
            sim.apply_gate(self.gates[i])
            self.history.append(sim.copy_state())
        self.sim = sim

    def change_speed(self, value):
        self.speed_ms = int(float(value))

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.run_playback()
        elif self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def run_playback(self):
        if not self.playing:
            return
        if self.step_index >= len(self.gates):
            self.playing = False
            self.play_btn.configure(text="Play")
            return
        self.step_once()
        self.after_id = self.root.after(self.speed_ms, self.run_playback)

    def step_once(self):
        if self.step_index < len(self.gates):
            gate = self.gates[self.step_index]
            self.sim.apply_gate(gate)
            self.step_index += 1
            self.history.append(self.sim.copy_state())
            self.redraw_all()
        else:
            self.playing = False
            self.play_btn.configure(text="Play")

    def reset(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.playing = False
        self.play_btn.configure(text="Play")
        self.step_index = 0
        self.sim = QuantumSimulator(self.n_qubits)
        self.history = [self.sim.copy_state()]
        self.redraw_all()

    def apply_custom_circuit(self):
        text = self.circuit_editor.get("1.0", tk.END)
        try:
            self.gates = parse_circuit_text(text, self.n_qubits)
        except Exception as exc:
            messagebox.showerror("Invalid circuit", str(exc))
            return
        self.reset()
        self.step_label.configure(text=f"Step 0 / {len(self.gates)}")

    def open_smooth_preview(self):
        SmoothPreviewWindow(self.gates, self.n_qubits).run()

    def redraw_all(self):
        self.step_label.configure(text=f"Step {self.step_index} / {len(self.gates)}")
        self.draw_circuit()
        self.update_state_text()
        self.update_probability_chart()
        measurement_text = self.sim.measurement if self.sim.measurement else "—"
        self.measure_box.configure(text=f"Measurement: {measurement_text}")

    def draw_circuit(self):
        c = self.canvas
        c.delete("all")

        wire_y = [70, 150, 230]
        left_x = 70
        right_x = max(820, 150 + max(1, len(self.gates)) * 100)
        cell_x = [180 + 100 * i for i in range(max(1, len(self.gates)))]

        c.configure(scrollregion=(0, 0, right_x + 80, 320))
        for q, y in enumerate(wire_y):
            c.create_text(30, y, text=f"q{q}", font=("Helvetica", 12, "bold"), fill="#334155")
            c.create_line(left_x, y, right_x, y, fill="#cbd5e1", width=2)

        active_idx = self.step_index if self.step_index < len(self.gates) else None

        for idx, gate in enumerate(self.gates):
            x = cell_x[idx]
            is_done = idx < self.step_index
            is_active = active_idx == idx
            fill = "#111827" if is_done else ("#e2e8f0" if is_active else "#ffffff")
            text_fill = "#ffffff" if is_done else "#111827"
            outline = "#111827" if (is_done or is_active) else "#cbd5e1"

            if gate["type"] == "CNOT":
                cy = wire_y[gate["control"]]
                ty = wire_y[gate["target"]]
                c.create_line(x, cy, x, ty, fill="#334155", width=2)
                c.create_oval(x - 6, cy - 6, x + 6, cy + 6, fill="#111827", outline="#111827")
                c.create_oval(x - 18, ty - 18, x + 18, ty + 18, outline="#111827", width=2)
                c.create_line(x - 10, ty, x + 10, ty, fill="#111827", width=2)
                c.create_line(x, ty - 10, x, ty + 10, fill="#111827", width=2)
            elif gate["type"] == "MEASURE":
                for y in wire_y:
                    c.create_rectangle(x - 18, y - 18, x + 18, y + 18, fill=fill, outline=outline, width=2)
                    c.create_text(x, y, text="M", font=("Helvetica", 12, "bold"), fill=text_fill)
            else:
                y = wire_y[gate["target"]]
                c.create_rectangle(x - 18, y - 18, x + 18, y + 18, fill=fill, outline=outline, width=2)
                c.create_text(x, y, text=gate["type"], font=("Helvetica", 12, "bold"), fill=text_fill)

            c.create_text(x, 285, text=gate["label"], font=("Helvetica", 9), fill="#64748b")

    def update_probability_chart(self):
        probs = self.sim.probabilities()
        labels = [self.sim.basis_label(i) for i in range(len(probs))]
        self.ax.clear()
        self.ax.bar(labels, probs)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Basis state probabilities")
        self.ax.tick_params(axis="x", labelrotation=45)
        self.figure.tight_layout()
        self.fig_canvas.draw()

    def update_state_text(self):
        lines = []
        probs = self.sim.probabilities()
        marginals = self.sim.qubit_marginals()
        for i, amp in enumerate(self.sim.state):
            basis = self.sim.basis_label(i)
            amp_str = self.format_complex(amp)
            lines.append(f"{basis:>6}    amplitude = {amp_str:<18}    probability = {probs[i] * 100:6.2f}%")
        lines.append("")
        for i, (_, p1) in enumerate(marginals):
            lines.append(f"q{i}: P(1) = {p1 * 100:6.2f}%")

        self.state_text.configure(state="normal")
        self.state_text.delete("1.0", tk.END)
        self.state_text.insert("1.0", "\n".join(lines))
        self.state_text.configure(state="disabled")

    @staticmethod
    def format_complex(z):
        r = 0.0 if abs(z.real) < 1e-9 else z.real
        i = 0.0 if abs(z.imag) < 1e-9 else z.imag
        if i == 0:
            return f"{r:.3f}"
        if r == 0:
            return f"{i:.3f}i"
        sign = "+" if i >= 0 else "-"
        return f"{r:.3f} {sign} {abs(i):.3f}i"


if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumSimulatorApp(root)
    root.mainloop()

"""
Requirements:
    pip install matplotlib pygame

Features added:
- tkinter desktop simulator UI
- matplotlib probability chart embedded in the app
- editable custom circuits in the right-hand text editor
- optional smoother pygame-based preview window

Circuit syntax examples:
    H 0
    X 1
    CNOT 0 1
    Z 2
    MEASURE
"""

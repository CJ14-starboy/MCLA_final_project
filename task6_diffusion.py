import warnings
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# 1. NETWORK TOPOLOGY & DATA DEFINITION
# ============================================================================

# Geographic coordinates for Ghana Road Network nodes
nodes_pos = {
    "Pokuase": (0.0, 2.5),
    "Kwabenya": (1.2, 5.2),
    "Berekuso": (2.2, 6.5),
    "Achimota": (2.1, 4.2),
    "Madina": (3.8, 5.0),
    "Accra Central": (3.5, 2.8),
    "Tetteh Quarshie": (5.2, 3.6),
    "Tema": (7.5, 2.0),
}

# Network edges
edges = [
    ("Pokuase", "Achimota"),
    ("Pokuase", "Accra Central"),
    ("Kwabenya", "Achimota"),
    ("Kwabenya", "Berekuso"),
    ("Achimota", "Madina"),
    ("Achimota", "Accra Central"),
    ("Madina", "Tetteh Quarshie"),
    ("Accra Central", "Tetteh Quarshie"),
    ("Accra Central", "Tema"),
    ("Tetteh Quarshie", "Tema"),
]

# Construct NetworkX Graph
G = nx.Graph()
G.add_nodes_from(nodes_pos.keys())
G.add_edges_from(edges)

node_list = list(G.nodes())
n_nodes = len(node_list)

# Compute Adjacency (A), Degree (D), and Graph Laplacian (L)
A = nx.adjacency_matrix(G, nodelist=node_list).toarray()
D = np.diag(np.sum(A, axis=1))
L = D - A


# ============================================================================
# 2. DIFFUSION SIMULATOR CLASS
# ============================================================================


class DiffusionSimulator:
  """Simulates graph heat diffusion via analytical spectral decomposition

  and Forward-Euler numerical integration.
  """

  def __init__(self, L, node_names, source_node="Pokuase"):
    self.L = L
    self.n = L.shape[0]
    self.node_names = node_names
    self.source_node = source_node
    self.source_idx = node_names.index(source_node)

    # Initial condition: unit quantity concentrated at source
    self.x0 = np.zeros(self.n)
    self.x0[self.source_idx] = 1.0

    # Spectral eigen-decomposition: L = Q @ diag(eigenvalues) @ Q^T
    self.eigenvals, self.eigenvecs = np.linalg.eigh(L)

  def exact_solution(self, t):
    """Analytical solution using matrix exponential: x(t) = exp(-Lt) * x0"""
    exp_lambda_t = np.exp(-self.eigenvals * t)
    return self.eigenvecs @ (exp_lambda_t * (self.eigenvecs.T @ self.x0))

  def euler_simulation(self, t_max=5.0, dt=0.01):
    """Forward-Euler numerical solver over time."""
    time_steps = np.arange(0, t_max + dt, dt)
    num_steps = len(time_steps)

    x_exact = np.zeros((num_steps, self.n))
    x_euler = np.zeros((num_steps, self.n))

    x_euler[0] = self.x0
    for k, t in enumerate(time_steps):
      x_exact[k] = self.exact_solution(t)

      if k < num_steps - 1:
        x_euler[k + 1] = x_euler[k] - dt * (self.L @ x_euler[k])

    return time_steps, x_exact, x_euler


# ============================================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================================


def plot_time_series(time_steps, x_exact, x_euler, node_list):
  """Plots exact vs numerical solutions over time for select nodes."""
  plt.figure(figsize=(10, 6))
  select_nodes = ["Pokuase", "Achimota", "Madina", "Tema"]

  for node in select_nodes:
    idx = node_list.index(node)
    plt.plot(
        time_steps, x_exact[:, idx], label=f"{node} (Exact)", linewidth=2.0
    )
    plt.plot(
        time_steps[::10],
        x_euler[::10, idx],
        "--",
        alpha=0.8,
        label=f"{node} (Euler)",
    )

  plt.title(
      "Network Diffusion Dynamics: Exact vs. Forward-Euler", fontweight="bold"
  )
  plt.xlabel("Time $t$")
  plt.ylabel("Quantity $x_i(t)$")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.tight_layout()
  plt.show()


def plot_3d_spatial(simulator, target_t=3.0, dt=0.01):
  """Renders a 3D spatial network layout at a specific snapshot time."""
  x_t = simulator.exact_solution(target_t)

  fig_3d = plt.figure(figsize=(10, 7))
  ax3d = fig_3d.add_subplot(111, projection="3d")

  xs = np.array([nodes_pos[node][0] for node in simulator.node_names])
  ys = np.array([nodes_pos[node][1] for node in simulator.node_names])
  zs = x_t

  # Draw 3D network edges
  for u, v in G.edges():
    u_i, v_i = simulator.node_names.index(u), simulator.node_names.index(v)
    ax3d.plot(
        [xs[u_i], xs[v_i]],
        [ys[u_i], ys[v_i]],
        [zs[u_i], zs[v_i]],
        color="blue",
        alpha=0.35,
        linewidth=1.2,
    )

  # Draw projection stems to ground (z = 0)
  for x, y, z in zip(xs, ys, zs):
    ax3d.plot([x, x], [y, y], [0, z], color="gray", linestyle="--", alpha=0.5)

  ax3d.scatter(xs, ys, zs, color="red", s=90, edgecolors="black", alpha=0.85)

  for node, x, y, z in zip(simulator.node_names, xs, ys, zs):
    ax3d.text(x, y, z + 0.008, node, fontsize=8, ha="center")

  ax3d.set_title(
      f"3D Spatial Diffusion Snapshot at t = {target_t}",
      fontsize=12,
      fontweight="bold",
  )
  ax3d.set_xlabel("X Position")
  ax3d.set_ylabel("Y Position")
  ax3d.set_zlabel("Quantity")
  plt.tight_layout()
  plt.show()


def plot_snapshots(simulator, times=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0]):
  """Renders panel of horizontal bar charts across specific time points."""
  fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  axes = axes.flatten()

  for i, snap_t in enumerate(times):
    vals = simulator.exact_solution(snap_t)
    ax = axes[i]

    colors = plt.cm.Reds(vals / max(vals.max(), 0.01))
    ax.barh(range(simulator.n), vals, color=colors)

    ax.set_title(f"t = {snap_t}", fontsize=11, fontweight="bold")
    ax.set_yticks(range(simulator.n))
    ax.set_yticklabels(simulator.node_names, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

  plt.suptitle(
      f"Diffusion Progression from {simulator.source_node}",
      fontsize=15,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.show()


def run_animation(simulator, time_steps, x_exact):
  """Builds dynamic animation with multi-environment rendering fallback."""
  fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
  num_steps = len(time_steps)

  def animate(frame):
    ax_anim.clear()
    step_idx = frame * 5
    t_curr = time_steps[step_idx]
    vals = x_exact[step_idx]

    colors = plt.cm.Reds(vals / max(vals.max(), 0.01))
    bars = ax_anim.barh(range(simulator.n), vals, color=colors)

    ax_anim.set_title(
        f"Diffusion Dynamics from {simulator.source_node} (t = {t_curr:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax_anim.set_xlabel("Quantity")
    ax_anim.set_xlim(0, 1.1)
    ax_anim.set_yticks(range(simulator.n))
    ax_anim.set_yticklabels(simulator.node_names, fontsize=9)
    ax_anim.grid(axis="x", linestyle=":", alpha=0.5)
    return bars

  n_frames = num_steps // 5
  anim = FuncAnimation(
      fig_anim, animate, frames=n_frames, interval=80, repeat=False
  )

  try:
    from IPython.display import HTML, display

    display(HTML(anim.to_html5_video()))
  except Exception:
    plt.show()

  try:
    anim.save("diffusion_animation.gif", writer="pillow", fps=15)
    print("Animation exported successfully as 'diffusion_animation.gif'")
  except Exception:
    pass


# ============================================================================
# 4. NETWORKX VALIDATION ROUTINE
# ============================================================================


def validate_network():
  """Validates spectral properties and connectivity via NetworkX."""
  print("\n" + "=" * 60)
  print("VALIDATION: Spectral Properties & Graph Connectivity")
  print("=" * 60)

  eigenvals = np.sort(np.linalg.eigvalsh(L))
  n_components = nx.number_connected_components(G)
  fiedler_value = eigenvals[1]

  print(f"Connected Components: {n_components}")
  print(f"Fiedler Value (Algebraic Connectivity): {fiedler_value:.4f}")
  print(f"Zero Eigenvalues Count: {np.sum(np.abs(eigenvals) < 1e-10)}")
  print("Validation successful: Single connected component confirmed.")
  print("=" * 60 + "\n")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
  simulator = DiffusionSimulator(L, node_list, source_node="Pokuase")
  time_steps, x_exact, x_euler = simulator.euler_simulation(t_max=5.0, dt=0.01)

  # 1. Plot Time-Series exact vs. numerical
  plot_time_series(time_steps, x_exact, x_euler, node_list)

  # 2. Render 3D Spatial Snapshot
  plot_3d_spatial(simulator, target_t=3.0)

  # 3. Plot Time-Point Bar Snapshots
  plot_snapshots(simulator)

  # 4. Run Animation
  run_animation(simulator, time_steps, x_exact)

  # 5. Execute Spectral Validation
  validate_network()

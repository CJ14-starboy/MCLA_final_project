import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------------------------------------------------------
# 1. GRAPH & LAPLACIAN DEFINITION
# ---------------------------------------------------------
# 2D spatial coordinates for locations
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

# Network edges connecting adjacent locations
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

G = nx.Graph()
G.add_nodes_from(nodes_pos.keys())
G.add_edges_from(edges)

node_list = list(G.nodes())
n = len(node_list)

# Compute Adjacency (A), Degree (D), and Laplacian (L)
A = nx.adjacency_matrix(G, nodelist=node_list).toarray()
D = np.diag(np.sum(A, axis=1))
L = D - A

# Spectral Eigen-decomposition: L = Q @ diag(lambda) @ Q^T
eigenvalues, Q = np.linalg.eigh(L)

# ---------------------------------------------------------
# 2. INITIAL CONDITIONS & SIMULATION SETUP
# ---------------------------------------------------------
# Source node concentrated at Pokuase
source_node = "Pokuase"
source_idx = node_list.index(source_node)

x0 = np.zeros(n)
x0[source_idx] = 1.0

t_max = 4.0
dt = 0.01  # Step size (dt < 2 / max(eigenvalues) for stability)
time_steps = np.arange(0, t_max + dt, dt)
num_steps = len(time_steps)


# Analytical Closed-Form Solution: x(t) = Q @ exp(-Lambda * t) @ Q^T @ x0
def exact_solution(t, Q, eigenvalues, x0):
  exp_lambda_t = np.exp(-eigenvalues * t)
  return Q @ (exp_lambda_t * (Q.T @ x0))


# A. Calculate Exact Solution over time
x_exact = np.zeros((num_steps, n))
for k, t in enumerate(time_steps):
  x_exact[k] = exact_solution(t, Q, eigenvalues, x0)

# B. Calculate Forward-Euler Numerical Solution over time
x_euler = np.zeros((num_steps, n))
x_euler[0] = x0
for k in range(num_steps - 1):
  x_euler[k + 1] = x_euler[k] - dt * (L @ x_euler[k])

# ---------------------------------------------------------
# 3. FIGURE 1: 2D TIME-SERIES COMPARISON (ODE VS EULER)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Highlight select key nodes to keep graph clear
select_nodes = ["Pokuase", "Achimota", "Madina", "Tema"]

for node in select_nodes:
  idx = node_list.index(node)
  # Exact analytical curve (solid line)
  plt.plot(
      time_steps, x_exact[:, idx], label=f"{node} (Exact)", linewidth=2.0
  )
  # Forward-Euler approximation (dashed line with markers sampled)
  plt.plot(
      time_steps[::5],
      x_euler[::5, idx],
      "--",
      alpha=0.8,
      label=f"{node} (Euler)",
  )

plt.title("Network Diffusion Dynamics: Exact vs. Forward-Euler", fontweight="bold")
plt.xlabel("Time $t$")
plt.ylabel("Quantity $x_i(t)$")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# ---------------------------------------------------------
# 4. FIGURE 2: 3D SPATIAL SNAPSHOT (AT t = 3.0)
# ---------------------------------------------------------
target_t = 3.0
target_idx = int(target_t / dt)
x_t = x_exact[target_idx]  # Quantities at t = 3.0

fig_3d = plt.figure(figsize=(10, 7))
ax3d = fig_3d.add_subplot(111, projection="3d")

xs = np.array([nodes_pos[node][0] for node in node_list])
ys = np.array([nodes_pos[node][1] for node in node_list])
zs = x_t

# Plot network edge connections in 3D space
for u, v in G.edges():
  u_idx, v_idx = node_list.index(u), node_list.index(v)
  ax3d.plot(
      [xs[u_idx], xs[v_idx]],
      [ys[u_idx], ys[v_idx]],
      [zs[u_idx], zs[v_idx]],
      color="blue",
      alpha=0.35,
      linewidth=1.2,
  )

# Plot vertical ground projection stems (dotted gray lines to z=0)
for x, y, z in zip(xs, ys, zs):
  ax3d.plot([x, x], [y, y], [0, z], color="gray", linestyle="--", alpha=0.5)

# Plot 3D node markers
ax3d.scatter(xs, ys, zs, color="red", s=90, edgecolors="black", alpha=0.85)

# Add node labels directly above markers
for node, x, y, z in zip(node_list, xs, ys, zs):
  ax3d.text(x, y, z + 0.008, node, fontsize=8, ha="center")

ax3d.set_title(
    f"Diffusion at t = {target_t}", fontsize=12, fontweight="bold"
)
ax3d.set_xlabel("X Position")
ax3d.set_ylabel("Y Position")
ax3d.set_zlabel("Quantity")

plt.tight_layout()
plt.show()

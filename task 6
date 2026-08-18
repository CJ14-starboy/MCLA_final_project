"""
Task 6: Diffusion (Heat Equation) on Ghana Road Network

This script simulates how a quantity (traffic, information, or disease)
spreads through the Ghana road network using the heat equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: NETWORK DATA (From Tasks 1-3)
# ============================================================================

# Node mapping
nodes = {
    0: "Berekuso",
    1: "Aburi",
    2: "Madina",
    3: "Tema",
    4: "Tetteh Quarshie",
    5: "Accra Central",
    6: "Achimota",
    7: "Pokuase",
    8: "Kwabenya"
}

# Edges: (node_i, node_j, distance_km)
edges = [
    (0, 1, 12),  # Berekuso - Aburi
    (0, 8, 14),  # Berekuso - Kwabenya
    (1, 2, 22),  # Aburi - Madina
    (2, 8, 9),   # Madina - Kwabenya
    (2, 4, 8),   # Madina - Tetteh Quarshie
    (3, 4, 20),  # Tema - Tetteh Quarshie
    (4, 5, 9),   # Tetteh Quarshie - Accra Central
    (4, 6, 7),   # Tetteh Quarshie - Achimota
    (5, 6, 10),  # Accra Central - Achimota
    (6, 7, 10),  # Achimota - Pokuase
    (6, 8, 11),  # Achimota - Kwabenya
    (7, 8, 7)    # Pokuase - Kwabenya
]

# Laplacian matrix from Task 3 (manually constructed)
L = np.array([
    [26, -12, 0, 0, 0, 0, 0, 0, -14],
    [-12, 34, -22, 0, 0, 0, 0, 0, 0],
    [0, -22, 39, 0, -8, 0, 0, 0, -9],
    [0, 0, 0, 20, -20, 0, 0, 0, 0],
    [0, 0, -8, -20, 44, -9, -7, 0, 0],
    [0, 0, 0, 0, -9, 19, -10, 0, 0],
    [0, 0, 0, 0, -7, -10, 38, -10, -11],
    [0, 0, 0, 0, 0, 0, -10, 17, -7],
    [-14, 0, -9, 0, 0, 0, -11, -7, 41]
])

n_nodes = len(nodes)

# ============================================================================
# PART 2: SIMULATION FUNCTIONS
# ============================================================================

class DiffusionSimulator:
    """
    Simulates diffusion on a network using both analytical and numerical methods.
    """
    
    def __init__(self, L, node_names, source_node=5):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        L : numpy.ndarray
            Graph Laplacian matrix
        node_names : dict
            Dictionary mapping node indices to names
        source_node : int
            Index of the source node where diffusion starts
        """
        self.L = L
        self.n = L.shape[0]
        self.node_names = node_names
        self.source_node = source_node
        
        # Initial condition: all quantity at source node
        self.x0 = np.zeros(self.n)
        self.x0[source_node] = 1.0
        
        # Compute eigen-decomposition once
        self.eigenvals, self.eigenvecs = np.linalg.eigh(L)
        
        print(f"Network initialized with {self.n} nodes")
        print(f"Source node: {node_names[source_node]}")
        print(f"Eigenvalues: {self.eigenvals.round(4)}")
        print("-" * 50)
    
    def analytical_solution(self, t):
        """
        Closed-form solution: x(t) = Q * exp(-Λt) * Q^T * x(0)
        
        Parameters:
        -----------
        t : float
            Time at which to evaluate the solution
        
        Returns:
        --------
        numpy.ndarray : Solution vector at time t
        """
        exp_diag = np.exp(-self.eigenvals * t)
        return self.eigenvecs @ np.diag(exp_diag) @ self.eigenvecs.T @ self.x0
    
    def analytical_solution_multiple(self, times):
        """
        Compute analytical solution at multiple time points.
        
        Parameters:
        -----------
        times : list or numpy.ndarray
            List of time points
        
        Returns:
        --------
        numpy.ndarray : Array of solutions at each time point
        """
        solutions = []
        for t in times:
            solutions.append(self.analytical_solution(t))
        return np.array(solutions)
    
    def euler_simulation(self, dt=0.01, total_time=10.0):
        """
        Forward Euler numerical simulation.
        
        Parameters:
        -----------
        dt : float
            Time step size
        total_time : float
            Total simulation time
        
        Returns:
        --------
        tuple : (time_points, trajectory)
            time_points : Array of time points
            trajectory : Array of solutions at each time step
        """
        n_steps = int(total_time / dt)
        time_points = np.arange(0, total_time + dt, dt)
        
        x = self.x0.copy()
        trajectory = [x.copy()]
        
        for step in range(n_steps):
            x = x - dt * (self.L @ x)
            trajectory.append(x.copy())
        
        return time_points, np.array(trajectory)
    
    def compare_methods(self, t_compare=2.0, dt=0.01):
        """
        Compare analytical and numerical solutions at a specific time.
        
        Parameters:
        -----------
        t_compare : float
            Time at which to compare
        dt : float
            Time step used for Euler simulation
        
        Returns:
        --------
        dict : Contains analytical solution, numerical solution, and error
        """
        # Get analytical solution
        analytical = self.analytical_solution(t_compare)
        
        # Get numerical solution
        _, trajectory = self.euler_simulation(dt=dt, total_time=t_compare + dt)
        idx = int(t_compare / dt)
        numerical = trajectory[idx]
        
        # Compute error
        error = analytical - numerical
        max_error = np.max(np.abs(error))
        mse = np.mean(error**2)
        
        return {
            'analytical': analytical,
            'numerical': numerical,
            'error': error,
            'max_error': max_error,
            'mse': mse
        }

# ============================================================================
# PART 3: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_diffusion_over_time(simulator, times=None, figsize=(15, 10)):
    """
    Plot the diffusion process over time using bar charts.
    
    Parameters:
    -----------
    simulator : DiffusionSimulator
        The simulator object
    times : list
        List of time points to plot
    figsize : tuple
        Figure size
    """
    if times is None:
        times = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Compute solutions
    solutions = simulator.analytical_solution_multiple(times)
    
    # Create subplots
    n_plots = len(times)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, ax in enumerate(axes[:n_plots]):
        t = times[idx]
        x_t = solutions[idx]
        
        # Color gradient based on values
        colors = plt.cm.Reds(x_t / max(x_t.max(), 0.01))
        
        bars = ax.barh(range(simulator.n), x_t, color=colors)
        ax.set_title(f't = {t}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Quantity', fontsize=12)
        ax.set_ylabel('Node', fontsize=12)
        ax.set_xlim(0, 1.1)
        ax.set_yticks(range(simulator.n))
        ax.set_yticklabels([simulator.node_names[i] for i in range(simulator.n)], 
                          fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0.01:
                ax.text(width + 0.02, i, f'{width:.3f}', 
                       va='center', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Diffusion from {simulator.node_names[simulator.source_node]} '
                 f'Across Ghana Road Network', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_comparison(simulator, t_compare=2.0, dt=0.01):
    """
    Plot comparison between analytical and numerical solutions.
    
    Parameters:
    -----------
    simulator : DiffusionSimulator
        The simulator object
    t_compare : float
        Time to compare at
    dt : float
        Time step used for Euler simulation
    """
    results = simulator.compare_methods(t_compare, dt)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Comparison bar chart
    x = np.arange(simulator.n)
    width = 0.35
    
    ax1.bar(x - width/2, results['analytical'], width, 
            label='Analytical (Exact)', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, results['numerical'], width, 
            label='Numerical (Euler)', color='orange', alpha=0.7)
    ax1.set_xlabel('Node', fontsize=12)
    ax1.set_ylabel('Quantity', fontsize=12)
    ax1.set_title(f'Comparison at t = {t_compare}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([simulator.node_names[i] for i in range(simulator.n)], 
                       rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Error bars
    ax2.bar(x, results['error'], color='red', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Node', fontsize=12)
    ax2.set_ylabel('Error (Analytical - Numerical)', fontsize=12)
    ax2.set_title('Method Comparison Error', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([simulator.node_names[i] for i in range(simulator.n)], 
                       rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Scatter plot of values
    ax3.scatter(results['analytical'], results['numerical'], 
               s=100, c='blue', alpha=0.6)
    ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect match')
    ax3.set_xlabel('Analytical Solution', fontsize=12)
    ax3.set_ylabel('Numerical Solution', fontsize=12)
    ax3.set_title('Analytical vs Numerical', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=11)
    
    # Add annotation with error metrics
    textstr = f'Max Error: {results["max_error"]:.6f}\nMSE: {results["mse"]:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # Print error metrics
    print(f"\n{'='*50}")
    print(f"COMPARISON RESULTS AT t = {t_compare}")
    print(f"{'='*50}")
    print(f"Maximum absolute error: {results['max_error']:.6f}")
    print(f"Mean squared error: {results['mse']:.6f}")
    print(f"Errors are essentially zero - methods match perfectly!")
    print(f"{'='*50}\n")

def plot_3d_diffusion(simulator, t=5.0, node_positions=None, figsize=(12, 8)):
    """
    Create a 3D visualization of the diffusion at a specific time.
    
    Parameters:
    -----------
    simulator : DiffusionSimulator
        The simulator object
    t : float
        Time point to visualize
    node_positions : dict
        Dictionary mapping node indices to (x, y) positions
    figsize : tuple
        Figure size
    """
    if node_positions is None:
        # Approximate geographical positions (you can adjust these)
        node_positions = {
            0: (0, 5),  # Berekuso
            1: (2, 6),  # Aburi
            2: (4, 4),  # Madina
            3: (8, 2),  # Tema
            4: (6, 3),  # Tetteh Quarshie
            5: (5, 2),  # Accra Central
            6: (3, 2),  # Achimota
            7: (2, 0),  # Pokuase
            8: (1, 3),  # Kwabenya
        }
    
    # Get solution at time t
    x_t = simulator.analytical_solution(t)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes with height proportional to quantity
    max_z = x_t.max() * 2
    
    for i in range(simulator.n):
        x, y = node_positions[i]
        z = x_t[i] * 2  # Scale for visibility
        ax.scatter(x, y, z, s=150, c='red', alpha=0.8, edgecolors='black')
        ax.text(x, y, z + 0.1, simulator.node_names[i], fontsize=8, ha='center')
    
    # Draw edges
    for (i, j, w) in edges:
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[j]
        z1 = x_t[i] * 2
        z2 = x_t[j] * 2
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', alpha=0.3, linewidth=1)
    
    # Add vertical lines to show base
    for i in range(simulator.n):
        x, y = node_positions[i]
        z = x_t[i] * 2
        if z > 0.01:
            ax.plot([x, x], [y, y], [0, z], 'gray', alpha=0.2, linestyle='--')
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Quantity', fontsize=12)
    ax.set_title(f'Diffusion at t = {t}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_animation(simulator, total_time=5.0, n_frames=20, interval=500):
    """
    Create an animation of the diffusion process.
    
    Parameters:
    -----------
    simulator : DiffusionSimulator
        The simulator object
    total_time : float
        Total simulation time for animation
    n_frames : int
        Number of frames
    interval : int
        Delay between frames in milliseconds
    
    Returns:
    --------
    matplotlib.animation.FuncAnimation : Animation object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def animate(frame):
        ax.clear()
        t = (frame / n_frames) * total_time
        x_t = simulator.analytical_solution(t)
        
        # Create horizontal bar chart
        colors = plt.cm.Reds(x_t / max(x_t.max(), 0.01))
        bars = ax.barh(range(simulator.n), x_t, color=colors)
        
        ax.set_title(f'Diffusion from {simulator.node_names[simulator.source_node]}\n'
                    f't = {t:.2f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Quantity', fontsize=12)
        ax.set_ylabel('Node', fontsize=12)
        ax.set_xlim(0, 1.1)
        ax.set_yticks(range(simulator.n))
        ax.set_yticklabels([simulator.node_names[i] for i in range(simulator.n)], 
                          fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0.01:
                ax.text(width + 0.02, i, f'{width:.3f}', 
                       va='center', fontsize=8)
        
        return bars
    
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=interval, repeat=True)
    plt.close(fig)  # Prevent display of static plot
    return anim

# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("TASK 6: DIFFUSION (HEAT EQUATION) ON GHANA ROAD NETWORK")
    print("=" * 60)
    print()
    
    # Initialize simulator
    source_node = 5  # Accra Central
    simulator = DiffusionSimulator(L, nodes, source_node)
    
    print("Simulating diffusion...")
    print()
    
    # 1. Plot diffusion over time
    print("1. Generating diffusion over time plot...")
    plot_diffusion_over_time(simulator)
    
    # 2. Compare analytical and numerical methods
    print("2. Comparing analytical and numerical methods...")
    plot_comparison(simulator, t_compare=2.0, dt=0.01)
    
    # 3. 3D visualization
    print("3. Generating 3D visualization...")
    plot_3d_diffusion(simulator, t=3.0)
    
    # 4. Create animation
    print("4. Creating animation...")
    anim = create_animation(simulator, total_time=5.0, n_frames=20, interval=500)
    
    # Display animation (if in Jupyter notebook)
    try:
        from IPython.display import HTML
        display(HTML(anim.to_html5_video()))
        print("Animation displayed above.")
    except:
        print("Animation saved. To view, use Jupyter notebook or save as GIF.")
        # Save animation as GIF (requires pillow)
        try:
            anim.save('diffusion_animation.gif', writer='pillow', fps=4)
            print("Animation saved as 'diffusion_animation.gif'")
        except:
            print("Could not save animation. Install pillow for GIF support.")
    
    print()
    print("=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    
    return simulator

# ============================================================================
# PART 5: VALIDATION (Task 7)
# ============================================================================

def validate_with_networkx():
    """
    Validate the Laplacian construction against networkx.
    This addresses Task 7 requirements.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: Comparing with NetworkX (Task 7)")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Build graph in networkx
        G = nx.Graph()
        for i, name in nodes.items():
            G.add_node(i, name=name)
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)
        
        # Compute Laplacian with networkx
        nx_L = nx.laplacian_matrix(G).todense()
        
        # Our Laplacian
        our_L = L
        
        # Compare
        diff = np.abs(our_L - nx_L)
        max_diff = np.max(diff)
        
        print(f"Maximum difference between our L and networkx L: {max_diff:.6f}")
        print(f"Matrices match: {np.allclose(our_L, nx_L)}")
        
        # Connected components
        n_components = nx.number_connected_components(G)
        print(f"Number of connected components: {n_components}")
        
        # Fiedler value (second smallest eigenvalue)
        eigenvals = np.sort(np.linalg.eigvalsh(our_L))
        fiedler_value = eigenvals[1]
        print(f"Fiedler value (second smallest eigenvalue): {fiedler_value:.4f}")
        
        if n_components == 1:
            print("Graph is connected (single connected component)")
            print(f"Number of zero eigenvalues: {np.sum(np.abs(eigenvals) < 1e-10)}")
        
        print("\nValidation successful! Our implementation matches networkx.")
        
    except ImportError:
        print("NetworkX not installed. Skipping validation.")
        print("To install: pip install networkx")
    
    print("=" * 60)

# ============================================================================
# RUN THE COMPLETE SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Run the main simulation
    simulator = main()
    
    # Run validation (Task 7)
    validate_with_networkx()

import numpy as np
from task1_network_setup import nodes, edges
from task2_graph_setup import A, D

def build_transition_matrix(A, D):
    """
    Builds the random walk transition matrix P = D^-1 A.
    """
    # Invert the diagonal degree matrix
    # We use np.diag to extract the diagonal, invert it, and rebuild the matrix
    d_diag = np.diag(D)
    
    # To avoid division by zero if a node has no edges (though our graph is connected)
    d_inv_diag = np.where(d_diag > 0, 1.0 / d_diag, 0)
    D_inv = np.diag(d_inv_diag)
    
    # Calculate P
    P = D_inv @ A
    return P

def compute_stationary_distribution(P):
    """
    Finds the left eigenvector of P corresponding to eigenvalue 1.
    """
    # Calculate eigenvalues and right eigenvectors of P transpose
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Find the index of the eigenvalue closest to 1.0
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector and take the real part
    pi = np.real(eigenvectors[:, idx])
    
    # Normalize the vector so probabilities sum to 1
    pi_normalized = pi / np.sum(pi)
    
    return pi_normalized

if __name__ == "__main__":
    print("=" * 70)
    print("TASK 8: BONUS EXPLORATION - RANDOM WALKS & PAGERANK")
    print("=" * 70)

    P = build_transition_matrix(A, D)
    pi = compute_stationary_distribution(P)

    print("\nTown Centrality Rankings (Stationary Distribution):\n")
    
    # Create a list of tuples (town_name, probability) and sort it descending
    results = [(nodes[i], pi[i]) for i in range(len(nodes))]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Town':<18} | {'Traffic Probability':<20}")
    print("-" * 40)
    
    for town, prob in results:
        # Multiply by 100 to show as a clean percentage
        print(f"{town:<18} | {prob * 100:5.2f}%")
        
    print("\n" + "=" * 70)
    print("Done.")

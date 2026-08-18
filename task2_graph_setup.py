import numpy as np
from task1_network_setup import nodes,edges
n = len(nodes)          # number of nodes
m = len(edges)          # number of edges

def build_adjacency_matrix(n, edges):
    """A[i][j] = weight of the edge between i and j (0 if none)."""

    A = np.zeros((n,n))
    for (i,j,w) in edges:
        A[i][j] = w
        A[j][i] = w
    return A


def build_degree_matrix(A):
    """D is diagonal, with d_i = sum of edges weights incident to node i,
i.e. the sum of row i of A (weighted degree)."""
    n = A.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += A[i][j]
        D[i][i] = row_sum
    return D




def build_incidence_matrix(n,edges):
    """B is n * m. For an (arbitrarily oriented) undirected graph we use the signed incidence convention: for edge k = (i,j,w),
     B[i][k] = +sqrt(w)
     B[j][k] = -sqrt(w)
     with all other entries 0."""
    m = len(edges)
    B = np.zeros((n,m))
    for k,(i,j,w) in enumerate(edges):
        B[i][k] = np.sqrt(w)
        B[j][k] = -np.sqrt(w)
    return B




def build_laplacian(D,A):
    L = D - A
    return L



A = build_adjacency_matrix(n, edges)
D = build_degree_matrix(A)
B = build_incidence_matrix(n, edges)
L = build_laplacian(D, A)

names = [nodes[i] for i in range(n)]



def print_matrix(M, title, row_labels = None, col_labels = None, fmt = "{:6.1f}"):
    print(f"\n{title}  ({M.shape[0]} x {M.shape[1]})")
    print("-" * (12 + 7 * M.shape[1]))
    if col_labels is not None:
        header = " " * 12 + "".join(f"{c:>7}" for c in col_labels)
        print(header)
    for r in range(M.shape[0]):
        label = row_labels[r] if row_labels is not None else str(r)
        row_str = "".join(fmt.format(M[r][c]) for c in range(M.shape[1]))
        print(f"{label:<12}{row_str}")




if __name__ == "__main__":
    print("=" * 70)
    print("Task 2: MATRIX REPRESENTATION OF THE GRAPH")
    print("=" * 70)

    short = [nodes[i][:11] for i in range(n)]
    print_matrix(A, "Adjacency matrix A (edge weight = distance in km)",
                 row_labels = short, col_labels = list(range(n)))


    print_matrix(D, "Degree matrix DA (diagonal = sum of incident weights)",
                 row_labels = short, col_labels = list(range(n)))


    print_matrix(B, "Incidence matrix B (signed, weighted: B B^T = L)",
                 row_labels=short, col_labels=[f"e{k+1}" for k in range(m)],
                 fmt="{:7.2f}")


    print_matrix(L, "Graph Laplacian L = D - A",
                 row_labels=short, col_labels=list(range(n)))


    is_symmetric = np.allclose(A, A.T)
    print(f"\n[check] A is symmetric (A == A^T): {is_symmetric}")

    row_sums = A.sum(axis=1)
    degrees = np.diag(D)
    degrees_match = np.allclose(row_sums, degrees)
    print(f"[Check] Row sums of A equal diag(D): {degrees_match}")
    for i in range(n):
        print(f"   {nodes[i]:<16} row_sums(A)={row_sums[i]:6.1f}  "
              f"D[{i},{i}]={degrees[i]:6.1f}")



 
    BBt_equals_L = np.allclose(B @ B.T, L)
    print(f"\n[Check] B B^T equals L: {BBt_equals_L}")

  
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (i,j,w) in edges:
            G.add_edge(i,j, weight=w)

        A_nx = nx.to_numpy_array(G, nodelist=range(n), weight="weight")
        L_nx = nx.laplacian_matrix(G, nodelist=range(n), weight="weight").toarray()


        print(f"\n[Validate vs networkx] A matches nx adjacency:"
              f"{np.allclose(A, A_nx)}")
        print(f"[Validate vs networkx] L matches Laplacian:"
              f"{np.allclose(L, L_nx)}")
    except ImportError:
        print("\n(networkx not installed - skipping external validation)")


    print("\n" + "=" * 70)
    print("Done.")
        

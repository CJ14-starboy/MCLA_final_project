import numpy as np
from task2_graph_setup import L



def check_symmetry(L):
    return np.allclose(L, L.T)



def check_positive_semidefinite(L, tolerance=1e-10):
    eigenvalues = np.linalg.eigvalsh(L)

    return np.all(eigenvalues >= -tolerance)


def check_all_ones_eigenvector(L):
    n = L.shape[0]

    ones = np.ones(n)

    result = L @ ones

    return result, np.allclose(result, np.zeros(n))


def get_eigenvalues(L):
    return np.linalg.eigvalsh(L)



def count_zero_eigenvalues(L, tolerance=1e-10):

    eigenvalues = np.linalg.eigvalsh(L)

    return np.sum(
        np.abs(eigenvalues) < tolerance
    )


def count_connected_components(n, edges):

    graph = [[] for _ in range(n)]

   
    for i, j, w in edges:
        graph[i].append(j)
        graph[j].append(i)

    visited = [False] * n

    components = 0

    def dfs(node):

        visited[node] = True

        for neighbour in graph[node]:

            if not visited[neighbour]:
                dfs(neighbour)

    for node in range(n):

        if not visited[node]:

            components += 1
            dfs(node)

    return components




def dirichlet_energy_from_edges(x, edges):

    energy = 0.0

    for i, j, w in edges:

        energy += w * (x[i] - x[j]) ** 2

    return energy


def dirichlet_energy_from_matrix(x, L):

    return x.T @ L @ x


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("QUESTION 3: PROPERTIES OF THE GRAPH LAPLACIAN")
    print("=" * 70)



    symmetric = check_symmetry(L)

    print("\n1. Symmetry")
    print("L is symmetric:", symmetric)



    eigenvalues = get_eigenvalues(L)

    print("\n2. Eigenvalues of L")
    print(eigenvalues)

    psd = check_positive_semidefinite(L)

    print("L is positive semidefinite:", psd)



    result, is_zero = check_all_ones_eigenvector(L)

    print("\n3. All-ones eigenvector")
    print("L @ 1 =")
    print(result)

    print("L1 = 0:", is_zero)




    x = np.arange(1, len(nodes) + 1, dtype=float)

    matrix_energy = dirichlet_energy_from_matrix(x, L)

    edge_energy = dirichlet_energy_from_edges(x, edges)

    print("\n4. Dirichlet-energy identity")

    print("x =", x)

    print("x^T L x =", matrix_energy)

    print(
        "Sum w_ij(x_i - x_j)^2 =",
        edge_energy
    )

    print(
        "Identity verified:",
        np.isclose(matrix_energy, edge_energy)
    )



    zero_count = count_zero_eigenvalues(L)

    print("\n5. Zero eigenvalues")

    print(
        "Number of zero eigenvalues:",
        zero_count
    )




    num_components = count_connected_components(
        len(nodes),
        edges
    )

    print("\n6. Connected components")

    print(
        "Number of connected components:",
        num_components
    )




    print("\n7. Final verification")

    print(
        "Zero eigenvalues = connected components:",
        zero_count == num_components
    )

    print("\n" + "=" * 70)
    print("QUESTION 3 COMPLETE")
    print("=" * 70)

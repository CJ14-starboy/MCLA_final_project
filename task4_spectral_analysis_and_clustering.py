import numpy as n_p
#Sub-Graph Computation  Chosen Subgraph: Tetteh Quarshie, Madina, Accra Central, Tema
A_sub = n_p.array([
    [0, 8, 20, 9],
    [8, 0, 0, 0],
    [20, 0, 0, 0],
    [9, 0, 0, 0]
])

degreeA_sub = n_p.sum(A_sub, axis=1)
D_sub = n_p.diag(degreeA_sub)

L_sub = n_p.subtract(D_sub, A_sub)

subvalues, subvectors = n_p.linalg.eigh(L_sub)


# Computation for Main Graph
from task2_graph_setup import A,D,L
l_values, l_vectors = n_p.linalg.eigh(L)

# Fiedler value identification
fiedler_value = l_values[1]
fiedler_vector = l_vectors[:, 1]




#Spectral clustering
from task1_network_setup import nodes, edges

cluster_1 = []
cluster_2 = []

for i, val in enumerate(fiedler_vector):
    towns = nodes[i]
    if val > 0:
        cluster_1.append(towns)
    else:
        cluster_2.append(towns)


#Displaying of results
print("-----Results-----")
print("-----------------")
print()
print("SUBGRAPH")
print("---------")
print("Eigenvalues:")
for i, val in enumerate(subvalues, start = 1):
    if i == 1:
        print(f"λ{i}: {int(val)}")
    else:
        print(f"λ{i}: {val}")
print("---------")
print("Eigenvectors:")
for i2, val2 in enumerate(subvectors, start=1):
    print(f"V{i2}: {val2}")
print()
print("Hand-derived eigenvalues: \n λ1 = 0\n λ2 = 52.57431242\n λ3 = 12.99446922\n λ4 = 8.43121835")
print()
print("---------")
print()
print("MAIN GRAPH")
print("---------")
print(f"λ2 (Fiedler value): {fiedler_value}")
print()
print(f"Fiedler vector: {fiedler_vector}")
print()
print("Cluster 1:", cluster_1)
print("Cluster 2:", cluster_2)



#Visualization of partition
import networkx as n_x
import matplotlib.pyplot as plot

VG = n_x.Graph()
for i, j, dist in edges:
    VG.add_edge(nodes[i], nodes[j], weight=dist)


color_by_town = {}
for i, val in enumerate(fiedler_vector):
    town = nodes[i]
    color_by_town[town] = "red" if val > 0 else "black"

town_colors_ordered = [color_by_town[town] for town in VG.nodes()]

pos = n_x.spring_layout(VG, seed=42)

fig, ax = plot.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")   # or any color you want
ax.set_facecolor("grey")

n_x.draw(VG, pos, ax=ax, with_labels=True, node_color=town_colors_ordered,
        node_size=2500, font_size=6, font_color="white", font_weight="bold")

edge_labels = n_x.get_edge_attributes(VG, 'weight')
n_x.draw_networkx_edge_labels(VG, pos, edge_labels=edge_labels, ax=ax)

plot.title("Spectral Clustering using Fiedler Vector sign pattern\n(red = Cluster A, black = Cluster B)")
plot.savefig("task4_spectral_clustering.png", dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())  # ensures saved image keeps the background color
plot.show()





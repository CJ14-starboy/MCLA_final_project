import numpy as np
import networkx as nx 

from task2_graph_setup import A,D,L
from task1_network_setup import nodes,edges


print('='*60)
print('TASK 7:GRAPH LAPLACIAN AND NETWORK ANALYSIS')
print('='*60)


print('\n EIGEN-DECOMPOSITION')
print('='*60)


eigenvalues,eigenvectors=np.linalg.eigh(L)
print('Eigenvalues')
for i, value in enumerate(eigenvalues):
    print(f'lambda_{i+1} = {value:6f}')
    
fiedler_value=eigenvalues[1]
print(f'\nfiedler value = {fiedler_value:.6f}')


print('\n DIRICHLET BOUNDARY-VALUE SOLVER')
print('='*60)

def solve_dirichlet(i,boundary_nodes,boundary_values):
     n=L.shape[0]
     free_nodes=[]
     for i in range(n):
         if i not in boundary_nodes:
             free_nodes.append(i)
             
     laplacian_ff = L[np.ix_(free_nodes,free_nodes)]
     laplacian_fb=L[np.ix_(free_nodes,boundary_nodes)]
     x_f=np.linalg.solve(laplacian_ff,-laplacian_fb @ boundary_values)
     x=np.zeros(n)
     x[boundary_nodes]=boundary_values
     x[free_nodes]=x_f
     return x,free_nodes
     
boundary_nodes=[3,7]
boundary_values=np.array([0.0,100.0])

x,free_nodes=solve_dirichlet(L,boundary_nodes,boundary_values)
print('Boundary nodes:',boundary_nodes)
print('Free nodes:',free_nodes)
print('\n Harmonic values:')
for i in range(len(nodes)):
    print(f'{nodes[i]:<20} = {x[i]:.2f}')


print('\n VERIFICATION OF DIRICHLET SOLUTION')
print('='*40)

Lx=L@x
residual=Lx[free_nodes]
for i,r in zip(free_nodes,residual):
    print(f'{nodes[i]:<20} = {r:.2e}')
print('\n Harmonic condition Lx =0:',np.allclose(residual,0))


print('\n DIFFUSION SIMULATION')
print('='*60)

def diffusion(L,boundary_nodes,boundary_values,steps=1000,dt=0.001):
    n=L.shape[0]
    x=np.zeros(n)
    x[boundary_nodes]=boundary_values
    for step in range(steps):    
         x=x-dt*(L@x)
         x[boundary_nodes]=boundary_values
    return x

diffusion_values = diffusion(L,boundary_nodes,boundary_values)
print('Values after diffusion:')
for i in range(len(nodes)):
        print(f'{nodes[i]:<20} ={diffusion_values[i]:.2f}')


print('\n VALIDATION USING NETWORKX')
print('='*40)
G=nx.Graph()
G.add_nodes_from(range(len(nodes)))
for i,j,weight in edges:
    G.add_edge(i,j,weight=weight)

L_networkx=nx.laplacian_matrix(G,weight='weight').toarray()
print('Laplacian agrees with Networkx:',np.allclose(L,L_networkx))


components =nx.number_connected_components(G)
print('Number of connected components:',components)


fiedler_networkx=nx.algebraic_connectivity(G,weight='weight')
print(f'Fiedler value ={fiedler_value:.6f}')
print(f'Networkx fiedler value ={fiedler_networkx:.6f}')
print('fiedler values agree:',np.isclose(fiedler_value,fiedler_networkx))



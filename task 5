
import numpy as np
import networkx as nx

from task1_network_setup import nodes,edges
from task2_graph_setup import L, A


#---------------------------------------------------------------
#TASK 5:HARMONIC FUNCTION ON THE NETWORK 
#---------------------------------------------------------------
n = len (nodes)
boundary_nodes=[3,7]
boundary_values=np.array([0.0,100.0])

free_nodes=[]
for i in range(n):
    if i not in boundary_nodes:
        free_nodes.append(i)
print('Boundary nodes:',boundary_nodes)
print('Free nodes:', free_nodes)



#---------------------------------------------------------------
#2.Solving the Dirichlet system 
#---------------------------------------------------------------
laplacian_ff = L[np.ix_(free_nodes,free_nodes)]
laplacian_fb=L[np.ix_(free_nodes,boundary_nodes)]

x_F=np.linalg.solve(laplacian_ff,-laplacian_fb @ boundary_values)


#---------------------------------------------------------------
#2.Combining boundary and free node values 
#---------------------------------------------------------------
x=np.zeros(n)
x[boundary_nodes]=boundary_values
x[free_nodes]=x_F


#---------------------------------------------------------------
#3.Harmonic Values 
#---------------------------------------------------------------
print('\nHarmonic values:')

for i in range (n):
        print (f'{nodes[i]}={x[i]:.2f}')

#---------------------------------------------------------------
#4.Verification of harmonic for Lx=0 at the free nodes
#---------------------------------------------------------------
    
Lx = L @ x
residual =Lx[free_nodes]
print('\nResiduals at free nodes:',residual)
for i, r in zip(free_nodes,residual):
    print(f'{nodes[i]}={r:.2e}')

print('\nHarmonics:',np.allclose(residual,0))
        

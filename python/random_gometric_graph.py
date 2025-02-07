import networkx as nx
from collections import namedtuple
import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from typing import List, Union, Tuple

Edge = namedtuple("Edge", ["i", "j", "weight"])

def nx_to_mac(G: nx.Graph) -> List[Edge]:

    edges = []
    for nxedge in G.edges():
        edge = Edge(nxedge[0], nxedge[1], G[nxedge[0]][nxedge[1]]['weight'])
        edges.append(edge)
    return edges

# Step 1: Generate a random geometric graph
n = 100  # number of nodes
radius = 0.2  # radius for connecting nodes
G = nx.random_geometric_graph(n, radius)
spanning_tree = nx.minimum_spanning_tree(G)
loop_graph = nx.difference(G, spanning_tree)

# Add a chain
for i in range(n-1):
    if G.has_edge(i+1, i):
        G.remove_edge(i+1, i)
    if not G.has_edge(i, i+1):
        G.add_edge(i, i+1)

print(G)
nx.draw(G)
plt.title("Original Graph")
plt.show()

# Step 2: Add weights to the edges
sigma = 1.0  # standard deviation for the Gaussian distribution
for (u, v) in G.edges():
    W = np.random.normal(0, sigma)
    weight = np.random.uniform(0, W)
    G[u][v]['weight'] = weight

Edge = nx_to_mac(G)


# Step 3: Compute the Laplacian matrix
L = nx.laplacian_matrix(G, weight='weight').toarray()

# Convert the Laplacian matrix to float type
L = L.astype(float)

# Step 4: Calculate the Laplacian matrix norm
laplacian_norm = norm(L)
print(f"Laplacian matrix norm: {laplacian_norm}")

# Step 5: Compute the Fiedler value (second smallest eigenvalue of the Laplacian matrix)
try:
    eigenvalues, _ = eigsh(L, k=2, which='SM', maxiter=5000)
    fiedler_value = eigenvalues[1]
    print(f"Fiedler value: {fiedler_value}")
except Exception as e:
    print(f"An error occurred: {e}")

# Plot the generated graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, node_size=10, with_labels=False, edge_color='b', width=[G[u][v]['weight'] for u, v in G.edges()])
plt.show()

pos = nx.get_node_attributes(spanning_tree, 'pos')
nx.draw(spanning_tree, pos, node_size=10, with_labels=False, edge_color='b', width=[G[u][v]['weight'] for u, v in G.edges()])
plt.show()

pos = nx.get_node_attributes(loop_graph, 'pos')
nx.draw(loop_graph, pos, node_size=10, with_labels=False, edge_color='b', width=[G[u][v]['weight'] for u, v in G.edges()])
plt.show()

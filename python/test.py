import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to generate a block symmetric Kronecker matrix
def generate_kron_matrix(adj_matrix, sigma):
    S1 = np.array([[0,  0,  0], [0,  0, -1], [0,  1,  0]])
    S2 = np.array([[0,  0,  1], [0,  0,  0], [-1, 0,  0]])
    S3 = np.array([[0, -1,  0], [1,  0,  0], [0,  0,  0]])

    n = adj_matrix.shape[0]
    D_matrices = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if adj_matrix[i, j] != 0:
                phi = np.random.normal(0, sigma, 3)
                D = phi[0] * S1 + phi[1] * S2 + phi[2] * S3
                D_matrices[i][j] = D
                D_matrices[j][i] = D
            else:
                D_matrices[i][j] = np.zeros((3, 3))
                D_matrices[j][i] = np.zeros((3, 3))

    kron_matrix = np.zeros((3 * n, 3 * n))

    for i in range(n):
        for j in range(n):
            kron_matrix[3*i:3*(i+1), 3*j:3*(j+1)] = D_matrices[i][j]

    return kron_matrix

# Function to generate a sum of B matrices and compute operator norm
def generate_symmetric_matrix_sum(adj_matrix, sigma):
    S1 = np.array([[0,  0,  0], [0,  0, -1], [0,  1,  0]])
    S2 = np.array([[0,  0,  1], [0,  0,  0], [-1, 0,  0]])
    S3 = np.array([[0, -1,  0], [1,  0,  0], [0,  0,  0]])

    n = adj_matrix.shape[0]
    sum_B = np.zeros((3 * n, 3 * n))

    edge_list = np.array(np.triu(adj_matrix).nonzero()).T

    for i, j in edge_list:
        phi = np.random.normal(0, sigma, 3)
        B_block = S1 + S2 + S3
        B = np.zeros((3 * n, 3 * n))
        B[3*i:3*(i+1), 3*j:3*(j+1)] = B_block
        B[3*j:3*(j+1), 3*i:3*(i+1)] = B_block
        sum_B += B

    sum_B_BT = sum_B @ sum_B.T
    operator_norm = np.linalg.norm(sum_B_BT, ord=2)
    return operator_norm

# Function to compute the largest eigenvalue of a matrix
def compute_largest_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

# Function to compute the theoretical bound
def theoretical_bound(t, num_nodes, vZ):
    return (num_nodes + num_nodes) * np.exp(-t**2 / (2 * vZ))

# Parameters
num_nodes = 50  # Number of nodes
sigma_values = [1.0]
num_trials = 1000  # Number of trials
W_max = 10**3  # Maximum weight

# Run experiments and collect results
results = {sigma: [] for sigma in sigma_values}

for sigma in sigma_values:
    for _ in range(num_trials):
        r = 1.25 * np.sqrt(np.log(num_nodes) / (np.pi * num_nodes))
        G = nx.random_geometric_graph(num_nodes, r)
        adj_matrix = nx.adjacency_matrix(G).todense()

        V_z = generate_symmetric_matrix_sum(adj_matrix, sigma)
        kron_matrix = generate_kron_matrix(adj_matrix, sigma)
        largest_eigenvalue = compute_largest_eigenvalue(kron_matrix)
        results[sigma].append((largest_eigenvalue, V_z))

# Plot the empirical distribution of the largest eigenvalues
fig, ax = plt.subplots(figsize=(10, 6))
for sigma, values in results.items():
    eigenvalues = [v[0] for v in values]
    V_z = values[0][1]  # Take V_z from the first trial as representative
    t_values = np.linspace(0, 30, 1000)
    bound_values = theoretical_bound(t_values, num_nodes, V_z)
    bound_values = bound_values / 100
    # Compute empirical probability
    empirical_probs = [np.mean(np.array(eigenvalues) >= t) for t in t_values]
    empirical_probs = np.array(empirical_probs) / np.max(empirical_probs)  # Normalize to 0-1

    ax.plot(t_values, bound_values, label=f'Theoretical Bound (k={k})', linestyle='dashed')
    ax.plot(t_values, empirical_probs, label=f'Empirical Probability (k={k})', linestyle='solid')

ax.set_xlabel('Largest Eigenvalue')
ax.set_ylabel('Probability')
ax.set_title('Theoretical Bound vs Empirical Probability')
ax.legend()
ax.grid(True)

plt.show()

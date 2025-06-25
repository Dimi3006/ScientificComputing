import numpy as np
import matplotlib.pyplot as plt

def build_laplacian_matrix(N):
    h = 1.0 / (N + 1)
    
    # 1D Laplacian matrix
    laplacian_1d = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            laplacian_1d[i, i - 1] = 1
        laplacian_1d[i, i] = -2
        if i < N - 1:
            laplacian_1d[i, i + 1] = 1
    
    identity = np.eye(N)
    
    # 2D Laplacian via Kronecker sum
    laplacian_2d = np.kron(identity, laplacian_1d) + np.kron(laplacian_1d, identity)
    return laplacian_2d/h**2

def harmonic_potential(x, y):
    """Harmonic potential V(x, y) = 0.5 * (x^2 + y^2) as lambda function."""
    return (x - 0.5)**2 + (y - 0.5)**2

def build_potential_matrix(N, potential=harmonic_potential):
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij') 
    
    V_values = potential(X, Y).reshape(N * N) 
    potential_matrix = np.diag(V_values)  # Diagonalmatrix mit V auf der Diagonalen
    return potential_matrix

def system_matrix(N):
    """Construct the system matrix for the Schrödinger equation."""
    A = build_laplacian_matrix(N)
    V = build_potential_matrix(N)
    return -0.5*A + V

def shifted_inverse_power_method(A, mu, solver=np.linalg.solve, tol=1e-6, max_iter=1000):
    """
    Shifted inverse power method to find eigenvector for eigenvalue closest to mu.
    solver: Funktion, die (Matrix, Vektor) nimmt und Lösung zurückgibt.
    """
    n = A.shape[0]
    x = np.ones(n) # Change initial guess (random?)

    M = A - mu * np.eye(n)
    for _ in range(max_iter):
        y = solver(M, x)
        if np.linalg.norm(x - y) < tol:
            break
        x = y

    lambda_x = np.dot(x, A @ x) / np.dot(x, x)
    return x, lambda_x


def cg(A,b):
    return 0

if __name__ == "__main__":
    N = 25
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    print("u (2D grid):")
    print(u)

    u_flat = u.reshape(N * N)
    print("\nu flattened:")
    print(u_flat)
    L = build_laplacian_matrix(N)

    print("\nLaplacian matrix A:")
    print(L)

    # Apply the Laplacian
    Lu = L @ u_flat
    print("\nA @ u (discrete Laplacian applied):")
    print(Lu.reshape(N, N))

    # Test potential matrix
    V = build_potential_matrix(N)
    print("\nPotential matrix V:") 
    print(V)

    # Construct the system matrix
    A = system_matrix(N)
    print("\nSystem matrix A:")
    print(A)

    v, lambda_v = shifted_inverse_power_method(A, mu=100)
    print("\nEigenvector v closest to mu=1:")
    print(v)
    print("\nEigenvalue lambda_v closest to mu=1:")
    print(lambda_v)

    # Plot the eigenvector over the grid
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij')
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, v.reshape(N, N), shading='auto', cmap='viridis')
    plt.colorbar(label='Eigenvector value')
    plt.title('Eigenvector closest to mu')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # Plot the potential
    plt.figure(figsize=(8, 6)) 
    plt.pcolormesh(X, Y, harmonic_potential(X, Y).reshape(N, N), shading='auto', cmap='plasma')
    plt.colorbar(label='Potential V(x, y)')
    plt.title('Harmonic Potential V(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



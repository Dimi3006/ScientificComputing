import numpy as np
import matplotlib.pyplot as plt

# TODO: use convolve / maybe even FFT for operator
def laplace(u_flat):
    """Compute the discrete Laplacian of a 2D grid."""
    N = int(np.sqrt(u_flat.shape[0]))
    h = 1.0 / (N + 1)
    laplacian_u = np.zeros_like(u_flat)

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            laplacian_u[idx] = (
                -4 * u_flat[idx] +
                (u_flat[idx - 1] if j > 0 else 0) +  # left neighbor
                (u_flat[idx + 1] if j < N - 1 else 0) +  # right neighbor
                (u_flat[idx - N] if i > 0 else 0) +  # upper neighbor
                (u_flat[idx + N] if i < N - 1 else 0)  # lower neighbor
            ) / h**2

    return laplacian_u

def multiply_potential(u_flat, potential):
    """Multiply the potential function with the flattened grid."""
    N = int(np.sqrt(u_flat.shape[0]))
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij')
    
    V_values = potential(X, Y).reshape(N * N)  # Flatten the potential matrix
    return u_flat * V_values  # Element-wise multiplication

# def build_laplacian_matrix(N):
    # h = 1.0 / (N + 1)
    
    # # 1D Laplacian matrix
    # laplacian_1d = np.zeros((N, N))
    # for i in range(N):
    #     if i > 0:
    #         laplacian_1d[i, i - 1] = 1
    #     laplacian_1d[i, i] = -2
    #     if i < N - 1:
    #         laplacian_1d[i, i + 1] = 1
    # print(laplacian_1d.shape)
    # identity = np.eye(N)
    
    # # 2D Laplacian via Kronecker sum
    # laplacian_2d = np.kron(identity, laplacian_1d) + np.kron(laplacian_1d, identity)
    # print(laplacian_2d.shape)
    # return laplacian_2d/h**2

def harmonic_potential(x, y):
    """Harmonic potential V(x, y) = 0.5 * (x^2 + y^2) as lambda function."""
    return (x - 0.5)**2 + (y - 0.5)**2

# def build_potential_matrix(N, potential=harmonic_potential):
#     h = 1.0 / (N + 1)
#     x_list = np.linspace(h, 1 - h, N)
#     X, Y = np.meshgrid(x_list, x_list, indexing='ij') 
    
#     V_values = potential(X, Y).reshape(N * N) 
#     potential_matrix = np.diag(V_values)  # Diagonalmatrix mit V auf der Diagonalen
#     return potential_matrix

# def system_matrix(N):
#     """Construct the system matrix for the Schrödinger equation."""
#     A = build_laplacian_matrix(N)
#     V = build_potential_matrix(N)
#     return -0.5*A + V

def shifted_inverse_power_method(operator, sigma, solver, N, tol=1e-6, max_iter=1000):
    """
    Shifted inverse power method to find eigenvector for eigenvalue closest to mu.
    solver: Funktion, die (Matrix, Vektor) nimmt und Lösung zurückgibt.
    """
    x = np.ones(N**2) # Change initial guess (random?)

    sigma_operator = lambda u: operator(u) - sigma * u
    
    for _ in range(max_iter):
        y = solver(sigma_operator, x)
        if np.linalg.norm(x - y) < tol:
            break
        x = y

    lambda_x = np.dot(x, operator(x)) / np.dot(x, x)
    return x, lambda_x


def cg(A,b):
    return 0

if __name__ == "__main__":
    N = 100
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    # print("u (2D grid):")
    # print(u)

    u_flat = u.reshape(N * N)
    # print("\nu flattened:")
    # print(u_flat)
    L = build_laplacian_matrix(N)

    # print("\nLaplacian matrix A:")
    # print(L)

    # Apply the Laplacian
    Lu = L @ u_flat
    # print("\nA @ u (discrete Laplacian applied):")
    # print(Lu.reshape(N, N))

    # Test potential matrix
    V = build_potential_matrix(N)
    # print("\nPotential matrix V:") 
    # print(V)

    # Construct the system matrix
    A = system_matrix(N)
    # print("\nSystem matrix A:")
    # print(A)

    mu = 0
    v, lambda_v = shifted_inverse_power_method(A, mu)
    # print(f"\nEigenvector v closest to mu={mu}:")
    # print(v)
    # print(f"\nEigenvalue lambda_v closest to mu={mu}:")
    # print(lambda_v)

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
    # plt.show()
    # Plot the potential
    plt.figure(figsize=(8, 6)) 
    plt.pcolormesh(X, Y, harmonic_potential(X, Y).reshape(N, N), shading='auto', cmap='plasma')
    plt.colorbar(label='Potential V(x, y)')
    plt.title('Harmonic Potential V(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()



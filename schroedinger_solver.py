import numpy as np
import matplotlib.pyplot as plt


# TODO: use convolve / maybe even FFT for operator
def laplace(u_flat):
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
    print(laplacian_1d.shape)
    identity = np.eye(N)
    
    # 2D Laplacian via Kronecker sum
    laplacian_2d = np.kron(identity, laplacian_1d) + np.kron(laplacian_1d, identity)
    print(laplacian_2d.shape)
    return laplacian_2d/h**2

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


def cg(operator, b, tol=1e-10, max_iter=1000):
    """Conjugate Gradient method to solve Ax = b for a linear operator A.
        operator: Function that applies the linear operator A to a vector.
        b: Right-hand side vector.
    """
    x = np.zeros_like(b)
    r = b - operator(x)
    p = r.copy()
    alpha_old = np.dot(r, r)

    # TODO: include max_iter
    for _ in range(b.shape[0]):
        Ap = operator(p)
        lambd = alpha_old / np.dot(p, Ap)
        x += lambd * p
        r -= lambd * Ap
        alpha_new = np.dot(r, r)
        if alpha_new < tol:
            break
        p = r + (alpha_new / alpha_old) * p
        alpha_old = alpha_new

    return x

def pcg(operator, b, preconditioner, tol=1e-10, max_iter=1000):
    """Preconditioned Conjugate Gradient method to solve Ax = b for a linear operator A.
        operator: Function that applies the linear operator A to a vector.
        b: Right-hand side vector.
    """
    x = np.zeros_like(b)
    r = b - operator(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    for _ in range(max_iter):
        Ap = operator(p)
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) < tol:
            break
        z = preconditioner(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x


def dimile_old():
    N = 25
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    # print("u (2D grid):")
    # print(u)

    u_flat = u.reshape(N * N)
    
    sigma = 8

    potential_operator = lambda u: multiply_potential(u, harmonic_potential)
    summed_operator = lambda u: potential_operator(u) - 0.5 * laplace(u)

    v, lambda_v = shifted_inverse_power_method(summed_operator, sigma, cg, N)
    # print(f"\nEigenvector v closest to mu={mu}:")
    # print(v)
    print(f"\nEigenvalue lambda_v closest to sigma={sigma}:")
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

def test_cg():
    pass

if __name__ == "__main__":
    A = build_laplacian_matrix(3)
    print("Laplacian matrix A:")    
    print(A)
    dimile_old()
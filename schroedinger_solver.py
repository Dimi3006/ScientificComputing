import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time


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

def sparse_laplacian_matrix(N):
    """Construct a sparse Laplacian matrix for a 2D grid (Dirichlet boundary)."""
    h = 1.0 / (N + 1)
    size = N * N

    # Index offsets
    main_diag = -4 * np.ones(size)
    right_diag = np.ones(size - 1)
    left_diag = np.ones(size - 1)
    up_diag = np.ones(size - N)
    down_diag = np.ones(size - N)

    # Zero out entries that cross row boundaries
    for i in range(1, N):
        left_diag[i * N - 1] = 0
        right_diag[i * N - 1] = 0

    diagonals = [up_diag, left_diag, main_diag, right_diag, down_diag]
    offsets = [-N, -1, 0, 1, N]

    laplacian = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    return laplacian / h**2

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

def sparse_potential_matrix(N, potential=harmonic_potential):
    """Construct a sparse potential matrix for the 2D grid."""
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij')
    
    V_values = potential(X, Y).reshape(N * N)
    return sp.diags(V_values, format='csr')  # Diagonal matrix with V on the diagonal

def system_matrix(N):
    """Construct the system matrix for the Schrödinger equation."""
    A = build_laplacian_matrix(N)
    V = build_potential_matrix(N)
    return -0.5*A + V

def sparse_system_matrix(N):
    """Construct the sparse system matrix for the Schrödinger equation."""
    A = sparse_laplacian_matrix(N)
    V = sparse_potential_matrix(N)
    return -0.5 * A + V  # Sparse matrix with -0.5 * Laplacian + Potential on the diagonal

# def shifted_inverse_power_method(operator, sigma, solver, N, tol=1e-6, max_iter=1000):
#     """
#     Shifted inverse power method to find eigenvector for eigenvalue closest to mu.
#     solver: Funktion, die (Matrix, Vektor) nimmt und Lösung zurückgibt.
#     """
#     x = np.ones(N**2) # Change initial guess (random?)

#     sigma_operator = lambda u: operator(u) - sigma * u
    
#     for _ in range(max_iter):
#         y = solver(sigma_operator, x)
#         if np.linalg.norm(x - y) < tol:
#             break
#         x = y

#     lambda_x = np.dot(x, operator(x)) / np.dot(x, x)
#     return x, lambda_x

def shifted_inverse_power_method(A, sigma, solver, precond=None, tol=1e-6, max_iter=1000):
    """
    Shifted inverse power method to find eigenvector for eigenvalue closest to mu.
    """
    N = int(np.sqrt(A.shape[0]))
    x = np.ones(N**2) # Change initial guess (random?)

    shifted_matrix = A - sigma * sp.eye(N**2, format='csr')
    # print(shifted_matrix.toarray())
    
    for _ in range(max_iter):
        y = solver(shifted_matrix, x, precond)
        if np.linalg.norm(x - y) < tol:
            break
        x = y
    
    lambda_x = np.dot(x, A @ x) / np.dot(x, x)
    return x, lambda_x


# def cg(operator, b, tol=1e-10, max_iter=1000):
#     """Conjugate Gradient method to solve Ax = b for a linear operator A.
#         operator: Function that applies the linear operator A to a vector.
#         b: Right-hand side vector.
#     """
#     x = np.zeros_like(b)
#     r = b - operator(x)
#     p = r.copy()
#     alpha_old = np.dot(r, r)

#     # TODO: include max_iter
#     for _ in range(b.shape[0]):
#         Ap = operator(p)
#         lambd = alpha_old / np.dot(p, Ap)
#         x += lambd * p
#         r -= lambd * Ap
#         alpha_new = np.dot(r, r)
#         if alpha_new < tol:
#             break
#         p = r + (alpha_new / alpha_old) * p
#         alpha_old = alpha_new

#     return x

def cg(A, b, precond=None, tol=1e-10, max_iter=1000):
    """Conjugate Gradient method to solve Ax = b for a linear operator A.
        operator: Function that applies the linear operator A to a vector.
        b: Right-hand side vector.
    """
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    alpha_old = np.dot(r, r)

    # TODO: include max_iter
    iterations = 0
    for _ in range(b.shape[0]):
        Ap = A @ p
        lambd = alpha_old / np.dot(p, Ap)
        x += lambd * p
        r -= lambd * Ap
        alpha_new = np.dot(r, r)
        if alpha_new < tol:
            break
        p = r + (alpha_new / alpha_old) * p
        alpha_old = alpha_new
        iterations += 1
    print(f"CG iterations: {iterations}")
    return x

# def pcg(operator, b, preconditioner, tol=1e-10, max_iter=1000):
#     """Preconditioned Conjugate Gradient method to solve Ax = b for a linear operator A.
#         operator: Function that applies the linear operator A to a vector.
#         b: Right-hand side vector.
#     """
#     x = np.zeros_like(b)
#     r = b - operator(x)
#     z = preconditioner(r)
#     p = z.copy()
#     rz_old = np.dot(r, z)

#     for _ in range(max_iter):
#         Ap = operator(p)
#         alpha = rz_old / np.dot(p, Ap)
#         x += alpha * p
#         r -= alpha * Ap
#         if np.linalg.norm(r) < tol:
#             break
#         z = preconditioner(r)
#         rz_new = np.dot(r, z)
#         beta = rz_new / rz_old
#         p = z + beta * p
#         rz_old = rz_new

#     return x

def pcg(A, b, preconditioner, tol=1e-10, max_iter=1000):
    """Preconditioned Conjugate Gradient method to solve Ax = b for a linear operator A.
        operator: Function that applies the linear operator A to a vector.
        b: Right-hand side vector.
    """
    x = np.zeros_like(b)
    r = b - A @ x
    z = preconditioner(r)
    p = z.copy()
    rz_old = np.dot(r, z)
    iterations = 0

    for _ in range(max_iter):
        Ap = A @ p
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
        iterations += 1
    print(f"PCG iterations: {iterations}")
    return x

# Return A, L, U matrices for the sparse matrix A
def split_matrix(A):
    """Split the sparse matrix A into its lower, upper, and diagonal parts with L and U having zero diagonals."""
    D = sp.diags(A.diagonal(), format='csr')  # Diagonal part
    L = sp.tril(A, k=-1, format='csr')        # Strictly lower triangular part (zero diagonal)
    U = sp.triu(A, k=1, format='csr')         # Strictly upper triangular part (zero diagonal)
    return L, U, D
    
def jacobi_preconditioner(A):
    """Jacobi preconditioner for the system matrix."""
    # D is the diagonal part of the matrix A
    D = A.diagonal()
    if np.any(D == 0):
        raise ValueError("Jacobi preconditioner: zero on diagonal.")

    def precondition(r):
        return r / D
    return precondition

def sgs_preconditioner(L, U, D):
    """Symmetric Gauss-Seidel preconditioner for the system matrix."""
    LD = (L + D).tocsr()
    UD = (U + D).tocsr()
    def preconditioner(r):
        z3 = sp.linalg.spsolve_triangular(LD, r, lower=True, unit_diagonal=False)
        z2 = D @ z3
        z = sp.linalg.spsolve_triangular(UD, z2, lower=False, unit_diagonal=False)
        return z
    return preconditioner

def ssor_preconditioner(L, D, U, omega=1.0):
    """Symmetric Successive Over-Relaxation (SSOR) preconditioner for the system matrix."""
    LD = (L + D / omega).tocsr()
    UD = (U + D / omega).tocsr()
    
    def preconditioner(r):
        z3 = sp.linalg.spsolve_triangular(1/(2-omega)*LD, r, lower=True, unit_diagonal=False)
        z2 = (D/omega) @ z3
        z = sp.linalg.spsolve_triangular(UD, z2, lower=False, unit_diagonal=False)
        return z
    return preconditioner

def dimile_old():
    N =40
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

def dimile_new():
    N = 64
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    # print("u (2D grid):")
    # print(u)

    u_flat = u.reshape(N * N)
    
    sigma = 0
    # potential_operator = lambda u: multiply_potential(u, harmonic_potential)
    # summed_operator = lambda u: potential_operator(u) - 0.5 * laplace(u)
    A = sparse_system_matrix(N)
    L, U, D = split_matrix(A)
    # print('A')
    # print(A.toarray())
    # print('L')
    # print(L.toarray())
    # print('U')
    # print(U.toarray())
    # print('D')
    # print(D.toarray())
    # print('L + U + D')
    # print((L+U+D).toarray())
    omega = 1.8
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ssor_preconditioner(L, D, U, omega))
    # v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, sgs_preconditioner(L, U, D))
    # v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, jacobi_preconditioner(A))
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
    # A = build_laplacian_matrix(3)
    # print("Laplacian matrix A:")    
    # print(A)
    # V1 = build_potential_matrix(3)
    # print("Potential matrix V1:")
    # print(V1)
    B = sparse_laplacian_matrix(3)
    # print("Sparse Laplacian matrix B:")
    # print(B.toarray())  # Convert sparse matrix to dense for printing
    V2 = sparse_potential_matrix(3)
    # print("Sparse Potential matrix V2:")
    # print(V2.toarray())  # Convert sparse matrix to dense for printing
    # true = -0.5 * B + V2
    # print(true.toarray())  # Convert sparse matrix to dense for printing
    # our = sparse_system_matrix(3)
    # print(our.toarray())  # Convert sparse matrix to dense for printing

    # dimile_old()
    dimile_new()
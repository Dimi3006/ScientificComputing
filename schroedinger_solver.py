import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from scipy.sparse.linalg import spilu

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

def sparse_potential_matrix(N, potential=harmonic_potential):
    """Construct a sparse potential matrix for the 2D grid."""
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij')
    
    V_values = potential(X, Y).reshape(N * N)
    return sp.diags(V_values, format='csr')  # Diagonal matrix with V on the diagonal

def sparse_system_matrix(N, potential=harmonic_potential):
    """Construct the sparse system matrix for the Schrödinger equation."""
    L = sparse_laplacian_matrix(N)
    V = sparse_potential_matrix(N, potential)
    A = (-0.5 * L + V).tocsr()  # Sparse matrix with -0.5 * Laplacian + Potential on the diagonal
    return A

def shifted_inverse_power_method(A, sigma, solver, precond=None, tol=1e-8, max_iter=1000):
    """
    Shifted inverse power method to find eigenvector for eigenvalue closest to mu.
    """
    N = int(np.sqrt(A.shape[0]))
    x = np.ones(N**2)
    x = x / np.linalg.norm(x)

    shifted_matrix = A - sigma * sp.eye(N**2, format='csr')
    
    for _ in range(max_iter):
        y = solver(shifted_matrix, x, precond)
        y = y / np.linalg.norm(y)
        if np.linalg.norm(x - y) < tol:
            break
        x = y

    lambda_x = np.dot(x, A @ x) / np.dot(x, x)
    return x, lambda_x

def cg(A, b, precond=None, tol=1e-8, max_iter=1000):
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
    res = [np.linalg.norm(r)]
    for _ in range(max_iter):
        Ap = A @ p
        lambd = alpha_old / np.dot(p, Ap)
        x += lambd * p
        r -= lambd * Ap
        res.append(np.linalg.norm(r))
        alpha_new = np.dot(r, r)
        if np.sqrt(alpha_new) < tol:
            break
        p = r + (alpha_new / alpha_old) * p
        alpha_old = alpha_new
        iterations += 1
    print(f"CG iterations: {iterations}")
    # plt.plot(res)
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel('Residual Norm')
    # plt.title('Convergence of CG Method')
    # plt.grid()
    # plt.show()
    return x

def pcg(A, b, preconditioner, tol=1e-8, max_iter=1000):
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
    res = [np.linalg.norm(r)]
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        res.append(np.linalg.norm(r))
        if np.linalg.norm(r) < tol:
            break
        z = preconditioner(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        iterations += 1
    # print(f"PCG iterations: {iterations}")
    # plt.plot(res)
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel('Residual Norm')
    # plt.title('Convergence of PCG Method')
    # plt.grid()
    # plt.show()
    # exit()
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

# def sgs_preconditioner(L, U, D):
#     """Symmetric Gauss-Seidel preconditioner for the system matrix."""
#     LD = (L + D).tocsr()
#     UD = (U + D).tocsr()
#     def preconditioner(r):
#         z3 = sp.linalg.spsolve_triangular(LD, r, lower=True, unit_diagonal=False)
#         z2 = D @ z3
#         z = sp.linalg.spsolve_triangular(UD, z2, lower=False, unit_diagonal=False)
#         return z
#     return preconditioner

# def ssor_preconditioner(L, D, U, omega=1.0):
#     """Symmetric Successive Over-Relaxation (SSOR) preconditioner for the system matrix."""
#     LD = (L + D / omega).tocsr()
#     UD = (U + D / omega).tocsr()
    
#     def preconditioner(r):
#         z3 = sp.linalg.spsolve_triangular(1/(2-omega)*LD, r, lower=True, unit_diagonal=False)
#         z2 = (D/omega) @ z3
#         z = sp.linalg.spsolve_triangular(UD, z2, lower=False, unit_diagonal=False)
#         return z
#     return preconditioner

def sgs_preconditioner(L, U, D):
    """Symmetric Gauss-Seidel preconditioner for the system matrix."""
    LD = (L + D).tocsr()
    UD = (U + D).tocsr()
    def preconditioner(r):
        # Use spsolve instead of spsolve_triangular for possible speedup
        z3 = sp.linalg.spsolve(LD, r)
        z2 = D @ z3
        z = sp.linalg.spsolve(UD, z2)
        return z
    return preconditioner

def ssor_preconditioner(L, D, U, omega=1.8):
    """Symmetric Successive Over-Relaxation (SSOR) preconditioner for the system matrix."""
    LD = (L + D / omega).tocsr()
    UD = (U + D / omega).tocsr()
    
    def preconditioner(r):
        # Use spsolve instead of spsolve_triangular for possible speedup
        z3 = sp.linalg.spsolve((1/(2-omega))*LD, r)
        z2 = (D/omega) @ z3
        z = sp.linalg.spsolve(UD, z2)
        return z
    return preconditioner

def ic0_preconditioner(A):
    """
    Incomplete Cholesky factorization with zero fill-in (IC(0)) preconditioner for symmetric positive definite sparse matrix A.
    Returns a function that applies the preconditioner to a vector.
    """
    # Ensure A is in CSR format and symmetric
    A = A.tocsr()
    n = A.shape[0]
    L = sp.lil_matrix((n, n))
    for i in range(n):
        for j in A.indices[A.indptr[i]:A.indptr[i+1]]:
            if j > i:
                continue  # Only fill lower triangle
            s = A[i, j]
            for k in range(A.indptr[i], A.indptr[i+1]):
                col_k = A.indices[k]
                if col_k < j:
                    s -= L[i, col_k] * L[j, col_k]
            if i == j:
                if s <= 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite for IC(0)")
                L[i, j] = np.sqrt(s)
            else:
                if L[j, j] == 0:
                    L[i, j] = 0
                else:
                    L[i, j] = s / L[j, j]
    L = L.tocsr()
    def precondition(r):
        # Solve L y = r
        y = sp.linalg.spsolve(L, r)
        # Solve L^T x = y
        x = sp.linalg.spsolve(L.transpose().tocsr(), y)
        return x
    return precondition


def analytical_laplacian_eigenvalue(p, q, N):
    """
    Compute the eigenvalue λ_{p,q} of the 2D Laplacian with Dirichlet boundary conditions.

    Parameters:
        p, q : int
            Mode indices (1-based, from 1 to N)
        N : int
            Grid size

    Returns:
        float
            Eigenvalue λ_{p,q}
    """
    h = 1.0 / (N + 1)
    return -(1 / h**2) * (np.cos(p * np.pi * h) + np.cos(q * np.pi * h) - 2)

def dimile_new():
    N = 45
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    # print("u (2D grid):")
    # print(u)

    u_flat = u.reshape(N * N)
    
    sigma = 0

    A = sparse_system_matrix(N, lambda x, y: 0*x*y)
    L, U, D = split_matrix(A)

    omega = 1.8
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ic0_preconditioner(A))
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with IC(0) preconditioner: {elapsed_time:.4f} seconds")
    # SSOR preconditioner
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ssor_preconditioner(L, D, U, omega))
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with SSOR preconditioner: {elapsed_time:.4f} seconds")

    # SGS preconditioner
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, sgs_preconditioner(L, U, D))
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with SGS preconditioner: {elapsed_time:.4f} seconds")

    # Jacobi preconditioner
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, jacobi_preconditioner(A))
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with Jacobi preconditioner: {elapsed_time:.4f} seconds")

    # No preconditioner (plain CG)
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, cg, None)
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with no preconditioner (plain CG): {elapsed_time:.4f} seconds")
    # print(f"\nEigenvector v closest to mu={mu}:")
    # print(v)
    print(f"\nEigenvalue lambda_v closest to sigma={sigma}:")
    print(lambda_v)
    print(f"Analytical eigenvalue: {analytical_laplacian_eigenvalue(1, 1, N)}")
    print(f"difference: {lambda_v - analytical_laplacian_eigenvalue(1, 1, N)}")

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
    # B = sparse_laplacian_matrix(3)
    # print("Sparse Laplacian matrix B:")
    # print(B.toarray())  # Convert sparse matrix to dense for printing
    # V2 = sparse_potential_matrix(3)
    # print("Sparse Potential matrix V2:")
    # print(V2.toarray())  # Convert sparse matrix to dense for printing
    # true = -0.5 * B + V2
    # print(true.toarray())  # Convert sparse matrix to dense for printing
    # our = sparse_system_matrix(3)
    # print(our.toarray())  # Convert sparse matrix to dense for printing

    # dimile_old()
    dimile_new()
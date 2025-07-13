import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import eigsh, eigs
import os

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

    diagonals = [down_diag, left_diag, main_diag, right_diag, up_diag]
    offsets = [-N, -1, 0, 1, N]

    laplacian = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    return laplacian / h**2

def harmonic_potential(x, y):
    """Harmonic potential V(x, y) = 0.5 * (x^2 + y^2) as lambda function."""
    return (x - 0.5)**2 + (y - 0.5)**2

def step_potential(x, y):
    """Step potential V(x, y) = 1 if x > 0.5 and y > 0.5, else 0."""
    return np.where((x > 0.5) & (y > 0.5), 2000, 0.0)

def double_well_potential(x, y):
    """Double well potential energy surface in 2D, centered at (0.5, 0.5), horizontal wells."""
    return 500 * ((x - 0.5)**2 - 0.25)**2 + 1000*(y - 0.5)**2


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

def shifted_inverse_power_method(A, sigma, solver, precond=None, tol=1e-8, max_iter=1000, start=None):
    """
    Shifted inverse power method to find eigenvector for eigenvalue closest to mu.
    """
    N = int(np.sqrt(A.shape[0]))
    if start is None:
        x = np.random.rand(N**2)  # Random initial guess
        # x = np.ones(N**2)  # Use a constant initial guess
    else:
        x = start.copy()
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
    # print(f"CG iterations: {iterations}")
    N = int(np.sqrt(A.shape[0]))
    with open(f"data/cg_{N}_residuals.txt", "a") as f:
        f.write(f"res: {res}\n")
    
    with open(f"data/cg_{N}_iterations.txt", "a") as f:
        f.write(f"iterations: {iterations}\n")

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
    
    # with open(f"data/pcg_{preconditioner.__name__}_residuals.txt", "a") as f:
    #     f.write(f"res: {res}\n")
    
    # with open(f"data/pcg_{preconditioner.__name__}_iterations.txt", "a") as f:
    #     f.write(f"iterations: {iterations}\n")
    N = int(np.sqrt(A.shape[0]))
    # if preconditioner.__name__ == 'ssor_precond':
    #     with open(f"data/pcg_{preconditioner.__name__}_{preconditioner.omega:.1f}_residuals.txt", "a") as f:
    #         f.write(f"res: {res}\n")
    
    #     with open(f"data/pcg_{preconditioner.__name__}_{preconditioner.omega:.1f}_iterations.txt", "a") as f:
    #         f.write(f"iterations: {iterations}\n")
    # else:
    with open(f"data/pcg_{preconditioner.__name__}_{N}_residuals.txt", "a") as f:
        f.write(f"res: {res}\n")

    with open(f"data/pcg_{preconditioner.__name__}_{N}_iterations.txt", "a") as f:
        f.write(f"iterations: {iterations}\n")


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

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def gmres_restarted(A, b, preconditioner=None, tol=1e-8, max_iter=1000, restart=50):
    """
    Restarted GMRES implementation (Algorithm 1.27 with restart).
    
    Args:
        A: Square matrix or function implementing matrix-vector product.
        b: Right-hand side vector.
        preconditioner: Preconditioner function (optional).
        tol: Convergence tolerance.
        max_iter: Maximum number of outer iterations.
        restart: Number of inner iterations before restart.

    Returns:
        x: Approximate solution.
    """
    n = np.shape(b)[0]
    x = np.zeros_like(b)  # Initial guess

    for outer_iter in range(max_iter):
        # Step 1: Initial residual
        r = b - A @ x if preconditioner is None else preconditioner(b - A @ x)
        gamma0 = np.linalg.norm(r)
        # Step 2: Check initial convergence
        if gamma0 < tol:
            return x

        # Step 3: Initialize
        V = np.zeros((n, restart + 1))
        H = np.zeros((restart + 1, restart))
        c_list = np.zeros(restart)
        s_list = np.zeros(restart)
        gamma = np.zeros(restart + 1)
        gamma[0] = gamma0
        V[:, 0] = r / gamma0

        converged = False
        for j in range(restart):
            # Step 5: Arnoldi loop
            w = A @ V[:, j] if preconditioner is None else preconditioner(A @ V[:, j])

            # Step 6–9: Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w -= H[i, j] * V[:, i]

            # Step 10: Compute h_{j+1,j}
            H[j + 1, j] = np.linalg.norm(w)

            # Step 11–13: Apply Givens rotations
            for i in range(j):
                temp = c_list[i] * H[i, j] + s_list[i] * H[i + 1, j]
                H[i + 1, j] = -s_list[i] * H[i, j] + c_list[i] * H[i + 1, j]
                H[i, j] = temp

            # Step 14: Compute new Givens rotation
            beta = np.hypot(H[j, j], H[j + 1, j])
            if beta == 0:
                c_list[j], s_list[j] = 1, 0
            else:
                c_list[j] = H[j, j] / beta
                s_list[j] = H[j + 1, j] / beta

            # Step 15: Apply to H and gamma
            H[j, j] = beta
            gamma[j + 1] = -s_list[j] * gamma[j]
            gamma[j] = c_list[j] * gamma[j]

            # Step 16: Check convergence
            if abs(gamma[j + 1]) < tol:
                # Step 17–20: Solve for y and update x
                alpha = np.zeros(j + 1)
                for i in range(j, -1, -1):
                    alpha[i] = gamma[i]
                    for k in range(i + 1, j + 1):
                        alpha[i] -= H[i, k] * alpha[k]
                    alpha[i] /= H[i, i]
                x = x + V[:, :j + 1] @ alpha
                converged = True
                break

            # Step 22: Continue with Arnoldi process
            V[:, j + 1] = w / H[j + 1, j]

        if not converged:
            alpha = np.zeros(restart)
            for i in range(restart - 1, -1, -1):
                alpha[i] = gamma[i]
                for k in range(i + 1, restart):
                    alpha[i] -= H[i, k] * alpha[k]
                alpha[i] /= H[i, i]
            x = x + V[:, :restart] @ alpha

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

    def jacobi_precond(r):
        return r / D
    return jacobi_precond

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
    def sgs_precond(r):
        # Use spsolve instead of spsolve_triangular for possible speedup
        z3 = sp.linalg.spsolve(LD, r)
        z2 = D @ z3
        z = sp.linalg.spsolve(UD, z2)
        return z
    return sgs_precond

def ssor_preconditioner(L, D, U, omega=1.8):
    """Symmetric Successive Over-Relaxation (SSOR) preconditioner for the system matrix."""
    LD = (L + D / omega).tocsr()
    UD = (U + D / omega).tocsr()
    
    def ssor_precond(r):
        ssor_precond.omega = omega
        # Use spsolve instead of spsolve_triangular for possible speedup
        z3 = sp.linalg.spsolve((1/(2-omega))*LD, r)
        z2 = (D/omega) @ z3
        z = sp.linalg.spsolve(UD, z2)
        return z
    return ssor_precond

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
    def ic0_precond(r):
        # Solve L y = r
        y = sp.linalg.spsolve(L, r)
        # Solve L^T x = y
        x = sp.linalg.spsolve(L.transpose().tocsr(), y)
        return x
    return ic0_precond


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
    N = 512
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    # print("u (2D grid):")
    # print(u)

    u_flat = u.reshape(N * N)
    potential = lambda x,y: 0*x*y #lambda x, y: double_well_potential(x,y)
    # potential = harmonic_potential  # Change to desired potential function
    sigma = 0
    
    # A = sparse_system_matrix(N, lambda x, y: 0*x*y)
    A = sparse_system_matrix(N, potential)
    L, U, D = split_matrix(A)


    # omega = 1.8
    # start_time = time.time()
    # v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ic0_preconditioner(A))
    # elapsed_time = time.time() - start_time
    # print(f"Time for shifted_inverse_power_method with IC(0) preconditioner: {elapsed_time:.4f} seconds")
    # # SSOR preconditioner
    # start_time = time.time()
    # v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ssor_preconditioner(L, D, U, omega))
    # elapsed_time = time.time() - start_time
    # print(f"Time for shifted_inverse_power_method with SSOR preconditioner: {elapsed_time:.4f} seconds")

    # # SGS preconditioner
    # start_time = time.time()
    # v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, sgs_preconditioner(L, U, D))
    # elapsed_time = time.time() - start_time
    # print(f"Time for shifted_inverse_power_method with SGS preconditioner: {elapsed_time:.4f} seconds")

    # # Jacobi preconditioner
    # start_time = time.time()
    # v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, jacobi_preconditioner(A))
    # elapsed_time = time.time() - start_time
    # print(f"Time for shifted_inverse_power_method with Jacobi preconditioner: {elapsed_time:.4f} seconds")

    # No preconditioner (plain CG)
    start_time = time.time()
    # open("cg_residuals.txt", "w").close()
    # open("cg_iterations.txt", "w").close()
    v, lambda_v = shifted_inverse_power_method(A, sigma, cg, None)
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with no preconditioner (plain CG): {elapsed_time:.4f} seconds")
    
    # CG with scipy
    # start_time = time.time()
    # v, lambda_v = shifted_inverse_power_method(A, sigma, sp.linalg.cg, None)
    # elapsed_time = time.time() - start_time
    # print(f"Time for shifted_inverse_power_method with scipy CG: {elapsed_time:.4f} seconds")

    # # GMRES restarted
    # start_time = time.time()
    # v, lambda_v = shifted_inverse_power_method(A, sigma, gmres_restarted, None)
    # elapsed_time = time.time() - start_time
    # print(f"Time for GMRES restarted: {elapsed_time:.4f} seconds")


    print(f"\nEigenvalue lambda_v closest to sigma={sigma}:")
    print(lambda_v)
    print(f"Analytical eigenvalue with h={1/(N+1)}: {analytical_laplacian_eigenvalue(1, 1, N)}")
    print(f"difference: {lambda_v - analytical_laplacian_eigenvalue(1, 1, N)}")
    print(f"real difference: {np.abs(lambda_v - np.pi**2)}")

    # # print analytical eigenvalues for some combinations of p and q
    # for p in range(1, 10):
    #     for q in range(1, 10):
    #         analytical_eigenvalue = analytical_laplacian_eigenvalue(p, q, N)
    #         print(f"Analytical eigenvalue for p={p}, q={q}: {analytical_eigenvalue}")

    # Plot the eigenvector over the grid
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij')
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, v.reshape(N, N), shading='auto', cmap='viridis')
    plt.colorbar(label='Eigenvector value')
    plt.title(f'Eigenvector closest to sigma={sigma}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # Plot the potential
    # plt.figure(figsize=(8, 6)) 
    # plt.pcolormesh(X, Y, potential(X, Y).reshape(N, N), shading='auto', cmap='plasma')
    # plt.colorbar(label='Potential V(x, y)')
    # plt.title('Potential V(x, y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    # 3D surface plot of the eigenvector

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, v.reshape(N, N), cmap='viridis', edgecolor='none')
    ax.set_title(f'3D Surface: Eigenvector closest to sigma={sigma}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Eigenvector value')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()

def optimal_strategy(N=128):
    # u = np.arange(N * N).reshape(N, N)
    u = np.ones((N, N))  # Using a simple constant function for demonstration
    # print("u (2D grid):")
    # print(u)

    u_flat = u.reshape(N * N)
    # potential = lambda x,y: 0*x*y #lambda x, y: double_well_potential(x,y)
    potential = harmonic_potential  # Change to desired potential function
    sigma = 0
    
    # A = sparse_system_matrix(N, lambda x, y: 0*x*y)
    A = sparse_system_matrix(N, potential)
    L, U, D = split_matrix(A)

    # SSOR preconditioner
    omega = 1.8
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ssor_preconditioner(L, D, U, omega), max_iter=8)
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with SSOR preconditioner: {elapsed_time:.4f} seconds")


    # No preconditioner (plain CG)
    start_time = time.time()
    v, lambda_v = shifted_inverse_power_method(A, sigma, cg, precond=None, start=v)
    elapsed_time = time.time() - start_time
    print(f"Time for shifted_inverse_power_method with no preconditioner (plain CG): {elapsed_time:.4f} seconds")


    print(f"\nEigenvalue lambda_v closest to sigma={sigma}:")
    print(lambda_v)
    print(f"Analytical eigenvalue with h={1/(N+1)}: {analytical_laplacian_eigenvalue(1, 1, N)}")
    print(f"difference: {lambda_v - analytical_laplacian_eigenvalue(1, 1, N)}")
    print(f"real difference: {np.abs(lambda_v - np.pi**2)}")

    # Plot the eigenvector over the grid
    h = 1.0 / (N + 1)
    x_list = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x_list, x_list, indexing='ij')
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, v.reshape(N, N), shading='auto', cmap='viridis')
    plt.colorbar(label='Eigenvector value')
    plt.title(f'Eigenvector closest to sigma={sigma}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X, Y, v.reshape(N, N), cmap='viridis', edgecolor='none')
    # ax.set_title(f'3D Surface: Eigenvector closest to sigma={sigma}')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('Eigenvector value')
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # plt.show()

def eigenvalue_convergence(N, sigma=0, p=1, q=1):
    u = np.ones((N, N))  # Using a simple constant function for demonstration

    u_flat = u.reshape(N * N)
    potential = lambda x,y: 0*x*y #lambda x, y: double_well_potential(x,y)

    # A = sparse_system_matrix(N, lambda x, y: 0*x*y)
    A = sparse_system_matrix(N, potential)
    L, U, D = split_matrix(A)


    v, lambda_v = shifted_inverse_power_method(A, sigma, cg, precond=None)

    
    continuous_value = (p**2 + q**2) * np.pi**2 / 2
    discrete_difference = np.abs(lambda_v - analytical_laplacian_eigenvalue(p, q, N))
    real_difference = np.abs(lambda_v - continuous_value)
    return discrete_difference, real_difference, continuous_value



def compute_residuals(N):
    u = np.ones((N, N))  

    u_flat = u.reshape(N * N)
    potential = harmonic_potential  
    sigma = 0
    
    A = sparse_system_matrix(N, potential)
    L, U, D = split_matrix(A)

    # preconditioner_list = [
    #     jacobi_preconditioner(A),
    #     sgs_preconditioner(L, U, D),
    #     ssor_preconditioner(L, D, U, omega=1.8),
    #     ic0_preconditioner(A),
    # ]


    # omega_list = np.arange(0.6, 2.3, 0.2)
    # preconditioner_list = [sgs_preconditioner(L, U, D)]
    # for omega in omega_list:
    #     preconditioner_list.append(ssor_preconditioner(L, D, U, omega))

    v, lambda_v = shifted_inverse_power_method(A, sigma, cg, None)
    v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, ssor_preconditioner(L, D, U, 1.8))
    # for precond in preconditioner_list:
    #     v, lambda_v = shifted_inverse_power_method(A, sigma, pcg, precond)


def read_residuals(filename="cg_residuals.txt"):
    """Read all residuals from a file and return them as a single concatenated list, also returning the first residual list."""
    all_res = []
    first_res = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("res:"):
                res_line = line.split("res:")[1].strip()
                res = eval(res_line)
                all_res.extend(res)
                if first_res is None:
                    first_res = res
    return first_res, all_res

def plot_residuals(res, solver_name="cg"):
    """Plot the residuals of the CG method."""
    os.makedirs('plots', exist_ok=True)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence of CG Method')
    plt.grid()
    plt.savefig(os.path.join('plots', f'{solver_name}_residuals.png'))
    plt.show()

def read_iterations(filename="cg_iterations.txt"):
    """Read all iteration counts from a file and return them as a list, also returning the first iteration count."""
    all_iters = []
    first_iters = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("iterations:"):
                iter_line = line.split("iterations:")[1].strip()
                try:
                    iters = int(iter_line)
                    all_iters.append(iters)
                    if first_iters is None:
                        first_iters = iters
                except ValueError:
                    pass
    return first_iters, all_iters

def plot_iterations(iters, solver_name="cg"):
    """Plot the iteration counts."""
    os.makedirs('plots', exist_ok=True)
    plt.plot(iters, marker='o')
    plt.xlabel('Run')
    plt.ylabel('Iterations')
    plt.title('CG Iteration Counts per Run')
    plt.grid()
    plt.savefig(os.path.join('plots', f'{solver_name}_iterations.png'))
    plt.show()


if __name__ == "__main__":
    # dimile_new()
    # optimal_strategy(128)

    N_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # Plot eigenvalue convergence for each N
    discrete_diffs = []
    real_diffs = []
    for N in N_list:
        discrete_diff, real_diff, cont_value = eigenvalue_convergence(N, sigma=24, p=1, q=2)
        discrete_diffs.append(discrete_diff)
        real_diffs.append(real_diff)
        print(f"N={N}, Discrete Difference: {discrete_diff}, Real Difference: {real_diff}")

    plt.figure(figsize=(8, 6))
    plt.plot(N_list, discrete_diffs, marker='o', label=r'$| \lambda_{1,1} - \lambda(\sigma = 24) |$')
    plt.plot(N_list, real_diffs, marker='s', label=r'$| \frac{5}{2} \pi^2 - \lambda(\sigma = 24) |$')
    plt.xlabel('Grid size N')
    plt.ylabel('Eigenvalue Difference')
    plt.title('Eigenvalue Convergence vs Grid Size')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('plots', 'eigenvalue_convergence_vs_N.png'))
    plt.show()
    exit()
    # first_res, res = read_residuals()
    # plot_residuals(first_res)

    # first_iters, iters = read_iterations()
    # plot_iterations(iters)
    # Delete all .txt files in the data folder
    

    # N = 128
    # CALCULATE = True  # Set to False to skip calculations and only plot results

    # if CALCULATE:
    #     data_folder = "data"
    #     if os.path.exists(data_folder):
    #         for fname in os.listdir(data_folder):
    #             if fname.endswith(".txt"):
    #                 os.remove(os.path.join(data_folder, fname))

    #     compute_residuals(N)
    
    N_list = [16, 32, 64, 128, 256]
    # for N in N_list:
    #     compute_residuals(N)

    # ----------------------------------
    #
    # PLOT ITERATIONS OVER N
    #
    # ----------------------------------
    cg_first_iters = []
    ssor_first_iters = []
    for N in N_list:
        filename = f"data/cg_{N}_iterations.txt"
        if os.path.exists(filename):
            first_iters, _ = read_iterations(filename)
            if first_iters is not None:
                cg_first_iters.append(first_iters)
        
        filename = f"data/pcg_ssor_precond_{N}_iterations.txt"
        if os.path.exists(filename):
            first_iters, _ = read_iterations(filename)
            if first_iters is not None:
                ssor_first_iters.append(first_iters)
 
    plt.figure(figsize=(8, 6))
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel('Grid size N', fontsize=18)
    plt.ylabel('First CG Iteration Count', fontsize=18)
    plt.title('CG Iterations vs Grid Size', fontsize=20)    
    plt.plot(N_list, cg_first_iters, marker='o', label='CG')
    plt.plot(N_list, ssor_first_iters, marker='o', label='SSOR PCG (omega=1.8)')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(os.path.join('plots', 'cg_iterations_vs_N.png'))
    plt.show()

    # ----------------------------------
    #
    # PLOT RESIDUALS
    #
    # ----------------------------------
    # cg_res, cg_all_res = read_residuals("data/cg_residuals.txt")
    # plt.plot(cg_res, label='cg')

    # precond_list = ['jacobi_precond', 'sgs_precond', 'ssor_precond', 'ic0_precond']

    # # omega_list = np.arange(0.6, 2.0, 0.2)
    # # precond_list = ['sgs_precond']
    # # for omega in omega_list:
    # #     precond_list.append(f'ssor_precond_{omega:.1f}')

    # precond_res = []
    # for precond in precond_list:
    #     first_res, res = read_residuals(f"data/pcg_{precond}_residuals.txt")
        
    #     plt.plot(first_res, label=precond)
    #     first_iters, iters = read_iterations(f"data/pcg_{precond}_iterations.txt")
    #     print(f"First iteration count for {precond}: {first_iters}")

    # plt.gcf().set_size_inches(12, 8)
    # plt.yscale('log')
    # plt.legend()
    # plt.grid()
    # plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.xlabel('Iteration', fontsize=18)
    # plt.ylabel('Residual Norm', fontsize=18)
    # plt.title(f'Convergence of PCG Method (N={N})', fontsize=20)
    # plt.legend(fontsize=16)
    # plt.grid(True, which='both')
    # plt.savefig(os.path.join('plots', f'residuals_{N}.png'))
    # plt.show()


    # ----------------------------------
    #
    # PLOT ITERATIONS
    #
    # ----------------------------------
    # cg_first_iters, cg_iters = read_iterations("data/cg_iterations.txt")
    # plt.plot(cg_iters, label='cg')

    # precond_list = ['jacobi_precond', 'sgs_precond', 'ssor_precond_1.8', 'ic0_precond']

    # # omega_list = np.arange(0.6, 2.0, 0.2)
    # # precond_list = ['sgs_precond']
    # # for omega in omega_list:
    # #     precond_list.append(f'ssor_precond_{omega:.1f}')

    # precond_res = []
    # for precond in precond_list:
    #     first_iter, iter = read_iterations(f"data/pcg_{precond}_iterations.txt")
    #     plt.plot(iter, label=precond)

    # plt.gcf().set_size_inches(12, 8)
    # # plt.yscale('log')
    # plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.xlabel('IPM Iteration', fontsize=18)
    # plt.ylabel('Number of solver iterations', fontsize=18)
    # plt.title(f'Convergence of PCG Method (N={N})', fontsize=20)
    # plt.legend(fontsize=16)
    # plt.grid(True, which='both')
    # plt.savefig(os.path.join('plots', f'iterations_{N}.png'))
    # plt.show()
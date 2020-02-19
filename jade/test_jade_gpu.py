import torch as th
import numpy as np

from jade.jade_gpu import memory_size, jade_par


def generate_case(k=5, m=5, noise=0):
    """
    generates a test case with known solution eigenvectors U

    Args:
        k: int
            number of matrices to stack
        m: int
            square dimension of matrices
        sigma: float
            std of noise to be added to the observed data

    Returns:
        M: np.ndarray
            tensor containing k matrices of shape m x m
        U: np.ndarray
            matrix containing the true joint eigenvalues of the matrices in M
    """
    U, S, V = th.svd(th.randn(m, m))
    rand = th.randn(k, m, m)
    diag = th.eye(m, m)
    diag = diag[None, :, :] * rand
    M = U[None, :, :] @ diag @ U.transpose(1, 0)[None, :, :] + rand * noise
    print('test case shape: ', M.shape, '\n has size: ', memory_size(M))
    return M, U


def check(U, V):
    """
    checks if U and V are orthogonal

    Args:
        U: th.tensor
        V: np.tensor

    Returns:
        None
    """
    # check if orthogonal and undo arbitrary sign
    UV = np.abs(U.T @ V)
    # normalize vectors to unit length
    UV /= np.max(UV, axis=1, keepdims=True)
    # accept numerical tolerance
    UV[np.abs(UV) < 1e-12] = 0.
    # sort diagonally
    UV = UV[np.lexsort(UV.T)]
    # assert diagonal matrix
    from numpy.testing import assert_array_equal
    assert_array_equal(UV, np.eye(len(UV)))


def test_jade_par():
    M, U = generate_case()
    A, V, n_iter = jade_par(M, threshold=None, maxiter=1000)
    check(U.numpy(), V.numpy())

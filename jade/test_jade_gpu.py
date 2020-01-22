import torch as th
import numpy as np

from jade.jade_gpu import memory_size, jade_par


def generate_case(m=5, k=5, noise=0):
    """
    k matrices of m by m size, normal noise multiplied by noise=0
    """
    U, S, V = th.svd(th.randn(m, m))
    rand = th.randn(k, m, m)
    diag = th.eye(m, m)
    diag = diag[None, :, :] * rand
    M = U[None, :, :] @ diag @ U.transpose(1, 0)[None, :, :] + rand * noise
    print('test case shape: ', M.shape, '\n has size: ', memory_size(M))
    return M, U


def check(U, V):
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

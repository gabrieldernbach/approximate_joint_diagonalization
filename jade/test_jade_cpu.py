import numpy as np
from jade.jade_cpu import jade, jade_jit, jade_parallel, pad


def check(U, V):
    """
    rough check about U and V should be orthogonal
    :param U: ndarray of shape (m x m)
    :param V: ndarray of shape (m x m)
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

def generate_case(k=5, m=6, sigma=0):
    """
    generates a test case with known solution U
    :return: M (k x m x m), U (m x m)
    """
    np.random.seed(1)
    U, S, _ = np.linalg.svd(np.random.randn(m, m))
    rand = np.random.randn(k, m, m)
    diag = np.eye(m, m)
    randdiag = diag[None, :, :] * rand
    M = U[None, :, :] @ randdiag @ U.T[None, :, :]
    M += np.random.randn(k, m, m) * sigma
    return M, U


def test_jade():
    M, U = generate_case()
    A, V = jade(M)
    check(V, U)
    pass


def test_jade_jit():
    M, U = generate_case()
    A, V = jade_jit(M)
    check(V, U)
    pass


def test_jade_par():
    M, U = generate_case(4, 7)
    A, V = jade_parallel(M)
    check(V, U)
    M, U = generate_case(5, 5)
    A, V = jade_parallel(M)
    check(V, U)
    pass

def test_pad():
    M, U = generate_case(5, 5, 0)
    assert pad(M)[0].shape[1] == 6
    M, U = generate_case(6, 6, 0)
    assert pad(M)[0].shape[1] == 6

# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#
# License: MIT
from time import time

import numpy as np

def transform_set(M, D):
    # apply matrix M to first dim of D
    K, N, _ = D.shape
    op = np.zeros((K, N, N)) 
    for i, d in enumerate(D): # enumerate goes along 1st dimension (aka batches)
        op[i] = M.dot(d.dot(M.T))
    return op

def ajd_pham(X, eps=1e-6, n_iter_max=15):
    """Approximate joint diagonalization based on pham's algorithm.
    This is a direct implementation of the PHAM's AJD algorithm [1].
    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_channels)
        A set of covariance matrices to diagonalize
    eps : float (default 1e-6)
        tolerance for stoping criterion.
    n_iter_max : int (default 1000)
        The maximum number of iteration to reach convergence.
    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        the diagonalizer
    D : ndarray, shape (n_trials, n_channels, n_channels)
        the set of quasi diagonal matrices
    Notes
    -----
    .. versionadded:: 0.2.4
    See Also
    --------
    rjd
    uwedge
    References
    ----------
    [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
    definite Hermitian matrices." SIAM Journal on Matrix Analysis and
    Applications 22, no. 4 (2001): 1136-1152.
    """
     # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(n_iter_max):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D



if __name__ == '__main__':
    from numpy.testing import assert_array_equal

    """Test approximate joint diagonalization."""
    n, p = 10, 3 
    rng = np.random.RandomState(42)
    
    diagonals = rng.uniform(size=(n, p))
    V = rng.randn(p, p)  # mixing matrix
    M = np.array([V.dot(d[:, None] * V.T) for d in diagonals])  # dataset
    Vhat, _ = ajd_pham(M)

    # check if V and Vhat are identical up to permutation and scaling
    VA = np.abs(Vhat.dot(V))  # undo negative scaling 
    VA /= np.max(VA, axis=1, keepdims=True) # normalize to 1
    VA[np.abs(VA) < 1e-12] = 0. # numerical tolerance
    assert_array_equal(VA[np.lexsort(VA)], np.eye(p))
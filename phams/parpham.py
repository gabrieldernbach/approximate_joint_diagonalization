from time import time
import numpy as np
from numba import jit, njit

def mean_rotation(C):
    C_mean = np.mean(C, axis=0)
    vals, vecs = np.linalg.eigh(C_mean)
    B = vecs.T / np.sqrt(vals[:, None])
    C = B[None,:,:] @ M @ B.T[None,:,:]
    return B, C

def scheduler(tournament):
    '''return next draw of tournament table'''
    old = tournament
    new = np.zeros(old.shape, dtype=np.int64)
    
    # players of row 0
    new[0, 0] = old[0, 0]
    new[0, 1] = old[1, 0]
    new[0, 2:] = old[0, 1:-1]
    # against players of row 1
    new[1, -1] = old[0, -1];
    new[1, :-1] = old[1, 1:]
    return new

def rotmat(C,tournament):
    '''
    compute update matrix according to phams method see:
    D. T. Pham, “Joint Approximate Diagonalization of Positive Definite Hermitian Matrices,”
    SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 4, pp. 1136–1152, Jan. 2001.
    '''

    m = C.shape[1]
    k = C.shape[0]

    i = tournament.min(axis=0)
    j = tournament.max(axis=0)

    C_ii = C[:, i, i] 
    C_jj = C[:, j, j]
    C_ij = C[:, i, j]

    # find g_ij (2.04)
    g_ij = np.mean(C_ij / C_ii, axis=0)
    g_ji = np.mean(C_ij / C_jj, axis=0)

    # find w_ij (2.07) with w_ii, w_jj = 1, 1
    w_ij = np.mean(C_jj / C_ii, axis=0)
    w_ji = np.mean(C_ii / C_jj, axis=0)

    # solve 2.10, that is find h such that W @ h = g
    w_tilde_ji = np.sqrt(w_ji / w_ij)
    w_prod = np.sqrt(w_ij * w_ji)
    tmp1 = (w_tilde_ji * g_ij + g_ji) / (w_prod + 1)
    tmp2 = (w_tilde_ji * g_ij - g_ji) / np.maximum(w_prod - 1, 1e-9) 
    h12 = tmp1 + tmp2 # (2.10)
    h21 = np.conj((tmp1 - tmp2) / w_tilde_ji)

    # cumulative decrease in current sweep
    decrease = np.sum(k * (g_ij * np.conj(h12) + g_ji * h21) / 2.0)

    # construct T by 2.08
    tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
    tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))

    T = np.eye(m)
    T[i, j] = -h12 / tmp
    T[j, i] = -h21 / tmp

    return T, decrease

def phams(Gamma, threshold=1e-50, maxiter=1000, mean_initialize=False):
    '''
    find approximate joint diagonalization of set of square matrices Gamma,
    returns joint basis B and corresponding set of approximate diagonals C
    '''
    C = np.copy(Gamma)
    m = C.shape[1]
    B = np.eye(m)
    
    tournament = np.random.permutation(m).reshape(2, m//2)

    # precompute B
    if mean_initialize:
        B, C = mean_rotation(C)
 
    active = 1
    n_iter = 0
    
    while active == 1 and n_iter < maxiter:
        # computation of rotations           
        T, decrease = rotmat(C, tournament)
        print(decrease)

        # update of C and B matrices
        C = T @ C @ T.T
        B = T @ B

        tournament = scheduler(tournament)
        n_iter += 1
        active = np.abs(decrease) > threshold

    return B, C 


if __name__ == '__main__':
    from numpy.testing import assert_array_equal

    """Test approximate joint diagonalization."""
    # create k matrices of shape m x m
    k, m = 150, 150

    rng = np.random.RandomState(42) 
    
    diagonals = rng.uniform(size=(k, m))
    B = rng.randn(m, m)  # mixing matrix
    M = np.array([B.dot(d[:, None] * B.T) for d in diagonals])  # dataset
    Bhat, _ = phams(M)

    # check if B and Bhat are identical up to permutation and scaling
    BA = np.abs(Bhat.dot(B))  # undo negative scaling 
    BA /= np.max(BA, axis=1, keepdims=True) # normalize to 1
    BA[np.abs(BA) < 1e-12] = 0. # numerical tolerance
    print(BA)
    import matplotlib.pyplot as plt
    plt.imshow(BA @ BA.T)
    plt.show()
    assert_array_equal(BA[np.lexsort(BA)], np.eye(m))

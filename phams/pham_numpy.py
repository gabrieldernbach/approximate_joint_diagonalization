from time import time
import numpy as np
from numba import jit, njit

def mean_rotation(C):
    C_mean = np.mean(C, axis=0)
    vals, vecs = np.linalg.eigh(C_mean)
    B = vecs.T / np.sqrt(vals[:, None])
    C = B[None,:,:] @ M @ B.T[None,:,:]
    return B, C

def rotmat(C,i,j):
    '''compute update matrix'''
    C_ii = C[:, i, i] 
    C_jj = C[:, j, j]
    C_ij = C[:, i, j]

    # find g_ij (2.04)
    g_ij = np.mean(C_ij / C_ii)
    g_ji = np.mean(C_ij / C_jj)

    # find w_ij (2.07) with w_ii, w_jj = 1, 1
    w_ij = np.mean(C_jj / C_ii)
    w_ji = np.mean(C_ii / C_jj)

    # solve 2.10, that is find h such that W @ h = g
    w_tilde_ji = np.sqrt(w_ji / w_ij)
    w_prod = np.sqrt(w_ij * w_ji)
    tmp1 = (w_tilde_ji * g_ij + g_ji) / (w_prod + 1)
    tmp2 = (w_tilde_ji * g_ij - g_ji) / max(w_prod - 1, 1e-9) 
    h12 = tmp1 + tmp2 # (2.10)
    h21 = np.conj((tmp1 - tmp2) / w_tilde_ji)

    # decrease in current step 
    decrease = k * (g_ij * np.conj(h12) + g_ji * h21) / 2.0

    # construct T by 2.08
    tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
    tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
    T = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]]) # stacks all scalar values
    return T, decrease

def phams(Gamma, threshold=1e-50, mean_initialize=False):
    '''
    find approximate joint diagonalization of set of square matrices Gamma,
    returns joint basis B and corresponding set of approximate diagonals C
    '''
    C = np.copy(Gamma)
    k, m, _ = C.shape
    B = np.eye(m)

    # precompute B
    if mean_initialize:
        B, C = mean_rotation(C)
 
    active = 1
    while active == 1:
        cum_decrease = 0
        for i in range(0, m):
            for j in range(0, i):
                # computation of rotations           
                T, decrease = rotmat(C,i,j)
                cum_decrease += decrease

                # update of C and B matrices
                pair = np.array((i,j))
                C[:,:,pair] = C[:,:,pair] @ T.T[None,:,:]
                C[:,pair,:] = T[None,:,:] @ C[:,pair,:]
                # C[:,:,pair] = np.einsum('ij,klj->kli',T,C[:,:,pair]) einsum alternative
                # C[:,pair,:] = np.einsum('ij,kjl->kil',T,C[:,pair,:]) einsum alternative
                B[pair,:] = T @ B[pair,:]

        active = np.abs(decrease) > threshold

    return B, C 


if __name__ == '__main__':
    from numpy.testing import assert_array_equal

    """Test approximate joint diagonalization."""
    # create k matrices of shape m x m
    k, m = 10, 50

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

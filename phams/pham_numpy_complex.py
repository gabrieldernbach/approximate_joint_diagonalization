import numpy as np


def loss(C):
    l = 0
    for i in range(C.shape[0]):
        l += np.sum(np.log(np.diag(C[i, :, :]))) - np.log(np.linalg.det(C[i, :, :]))
    return l


def mean_rotation(C):
    C_mean = np.mean(C, axis=0)
    vals, vecs = np.linalg.eigh(C_mean)
    B = vecs.T / np.sqrt(vals[:, None])
    C = B[None, :, :] @ C @ B.T[None, :, :]
    return B, C


def rotmat(C, i, j):
    '''
    compute update matrix according to phams method see:
    D. T. Pham, “Joint Approximate Diagonalization of Positive Definite Hermitian Matrices,”
    SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 4, pp. 1136–1152, Jan. 2001.
    '''

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
    h12 = tmp1 + tmp2  # (2.10)
    h21 = np.conj((tmp1 - tmp2) / w_tilde_ji)

    # decrease in current step 
    decrease = C.shape[0] * (g_ij * np.conj(h12) + g_ji * h21) / 2.0

    # construct T by 2.08
    tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
    tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
    T = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])  # stacks all scalar values
    return T, decrease


def phams(Gamma, threshold=1e-50, max_iter=8000, mean_initialize=False):
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

    active, n_iter = 1, 0
    while active == 1 and n_iter < max_iter:
        cum_decrease = 0
        for i in range(0, m):
            for j in range(0, i):
                # computation of rotations           
                T, decrease = rotmat(C, i, j)
                cum_decrease += decrease

                # update of C and B matrices
                pair = np.array((i, j))
                C[:, :, pair] = C[:, :, pair] @ T.T[None, :, :]
                C[:, pair, :] = T[None, :, :] @ C[:, pair, :]
                # C[:,:,pair] = np.einsum('ij,klj->kli',T,C[:,:,pair]) einsum alternative
                # C[:,pair,:] = np.einsum('ij,kjl->kil',T,C[:,pair,:]) einsum alternative
                B[pair, :] = T @ B[pair, :]

        # evaluate stopping criteria 
        active = np.abs(decrease) > threshold
        n_iter += 1

    return B, C, n_iter


def gentest(num_matrices=40, shape_matrices=60):
    '''
    generate testcase for joint approximate diagonalization
    under assumption of non orthogonal joint basis.
    '''
    rng = np.random.RandomState(42)
    # draw random diagonal
    diagonals = rng.uniform(size=(num_matrices, shape_matrices))
    # generate joint mixing matrix
    B = rng.randn(shape_matrices, shape_matrices)
    # rotate diagonals by mixing matrix
    M = np.array([B.dot(d[:, None] * B.T) for d in diagonals])
    return B, M


if __name__ == '__main__':
    """Test approximate joint diagonalization."""
    # create k matrices of shape m x m
    basis, setM = gentest(num_matrices=40, shape_matrices=60)

    print(f'initial loss: {loss(setM):.5f}')
    basis_hat, setM_hat, n_iter = phams(setM)
    print(f'final loss: {loss(setM_hat)}')

    # check if basis and basis_hat are identical up to permutation and scaling
    from numpy.testing import assert_array_equal

    BA = np.abs(basis_hat.dot(basis))  # undo negative scaling
    BA /= np.max(BA, axis=1, keepdims=True)  # normalize to 1
    BA[np.abs(BA) < 1e-12] = 0.  # numerical tolerance
    print(BA)
    assert_array_equal(BA[np.lexsort(BA)], np.eye(BA.shape[0]))

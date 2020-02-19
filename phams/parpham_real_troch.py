import numpy as np
import torch as th


def loss(C):
    l = 0
    for i in range(C.shape[0]):
        log_determinant_diagonal = th.sum(th.log(th.diag(C[i, :, :])))
        log_determinant = 2 * th.sum(th.log(th.diag(th.cholesky(C[i, :, :]))))
        l += log_determinant_diagonal - log_determinant
    return l


def mean_rotation(C):
    C_mean = th.mean(C, dim=0)
    vals, vecs = th.symeig(C_mean)
    B = vecs.T / th.sqrt(vals[:, None])
    C = B[None, :, :] @ C @ B.T[None, :, :]
    return B, C


def init_tournament(m):
    '''initialize random tournament table with pairwise groups'''
    if m % 2 == 0:
        tournament = th.randperm(m).reshape(2, m // 2)
        padflag = 0
    else:
        tournament = th.randperm(m)
        tournament = th.insert(tournament, 0, m).reshape(2, (m + 1) // 2)
        padflag = 1
    return tournament, padflag


def scheduler(tournament):
    '''return next draw of tournament table'''
    old = tournament
    new = th.zeros(old.shape, dtype=th.int64)

    # players of row 0
    new[0, 0] = old[0, 0]
    new[0, 1] = old[1, 0]
    new[0, 2:] = old[0, 1:-1]
    # against players of row 1
    new[1, -1] = old[0, -1];
    new[1, :-1] = old[1, 1:]
    return new


def rotmat(C, tournament, padflag):
    '''
    compute update matrix according to phams method see:
    D. T. Pham, “Joint Approximate Diagonalization of Positive Definite Hermitian Matrices,”
    SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 4, pp. 1136–1152, Jan. 2001.
    '''
    m = C.shape[1]
    k = C.shape[0]

    i = tournament[:, padflag:].min(dim=0)[0]
    j = tournament[:, padflag:].max(dim=0)[0]

    C_ii = C[:, i, i]
    C_jj = C[:, j, j]
    C_ij = C[:, i, j]

    # find g_ij (2.04)
    g_ij = th.mean(C_ij / C_ii, dim=0)
    g_ji = th.mean(C_ij / C_jj, dim=0)

    # find w_ij (2.07) with w_ii, w_jj = 1, 1
    w_ij = th.mean(C_jj / C_ii, dim=0)
    w_ji = th.mean(C_ii / C_jj, dim=0)

    # solve 2.10, that is find h such that W @ h = g
    w_tilde_ji = th.sqrt(w_ji / w_ij)
    w_prod = th.sqrt(w_ij * w_ji)
    tmp1 = (w_tilde_ji * g_ij + g_ji) / (w_prod + 1)
    tmp2 = (w_tilde_ji * g_ij - g_ji) / th.max(w_prod - 1, th.tensor(1e-9).type(th.float64))
    h12 = tmp1 + tmp2  # (2.10)
    h21 = ((tmp1 - tmp2) / w_tilde_ji)

    # cumulative decrease in current sweep
    decrease = th.sum(k * (g_ij * h12 + g_ji * h21) / 2.0)

    # construct T by 2.08
    tmp = 1 + th.sqrt(1 - h12 * h21)

    T = th.eye(m).type(th.float64)
    T[i, j] = -h12 / tmp
    T[j, i] = -h21 / tmp

    return T, decrease


def phams(Gamma, threshold=1e-50, maxiter=1000, mean_initialize=False):
    """
    find approximate joint diagonalization of set of square matrices Gamma,
    returns joint basis B and corresponding set of approximate diagonals C
    """
    C = Gamma.clone()
    m = C.shape[1]
    B = th.eye(m).type(th.float64)

    tournament, padflag = init_tournament(m)

    # precompute B
    if mean_initialize:
        B, C = mean_rotation(C)

    active = 1
    n_iter = 0

    while active == 1 and n_iter < maxiter:
        # computation of rotations           
        T, decrease = rotmat(C, tournament, padflag)
        print(decrease)

        # plt.imshow(T.numpy(),vmin=-1,vmax=1)
        # plt.savefig(f'plot_{n_iter:00f}.jpg')
        # update of C and B matrices
        C = T @ C @ T.T
        B = T @ B

        tournament = scheduler(tournament)
        n_iter += 1
        active = th.abs(decrease) > threshold

    return B, C, n_iter


def gentest(num_matrices=40, shape_matrices=60):
    '''
    generate testcase for joint approximate diagonalization
    under assumption of non orthogonal joint basis.
    '''
    k = num_matrices
    m = shape_matrices
    rng = th.manual_seed(42)
    # draw random diagonal
    rand = th.abs(th.rand(k, m, m))
    diag = th.eye(m, m)
    diag = diag[None, :, :] * rand
    # generate joint mixing matrix
    B = th.randn(m, m)
    # rotate diagonals by mixing matrix
    M = B[None, :, :] @ diag @ B.transpose(1, 0)[None, :, :]
    return B, M


if __name__ == '__main__':
    """Test approximate joint diagonalization."""
    import matplotlib.pyplot as plt

    # matplotlib.use('TkAgg')
    # create k matrices of shape m x m
    basis, setM = gentest(num_matrices=40, shape_matrices=60)
    test = np.load('test.npz')

    print(f'initial loss: {loss(setM):.5f}')
    basis_hat, setM_hat, n_iter = phams(th.from_numpy(test['setM']).type(th.float64))
    print(f'final loss: {loss(setM_hat)}')

    print(th.sum((basis_hat @ th.from_numpy(test['setM']).type(th.float64) @ basis_hat.T - setM_hat) ** 2))

    print(f"expected loss: {loss(th.from_numpy(test['setM_hat']).type(th.float))}")
    plt.subplot(211)
    plt.imshow(basis.numpy())
    plt.subplot(212)
    plt.imshow(basis_hat.numpy())
    plt.show()

    # check if basis and basis_hat are identical up to permutation and scaling
    import numpy as np
    from numpy.testing import assert_array_equal

    basis_hat = basis_hat.numpy()
    basis = test['basis']
    BA = np.abs(basis_hat.dot(basis))  # undo negative scaling 
    BA /= np.max(BA, axis=1, keepdims=True)  # normalize to 1
    BA[np.abs(BA) < 1e-12] = 0.  # numerical tolerance
    print(BA)
    assert_array_equal(BA[np.lexsort(BA)], np.eye(BA.shape[0]))

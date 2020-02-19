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
    """initialize random tournament table with pairwise groups"""
    if m % 2 == 0:
        tournament = th.randperm(m).reshape(2, m // 2)
        padflag = 0
    else:
        tournament = th.randperm(m)
        tournament = th.insert(tournament, 0, m).reshape(2, (m + 1) // 2)
        padflag = 1
    return tournament, padflag


def scheduler(tournament):
    """return next draw of tournament table"""
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
    """
    compute update matrix according to phams method see:
    D. T. Pham, “Joint Approximate Diagonalization of Positive Definite Hermitian Matrices,”
    SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 4, pp. 1136–1152, Jan. 2001.
    """
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

    if mean_initialize:
        B, C = mean_rotation(C)

    active = 1
    n_iter = 0
    while active == 1 and n_iter < maxiter:
        # computation of rotations           
        T, decrease = rotmat(C, tournament, padflag)
        print(decrease)

        C = T @ C @ T.T
        B = T @ B

        tournament = scheduler(tournament)
        n_iter += 1
        active = th.abs(decrease) > threshold

    return B, C, n_iter


def gentest(num_matrices=40, shape_matrices=60):
    """
    generate test case for joint approximate diagonalization
    under assumption of non orthogonal joint basis.
    """
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
    import os

    baseline_test_file = 'test.npz'
    assert os.path.exists(baseline_test_file), 'please run base_implementation.py to generate a case'

    test = np.load(baseline_test_file)
    A, _, C = test['A'], test['B'], test['C']
    A = th.tensor(A).double()
    C = th.tensor(C).double()

    B, _, _ = phams(C, threshold=1e-50, maxiter=1000, mean_initialize=False)

    # check if A and B are identical up to permutation and scaling
    A, B = A.numpy(), B.numpy()
    BA = np.abs(B.dot(A))  # undo negative scaling
    BA /= np.max(BA, axis=1, keepdims=True)  # normalize to 1
    BA[np.abs(BA) < 1e-8] = 0.  # numerical tolerance

    import matplotlib.pyplot as plt
    plt.imshow(BA @ BA.T)
    plt.show()

    from numpy.testing import assert_array_equal
    assert_array_equal(BA[np.lexsort(BA)], np.eye(len(BA)))  # assert identity

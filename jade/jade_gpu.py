#!/Usr/bin/env python
# coding: utf-8
import math

import torch as th
th.set_default_tensor_type(th.DoubleTensor)


def memory_size(DataTensor):
    """
    counts the elements of the tensor, infers the data type
    and returns the memory size in human readable format

    :param DataTensor: torch tensor
    :return: string of size in human readable format
    """
    size_bytes = DataTensor.element_size() * DataTensor.nelement()
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def scheduler(tournament):
    '''
    returns next draw of tournament table

    :param tournament: ndarray of (2 x n//2) players
    :return: ndarray of (2 x n//2) players
    '''
    old = tournament
    new = th.zeros(old.shape, dtype=th.int64)

    # players of row 0
    new[0, 0] = old[0, 0]
    new[0, 1] = old[1, 0]
    new[0, 2:] = old[0, 1:-1]
    # against players of row 1
    new[1, -1] = old[0, -1]
    new[1, :-1] = old[1, 1:]
    return new


def pad(A):
    """
    adds 0 padding to last row and last column of tensor A.

    :param A: ndarray of shape k x m x m
    :return: ndarray of shape k x (m+1) x (m+1),
             bool that indicates whether padding was applied
    """
    if A.shape[1] % 2 is not 0:
        pad_flag = 1
        j, k = A.shape[0], A.shape[1]
        zers1 = th.zeros((j, k)).reshape(j, k, 1)
        A = th.cat((A, zers1), 2)
        zers2 = th.zeros((j, k + 1)).reshape(j, 1, k + 1)
        A = th.cat((A, zers2), 1)
    else:
        pad_flag = 0
    return A, pad_flag


def offdiag(A):
    """
    computes the frobenius norm of the off diagonal elements
    of the torch tensor A (k x m x m)

    :param A: torch tensor of shape k x m x m
    :return: norm, the frobenius norm of the offdiagonal of A
    """
    mask = ((th.eye(A.shape[1])) == 0).type(th.float)
    offdiag = A * mask
    norm = th.sqrt(th.sum(offdiag ** 2))
    return norm


def rotation_matrix(A, tournament):
    """

    :param A: torch tensor of shape (k x m x m)
    :param tournament: torch tensor of shape (2 x m//2)
    :return: J (m x m) and ssum (squared angle)
    """
    m = A.shape[1]
    J = th.zeros((m, m))

    p = tournament.min(dim=0)[0]
    q = tournament.max(dim=0)[0]

    matp = A[:, p, p] - A[:, q, q]
    matm = A[:, p, q] + A[:, q, p]
    ton = th.sum(matp * matp, dim=0) - th.sum(matm * matm, dim=0)
    toff = 2 * th.sum(matp * matm, dim=0)
    theta = 0.5 * th.atan2(toff, ton + th.sqrt(ton * ton + toff * toff))
    c = th.cos(theta)
    s = th.sin(theta)

    J[p, p] = c
    J[q, p] = s
    J[p, q] = -s
    J[q, q] = c
    ssum = th.abs(s)
    return J, ssum


def jade_par(A, threshold=10e-50, maxiter=1000):
    """
    Performs a parallel joint approximate diagonalization of tensor A and
    accumulates the necessary rotations in matrix V.

    :param A: torch tensor of shape (k x m x m)
    :param threshold: float as stopping criterion
    :param maxiter: int determining maximum iterations
    :return: A (k x m x m), V (m x m), n_iter
    """
    A = A.clone()  # avoid override of original
    A, pad_flag = pad(A)  # pad if necessary
    m = A.shape[1]  # shape of matrices, m by m

    V = th.eye(m)  # init joint basis to identity
    tournament = th.randperm(m).reshape(2, m // 2)  # init tournament table

    # assign stopping criteria
    active = th.tensor(1, dtype=th.uint8)
    n_iter = 0

    while active == 1 and n_iter < maxiter:
        # matrix to set offdiag element to 0
        J, ssum = rotation_matrix(A, tournament)

        # apply rotation of offdiagonal
        A = J.transpose(1, 0) @ A @ J
        # collect rotation
        V = V @ J

        # schedule successive tournament table
        tournament = scheduler(tournament)

        # evaluate stopping criteria
        n_iter += 1
        if threshold is not None:
            active = th.sum(ssum) > threshold

        # verbose monitoring:
        if n_iter % 10 == 0:
            print(n_iter, offdiag(A))

    print(n_iter, 'of', maxiter, 'iterations')
    print('desired convergence met?', not (bool(active)))
    print('Frobenius Norm of Offdiagonal(A):', offdiag(A))

    # undo zero padding, if flag is set
    A = A[:, :m - pad_flag, :m - pad_flag]
    V = V[:m - pad_flag, :m - pad_flag]

    return A, V, n_iter

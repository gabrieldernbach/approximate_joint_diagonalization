#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import njit
from numba import prange


def offdiagonal_frobenius(A):
    """
    computes the frobenius norm of the off diagonal elements
    of the tensor A (k x m x m)

    :param A: ndarray of shape k x m x m
    :return: norm, the frobenius norm of the offdiagonal of A
    """
    mask = 1 - np.eye(A.shape[1])
    offdiag = A * mask
    norm = np.sqrt(np.mean(offdiag ** 2))
    return norm


def jade(A, threshold=10e-16):
    """
    computes the joint diagonal basis V (m x m) of a set of
    k square matrices provided in A (k x m x m).

    It returns the basis V as well as the remaining diagonal
    and possible residual terms of A.

    :param A: ndarray of shape k x m x m
    :param threshold: float used for stop
    :return: A (k x m x m) V (m x m)
    """
    A = np.copy(A)
    m = A.shape[1]
    V = np.eye(m)
    active = 1
    while active == 1:
        active = 0
        for p in range(0, m):
            for q in range(p + 1, m):
                # computation of rotations           
                vecp = A[:, p, p] - A[:, q, q]
                vecm = A[:, p, q] + A[:, q, p]
                ton = vecp @ vecp - vecm @ vecm
                toff = 2 * vecp @ vecm
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                J = np.array([[c, s], [-s, c]])

                active = active | (np.abs(s) > threshold)
                # update of A and V matrices
                if abs(s) > threshold:
                    pair = np.array((p, q))
                    A[:, :, pair] = np.einsum('ij,klj->kli', J, A[:, :, pair])
                    A[:, pair, :] = np.einsum('ij,kjl->kil', J, A[:, pair, :])
                    V[:, pair] = np.einsum('ij,kj->ki', J, V[:, pair])
    return A, V


@njit
def jade_jit(A, threshold=10e-16):
    """
    computes the joint diagonal basis v (m x m) of a set of
    k square matrices provided in a (k x m x m).

    it returns the basis v as well as the remaining diagonal
    and possible residual terms of a.

    note: numba did not support einsum


    :param A: ndarray of shape k x m x m
    :param threshold: float used for stop
    :return: a (k x m x m) v (m x m)
    """
    A = np.copy(A)
    m = A.shape[1]
    k = A.shape[0]
    V = np.eye(m)
    active = 1
    while active == 1:
        active = 0
        for p in range(0, m):
            for q in range(p + 1, m):
                g = np.zeros((2, k))
                g[0, :] = A[:, p, p] - A[:, q, q]
                g[1, :] = A[:, p, q] + A[:, q, p]
                g = g @ g.T
                ton = g[0, 0] - g[1, 1]
                toff = g[0, 1] + g[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)

                active = active | (np.abs(s) > threshold)
                # update of A and V matrices
                if abs(s) > threshold:
                    colp = np.copy(A[:, :, p])
                    colq = np.copy(A[:, :, q])
                    A[:, :, p] = c * colp + s * colq
                    A[:, :, q] = c * colq - s * colp
                    rowp = np.copy(A[:, p, :])
                    rowq = np.copy(A[:, q, :])
                    A[:, p, :] = c * rowp + s * rowq
                    A[:, q, :] = c * rowq - s * rowp
                    temp = np.copy(V[:, p])
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * temp
    return A, V


# @njit
def tournament(top, bot, n):
    """
    computes the next round in the tournament.
    The n_th player of top will play against the n_th player
    of the group bottom.
    Rotation with fixed player at position 1 in top ensures
    that all players will have played against each other exactly
    one time during the complete tournament.

    :param top: previous players ids in group top
    :param bot: prvious player ids in group bottom
    :param n: number of players
    :return: new grouping for tournament oppositions
    """
    m = n // 2
    newtop = np.arange(m)
    newtop[2:] = top[1:-1]
    newtop[1] = bot[0]

    newbot = np.arange(m)
    newbot[:-1] = bot[1:]
    newbot[-1] = top[-1]
    return newtop, newbot

@njit
def pad(A):
    """
    if square shape of A (k x m x m) is odd - e.g. (m % 2) != 0
    add 0 padding to last row and last column of tensor A

    :param A: ndarray of shape k x m x m
    :return: A ndarray of shape (k x (m+1) x (m+1)) or (k x m x m),
             pad_flag bool that indicates whether padding was applied
    """
    if A.shape[1] % 2 is not 0:
        pad_flag = 1
        j, k = A.shape[0], A.shape[1]
        zers1 = np.zeros((j, k)).reshape((j, k, 1))
        A = np.concatenate((A, zers1), 2)
        zers2 = np.zeros((j, k + 1)).reshape((j, 1, k + 1))
        A = np.concatenate((A, zers2), 1)
    else:
        pad_flag = 0
    return A, pad_flag


@njit
def jade_parallel(A, threshold=10e-12):
    """
    Parallel implementation of joint approximate diagonalization

    :param A: ndarray of shape k x m x m
    :param threshold: thershold for stopping (is more conservative as non parallel!)
    :return: A (k x m x m), V (m x m)
    """
    A = np.copy(A)
    A, pad_flag = pad(A)
    m = A.shape[1]  # shape of matrix
    n = A.shape[0]  # number of matrices
    V = np.eye(m)  # initialize start

    top = np.arange(0, m, 2)
    bot = np.arange(1, m, 2)

    threshold = threshold * 100
    active = 1
    maxiter = 100
    n_iter = 0
    while active == 1 and n_iter < maxiter:
        active = 0
        n_iter += 1
        for sweep in range(m - 1):
            J = np.zeros((m, m))
            ssum = 0
            for k in prange(m // 2):
                p = min(top[k], bot[k])
                q = max(top[k], bot[k])

                vecp = A[:, p, p] - A[:, q, q]
                vecm = A[:, p, q] + A[:, q, p]
                ton = vecp @ vecp - vecm @ vecm
                toff = 2 * vecp @ vecm
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)

                J[p, p] = c
                J[q, p] = s
                J[p, q] = -s
                J[q, q] = c
                ssum += s ** 2

            active = active | ((ssum / m) > threshold)
            # update of A and V matrices

            for i in prange(n):
                A[i] = J.T @ A[i] @ J
            V = V @ J
            top, bot = tournament(top, bot, m)

    A = A[:, :m - pad_flag, :m - pad_flag]
    V = V[:m - pad_flag, :m - pad_flag]
    return A, V

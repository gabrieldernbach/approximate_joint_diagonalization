#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import njit

def jade(A, threshold=np.sqrt(np.spacing(1))):
    A = np.copy(A)
    m = A.shape[1]
    k = A.shape[0]
    V = np.eye(m)
    threshold = np.sqrt(np.spacing(1))
    active = 1
    while active == 1:
        active = 0
        for p in range(0, m):
            for q in range(p + 1, m):
                # computation of rotations           
                vecp = A[:,p,p] - A[:,q,q]
                vecm = A[:,p,q] + A[:,q,p]
                ton = vecp@vecp - vecm@vecm
                toff = 2 * vecp@vecm
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                J = np.array([[c,s],[-s,c]])

                active = active | (np.abs(s) > threshold)
                # update of A and V matrices
                if (abs(s) > threshold):
                    pair = np.array((p,q))
                    A[:,:,pair] = np.einsum('ij,klj->kli',J,A[:,:,pair])
                    A[:,pair,:] = np.einsum('ij,kjl->kil',J,A[:,pair,:])
                    V[:,pair] = np.einsum('ij,kj->ki',J,V[:,pair])
    return A, V


# numba ready python implementation of cardosos matlab
def jade_cardoso(A, threshold=1e-12):
    A = np.copy(A)
    m = A.shape[1]
    k = A.shape[0]
    V = np.eye(m)
    threshold = np.sqrt(np.spacing(1))
    active = 1
    while active == 1:
        active = 0
        for p in range(0, m):
            for q in range(p + 1, m):
                # computation of rotations
                g = np.zeros((2, k))
                g[0, :] = A[:, p, p] - A[:, q, q]
                g[1, :] = A[:, p, q] + A[:, q, p]
                g = g @ g.T
                ton = g[0, 0] - g[1, 1];
                toff = g[0, 1] + g[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta);
                s = np.sin(theta);

                active = active | (np.abs(s) > threshold)
                # update of A and V matrices
                if (abs(s) > threshold):
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


jade_precompiled_jit = njit("Tuple((float64[:,:,:],float64[:,:]))(float64[:,:,:],float64)")(jade_cardoso)

@njit
def jadejit(A, threshold=np.sqrt(np.spacing(1))):
    A = np.copy(A)
    m = A.shape[1]
    k = A.shape[0]
    V = np.eye(m)
    threshold = np.sqrt(np.spacing(1))
    active = 1
    while active == 1:
        active = 0
        for p in range(0, m):
            for q in range(p + 1, m):
                # computation of rotations
                g = np.zeros((2, k))
                g[0, :] = A[:, p, p] - A[:, q, q]
                g[1, :] = A[:, p, q] + A[:, q, p]
                g = g @ g.T
                ton = g[0, 0] - g[1, 1];
                toff = g[0, 1] + g[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta);
                s = np.sin(theta);

                active = active | (np.abs(s) > threshold)
                # update of A and V matrices
                if (abs(s) > threshold):
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

from numba import njit,prange

@njit
def tournament(top, bot, n):
    m = n // 2
    newtop = np.arange(m)
    newtop[2:] = top[1:-1]
    newtop[1] = bot[0]

    newbot = np.arange(m)
    newbot[:-1] = bot[1:]
    newbot[-1] = top[-1]
    return newtop, newbot

@njit
def offdiag(A):
    mask = np.eye(A.shape[1])
    mask = (mask==0)*1.0
    offdiag = A * mask
    norm = np.sum(offdiag)
    return norm

@njit(parallel=True,nogil=True)
def parjade(A, thres=1.0E-12):

    A = np.copy(A)
    m = A.shape[1] # shape of matrix
    n = A.shape[0] # number of matrices
    V = np.eye(m) # initialize start

    top = np.arange(0, m, 2)
    bot = np.arange(1, m, 2)

    threshold = thres*100
    active = 1
    maxiter = 100
    n_iter = 0
    while active == 1 and n_iter < maxiter:
        active = 0
        n_iter += 1
        for sweep in range(m-1):
            J = np.zeros((m,m))
            ssum = 0 
            for k in prange(m // 2):
                p = min(top[k],bot[k])
                q = max(top[k],bot[k])

                vecp = A[:,p,p] - A[:,q,q]
                vecm = A[:,p,q] + A[:,q,p]
                ton = vecp@vecp - vecm@vecm
                toff = 2 * vecp@vecm
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c = np.cos(theta)
                s = np.sin(theta)
                
                J[p,p] = c
                J[q,p] = s
                J[p,q] = -s
                J[q,q] = c
                ssum += s**2
                
            active = active | ((ssum/m) > threshold)
            # update of A and V matrices
            
            for i in prange(n):
                A[i,:,:] = J.T @ A[i,:,:] @ J
            V = V @ J
            top, bot = tournament(top, bot, m)
    return A, V, n_iter

if __name__ == '__main__':
    m = 10
    k = 10
    U,S,V = np.linalg.svd(np.random.randn(m,m))
    rand = np.random.randn(k,m,m)
    diag = np.eye(m,m)
    randdiag = diag[None,:,:] * rand
    M = U[None,:,:] @ randdiag @ U.T[None,:,:]

    A,V = jade(M)
    print(A,V)

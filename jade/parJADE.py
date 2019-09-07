#!/Usr/bin/env python
# coding: utf-8
import torch as th
import torch


def gentest(m=100, k=100, noise=0):
    '''k matrices of m by m size, normal noise multiplied by noise=0'''
    U, S, V = th.svd(th.randn(m, m))
    rand = th.randn(k, m, m)
    diag = th.eye(m, m)
    diag = diag[None, :, :] * rand
    M = U[None, :, :] @ diag @ U.transpose(1, 0)[None, :, :] + rand*noise
    print('testcase shape: ', M.shape, '\n has size: ', memsize(M))
    return M

import math
def memsize(DataTensor):
    size_bytes = DataTensor.element_size() * DataTensor.nelement()
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# JADE utility functions
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

def pad(A):
    if A.shape[1] %2 is not 0:
        pad_flag = 1
        j,k = A.shape[0], A.shape[1]
        zers1 = th.zeros((j, k)).reshape(j, k, 1)
        A = th.cat((A, zers1), 2)
        zers2 = th.zeros((j, k+1)).reshape(j, 1, k+1)
        A = th.cat((A, zers2), 1)
    else:
        pad_flag = 0
    return A, pad_flag

def offdiag(A):
    mask = ((th.eye(A.shape[1]))==0).type(th.float)
    offdiag = A * mask
    norm = th.sqrt(th.sum(offdiag**2))
    return norm


def rotmat(A, tournament):
    m = A.shape[1]
    J = th.zeros((m, m))

    p = tournament.min(dim=0)[0]
    q = tournament.max(dim=0)[0]

    matp = A[:, p, p] - A[:, q, q]
    matm = A[:, p, q] + A[:, q, p]
    ton = th.sum(matp*matp, dim=0) - th.sum(matm*matm, dim=0)
    toff = 2 * th.sum(matp*matm, dim=0)
    theta = 0.5 * th.atan2(toff, ton + th.sqrt(ton * ton + toff * toff))
    c = th.cos(theta)
    s = th.sin(theta)

    J[p, p] = c
    J[q, p] = s
    J[p, q] = -s
    J[q, q] = c
    ssum = s**2
    return J, ssum


def parjade(A, thres=1.0E-12, maxiter=1000):

    A = A.clone() # avoid override of original
    A, pad_flag = pad(A) # pad if necessary
    m = A.shape[1] # shape of matrices, m by m
    
    V = th.eye(m) # init joint basis to identity
    tournament = th.randperm(m).reshape(2, m//2); # init tournament table
    
    # assign stopping criteria
    threshold = thres
    active = th.tensor(1, dtype=th.uint8)
    n_iter = 0
    
    while active == 1 and n_iter < maxiter:
        # matrix to set offdiag element to 0
        J, ssum = rotmat(A, tournament)

        # apply rotation of offdiagonal
        A = J.transpose(1, 0) @ A @ J
        # collect rotation
        V = V @ J
        
        # schedule successive tournament table
        tournament = scheduler(tournament)

        # evaluate stopping criteria
        n_iter += 1
        active = ((th.sum(ssum))/m) > threshold
        
        # verbose monitoring:
        if n_iter % 100 == 0:
            print(n_iter, offdiag(A))

    print(n_iter, 'of', maxiter, 'iterations')
    print('reached convergence threshold:', not(bool(active)))
    print('Frobenius Norm of Offdiagonal(A):', offdiag(A))

    # undo zero padding, if flag is set
    A = A[:, :m-pad_flag, :m-pad_flag]
    V = V[:m-pad_flag, :m-pad_flag]
    
    return A, V, n_iter

if __name__ == '__main__':
        
    th.set_default_tensor_type('torch.cuda.FloatTensor')
    device = th.device("cuda") 
    th.cuda.get_device_name(device)
    M = gentest(100, 100, 0)

    A, V, n_iter = parjade(M)

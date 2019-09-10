#!/Usr/bin/env python
# coding: utf-8

import torch as th
import torch

def mean_rotation(C):
    C_mean = th.mean(C, dim=0)
    vals, vecs = th.symeig(C_mean)
    B = vecs.T / th.sqrt(vals[:, None])
    C = B[None,:,:] @ M @ B.T[None,:,:]
    return B, C


def init_tournament(m):
    '''initialize random tournament table with pairwise groups'''
    if m % 2 == 0:
        tournament = th.randperm(m).reshape(2, m//2)
        padflag = 0
    else:
        tournament = th.randperm(m)
        m0 = torch.tensor(m).view(1)
        tournament = th.cat((m0,tournament)).reshape(2, (m+1)//2)
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


def rotmat(C,tournament, padflag):
    '''
    compute update matrix according to phams method see:
    D. T. Pham, “Joint Approximate Diagonalization of Positive Definite Hermitian Matrices,”
    SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 4, pp. 1136–1152, Jan. 2001.
    '''

    m = C.shape[1]
    k = C.shape[0]

    i = tournament[:,padflag:].min(dim=0)[0]
    j = tournament[:,padflag:].max(dim=0)[0]

    C_ii = C[:, i, i] 
    C_jj = C[:, j, j]
    C_ij = C[:, i, j]

    # find g_ij (2.04)
    g_ij = th.mean(C_ij / C_ii, dim=0)
    g_ji = th.mean(C_ij / C_jj, dim=0)
    g = th.stack([ g_ij, g_ji])

    # find w_ij (2.07) with w_ii, w_jj = 1, 1
    w_ij = th.mean(C_jj / C_ii, dim=0)
    w_ji = th.mean(C_ii / C_jj, dim=0)
    ones = th.ones(w_ij.shape)
    Wu = th.stack([w_ij, ones])
    Wl = th.stack([ones, w_ji])
    W = th.stack([Wu, Wl]).permute(2,0,1)

    
    # solve 2.10, that is find h such that W @ h = g
    U,S,V = th.svd(W)
    c = U.transpose(2,1) @ g.transpose(1,0)[:,:,None]
    y = 1/S * c.squeeze()
    h = (U @ y[:,:,None]).squeeze().transpose(1,0)
    h12, h21 = h[0], h[1]

    #y = 1/diag(U.transpose(1,0) @ g

   # h = V.transpose(0,1) @ 

    # w_tilde_ji = th.sqrt(w_ji / w_ij)
    # w_prod = th.sqrt(w_ij * w_ji)
    # tmp1 = (w_tilde_ji * g_ij + g_ji) / (w_prod + 1)
    # tmp2 = (w_tilde_ji * g_ij - g_ji) / th.max(w_prod - 1, th.tensor(1e-9).view(1)) 
    # h12 = tmp1 + tmp2 # (2.10)
    # w_tilde_ij = th.sqrt(w_ij / w_ji)
    # tmp1 = (w_tilde_ij * g_ji + g_ij) / (w_prod + 1)
    # tmp2 = (w_tilde_ij * g_ji - g_ij) / th.max(w_prod - 1, th.tensor(1e-9).view(1)) 
    # h21 = tmp1 - tmp2 / w_tilde_ji # h21 = th.conj((tmp1 - tmp2) / w_tilde_ji)

    # cumulative decrease in current sweep
    # decrease = th.sum(k * (g_ij * th.conj(h12) + g_ji * h21) / 2.0)
    decrease = th.sum(k * (g_ij * h12 + g_ji * h21) / 2.0)
    

    # construct T by 2.08
    # tmp = 1 + 1.j * 0.5 * th.imag(h12 * h21)
    # tmp = th.real(tmp + th.sqrt(tmp ** 2 - h12 * h21))
    tmp = 2 / (1 + th.sqrt(1 - 4 * h12 * h21))

    T = th.eye(m)
    T[i, j] = -h12 #/ tmp
    T[j, i] = -h21 #/ tmp

    return T, decrease


def phams(Gamma, threshold=1e-50, maxiter=1000, mean_initialize=False):
    '''
    find approximate joint diagonalization of set of square matrices Gamma,
    returns joint basis B and corresponding set of approximate diagonals C
    '''
    C = Gamma.clone()
    m = C.shape[1]
    B = th.eye(m)

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

        # update of C and B matrices
        C = T @ C @ T.transpose(0,1)
        B = T @ B

        tournament = scheduler(tournament)
        n_iter += 1
        active = th.abs(decrease) > threshold

    return B, C 

if __name__ == '__main__':

    """Test approximate joint diagonalization."""
    # create k matrices of shape m x m
    k, m = 20, 40

    rng = th.manual_seed(42) 
    
    rand = th.abs(th.rand(k, m, m))
    diag = th.eye(m, m)
    diag = diag[None, :, :] * rand
    B = th.randn(m, m)
    M = B[None, : ,:] @ diag @ B.transpose(1, 0)[None, :, :]
    
    Bhat, _ = phams(M)

    
    # check if B and Bhat are identical up to permutation and scaling
    BA = th.abs(Bhat @ B)  # undo negative scaling 
    BA /= th.max(BA, dim=1, keepdims=True)[0] # normalize to 1
    BA[th.abs(BA) < 1e-12] = 0. # numerical tolerance
    print(BA)
    import matplotlib.pyplot as plt
    plt.imshow(BA @ BA.T)
    plt.show()
    
    from numpy.testing import assert_array_equal
    import numpy as np
    BA = BA.numpy()
    assert_array_equal(BA[np.lexsort(BA)], np.eye(m))

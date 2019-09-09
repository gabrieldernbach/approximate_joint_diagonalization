from time import time
import numpy as np


def transform_set(M, D):
    # apply matrix M to first dim of D
    K, N, _ = D.shape
    op = np.zeros((K, N, N)) 
    for i, d in enumerate(D): # enumerate goes along 1st dimension (aka batches)
        op[i] = M.dot(d.dot(M.T))
    return op

def phams(A, threshold=np.sqrt(np.spacing(1))):
    '''
    einsum based implementation of jade algorithm
    '''
    A = np.copy(A)
    m = A.shape[1]
    k = A.shape[0]
    V = np.eye(m)
    A_mean = np.mean(A, axis=0)
    vals, vecs = np.linalg.eigh(A_mean)
    V = vecs.T / np.sqrt(vals[:, None])
    A = transform_set(V, A)

    threshold = np.sqrt(np.spacing(1))
    active = 1
    while active == 1:
        active = 0
        decr = 0
        for i in range(0, m):
            for j in range(0, i):
                # computation of rotations           

                Cii = A[:, i, i] 
                Cjj = A[:, j, j]
                Cij = A[:, i, j]

                # find g_ij (2.4)
                g12 = np.mean(Cij / Cii)
                g21 = np.mean(Cij / Cjj)

                # find w_ij (2.7)
                omega21 = np.mean(Cii / Cjj)
                omega12 = np.mean(Cjj / Cii)
                omega = np.sqrt(omega12 * omega21)

                # solve 2.10 
                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1) # wji * gij + gbar_ji
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9) # (2.10 with max, see numerical consideration)
                h12 = tmp1 + tmp2 # (2.10) sum
                h21 = np.conj((tmp1 - tmp2) / tmp) # actually says to change the indexes, is equivalent?

                decr += k * (g12 * np.conj(h12) + g21 * h21) / 2.0
                print(decr)

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                J = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]]) # stacks all scalar values

                active = active | (np.abs(decr) > threshold)
                # update of A and V matrices
                if (abs(decr) > threshold):
                    pair = np.array((i,j))
                    A[:,:,pair] = np.einsum('ij,klj->kli',J,A[:,:,pair])
                    A[:,pair,:] = np.einsum('ij,kjl->kil',J,A[:,pair,:])
                    V[pair,:] = J @ V[pair,:]
    return V, A 


if __name__ == '__main__':
    from numpy.testing import assert_array_equal

    """Test approximate joint diagonalization."""
    n, p = 10, 3
    rng = np.random.RandomState(42)
    
    diagonals = rng.uniform(size=(n, p))
    V = rng.randn(p, p)  # mixing matrix
    M = np.array([V.dot(d[:, None] * V.T) for d in diagonals])  # dataset
    Vhat, _ = phams(M)

    # check if V and Vhat are identical up to permutation and scaling
    VA = np.abs(Vhat.dot(V))  # undo negative scaling 
    VA /= np.max(VA, axis=1, keepdims=True) # normalize to 1
    VA[np.abs(VA) < 1e-12] = 0. # numerical tolerance
    print(VA)
    assert_array_equal(VA[np.lexsort(VA)], np.eye(p))

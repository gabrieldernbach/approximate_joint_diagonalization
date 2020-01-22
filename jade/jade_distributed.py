import os
import math
import torch as th
from torch.multiprocessing import Process, Manager
import torch.distributed as dist

def gentest(m=100, k=100, noise=0):
    '''k matrices of m by m size, normal noise multiplied by noise=0'''
    print('prepare joint basis (symmetric orthogonal)')
    U, S, V = th.svd(th.randn(m, m))
    rand = th.randn(k, m, m)
    print('build random diagonal')
    diag = th.unsqueeze(th.eye(m, m), 0) * rand
    print('rotate random diagonal by joint basis')
    M = th.unsqueeze(U, 0) @ diag @ th.unsqueeze(U.transpose(1, 0), 0) + rand*noise
    print(f'testcase of shape {M.shape} has size {memsize(M)}')
    return M

def memsize(M):
    ''' measure memory size of torch tensor M, human readable format'''
    size_bytes = M.element_size() * M.nelement()
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f'{s} {size_name[i]}'

# jade utils
def partition(M, rank):
    '''partition 'M' into 'rank' equal parts'''
    matN = M.shape[0]
    idx = th.arange(matN).reshape(rank, -1)
    Msplit = M[idx]
    print(f'dataset shape is: {Msplit.shape}')
    return Msplit

def tournament_scheduler(tournament):
    '''return next draw of tournament table'''
    old = tournament
    new = th.zeros(old.shape, dtype=th.int64)
    
    # players of row 0 will play ...
    new[0, 0] = old[0, 0] # keep player 0 fixed
    new[0, 1] = old[1, 0] # take first of row 1 to row 2
    new[0, 2:] = old[0, 1:-1] # put all others one to the right
    # ... against players of row 1
    new[1, -1] = old[0, -1]; # take last of row 1 to row 2
    new[1, :-1] = old[1, 1:] # put all others one to the left
    return new

def pad(A):
    '''pad A with 0 such that it's shape becomes an even integer'''
    pad_flag = A.shape[1] % 2
    if pad_flag:
        j, k, _ = A.shape
        zers1 = th.zeros((j, k, 1))
        A = th.cat((A, zers1), 2)
        zers2 = th.zeros((j, 1, k+1))
        A = th.cat((A, zers2), 1)
    return A, pad_flag

def offdiag(A, device):
    '''compute the forbenius norm of the off diagonal elements only'''
    mask = 1 - th.eye(A.shape[1], device=device)
    offdiag = A * mask # multiply the diagonal with zero
    norm = th.sqrt(th.mean(offdiag**2))
    return norm

def rotmat(A, tournament, device):
    '''determine a matrix to incrementally diagonalize A'''
    m = A.shape[1]
    J = th.zeros((m, m), device=device)

    p = tournament.min(dim=0)[0]
    q = tournament.max(dim=0)[0]

    matp = A[:, p, p] - A[:, q, q]
    matm = A[:, p, q] + A[:, q, p]
    ton = th.sum(matp*matp, dim=0) - th.sum(matm*matm, dim=0)
    toff = 2 * th.sum(matp*matm, dim=0)
    dist.all_reduce(ton, op=dist.ReduceOp.SUM)
    dist.all_reduce(toff, op=dist.ReduceOp.SUM)
    theta = 0.5 * th.atan2(toff, ton + th.sqrt(ton * ton + toff * toff))
    c = th.cos(theta)
    s = th.sin(theta)

    J[p, p] = c
    J[q, p] = s
    J[p, q] = -s
    J[q, q] = c
    ssum = s**2
    
    return J, ssum


def parjade(A, rank, solutions, **kwargs):

    maxiter = kwargs.get('maxiter', 1000)
    thres = kwargs.get('thres', 1.0E-12)
    verbose = kwargs.get('verbose', 2)
    seed = kwargs.get('seed', 1)

    th.set_default_tensor_type('torch.cuda.FloatTensor')
    device = th.device(f'cuda:{rank}') # select gpu
    A = A[rank].to(device) # take batch to device
    A, pad_flag = pad(A) # pad if necessary
    verbose > 1 and  print(f'par shape of A loaded on gpu {A.shape}')

    # initialize joint basis
    m = A.shape[1] # shape of matrices, m by m
    V = th.eye(m, device=device)

    # init tournament table
    th.manual_seed(seed) # identical seed necessary for parallel processing
    tournament = th.randperm(m).reshape(2, m//2);

    # assign stopping criteria
    threshold = thres
    active = th.tensor(1, dtype=th.uint8)
    n_iter = 0

    while active and n_iter < maxiter:
        # matrix to set offdiag element to 0
        J, ssum = rotmat(A, tournament, device)

        # apply zeroing of offdiagonal
        A = J.transpose(1, 0) @ A @ J
        # collect rotation
        V = V @ J

        # schedule successive tournament table
        tournament = tournament_scheduler(tournament)

        # evaluate stopping criteria
        n_iter += 1
        active = ((th.sum(ssum))/m) > threshold

        # verbose monitoring:
        if n_iter % 10 == 0:
            verbose > 1 and print(f'device: {rank} | iteration: {n_iter} | offdiag: {offdiag(A, device)}')

    verbose > 0 and print(f'process {rank}',
                            f'{n_iter} of maximum {maxiter} iterations',
                            f'reached threshold: {not(bool(active))}',
                            f'Norm of offdiag(A): {offdiag(A, device)}', sep='\n\t')

    # undo zero padding, if flag is set
    A = A[:, :m-pad_flag, :m-pad_flag]
    V = V[:m-pad_flag, :m-pad_flag]

    solutions['As'].append(A.cpu())
    solutions['Vs'].append(V.cpu())
    solutions['n_iter'].append(n_iter) #seems to live on cpu already?

# process management
def init_processes(M, rank, size, solutions, backend='gloo', *args, **kwargs):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    parjade(M, rank, solutions, *args, **kwargs)

def distributed_jade(M, world_size, *args, **kwargs):
    manager = Manager()
    solutions = manager.dict()
    solutions['As'] = manager.list()
    solutions['Vs'] = manager.list()
    solutions['n_iter'] = manager.list()

    processes = []
    for rank in range(world_size):
        processargs = [M, rank, world_size, solutions]+list(args)
        p = Process(target=init_processes, args=processargs, kwargs=kwargs)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    A = th.cat([i for i in solutions['As']], dim=0)
    V = th.stack([i for i in solutions['Vs']])
    n_iter = [i for i in solutions['n_iter']]
    
    return A, V, n_iter

# working example
if __name__ == "__main__":
    world_size = 2 # number of gpus
    mat_shape = 1000 # m x m matrix
    num_mats = 30
    noise = 0 # variance, gaussian

    M = gentest(mat_shape, num_mats, noise)
    M = partition(M, world_size)

    Mdiag, V, n_iter = distributed_jade(M, world_size, maxiter=50000)

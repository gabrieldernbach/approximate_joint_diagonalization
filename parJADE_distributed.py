import os
import math
import torch as th
from torch.multiprocessing import Process, Manager
import torch.distributed as dist

def gentest(m=100,k=100,noise=0):
    '''k matrices of m by m size, normal noise multiplied by noise=0'''
    print('create rotation matirx')
    U,S,V = th.svd(th.randn(m,m))
    rand = th.randn(k,m,m)
    print('build random diagonal')
    diag = th.eye(m,m)
    diag = diag[None,:,:] * rand
    print('rotate random diagonal by joint basis')
    M = U[None,:,:] @ diag @ U.transpose(1,0)[None,:,:] + rand*noise
    print('testcase of shape ',M.shape,' has size ',memsize(M))
    return M

def memsize(DataTensor):
    size_bytes = DataTensor.element_size() * DataTensor.nelement()
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

# jade utils
def partition(M,rank):
    '''partition M into 'rank' equal parts'''
    matN = M.shape[0]
    idx = th.arange(matN).reshape(rank,-1)
    Msplit = M[idx]
    print('dataset shape is: ',Msplit.shape)
    return Msplit

def scheduler(tournament):
    '''return next draw of tournament table'''
    old = tournament
    new = th.zeros(old.shape,dtype=th.int64)
    
    # players of row 0
    new[0,0] = old[0,0]
    new[0,1] = old[1,0]
    new[0,2:] = old[0,1:-1]
    # against players of row 1
    new[1,-1] = old[0,-1];
    new[1,:-1] = old[1,1:]
    return new

def pad(A):
    if A.shape[1] %2 is not 0:
        pad_flag = 1
        j,k = A.shape[0],A.shape[1]
        zers1 = th.zeros((j,k)).reshape(j,k,1)
        A = th.cat((A,zers1),2)
        zers2 = th.zeros((j,k+1)).reshape(j,1,k+1)
        A = th.cat((A,zers2),1)
    else:
        pad_flag = 0
    return A,pad_flag

def offdiag(A,device):
    mask = ((th.eye(A.shape[1],device=device))==0).type(th.float)
    offdiag = A * mask
    norm = th.sqrt(th.mean(offdiag**2))
    return norm

def rotmat(A,tournament,device):
    m = A.shape[1]
    J = th.zeros((m,m),device=device)

    p = tournament.min(dim=0)[0]
    q = tournament.max(dim=0)[0]

    matp = A[:,p,p] - A[:,q,q]
    matm = A[:,p,q] + A[:,q,p]
    ton = th.sum(matp*matp,dim=0) - th.sum(matm*matm,dim=0)
    toff = 2 * th.sum(matp*matm,dim=0)
    dist.all_reduce(ton,op=dist.ReduceOp.SUM)
    dist.all_reduce(toff,op=dist.ReduceOp.SUM)
    theta = 0.5 * th.atan2(toff, ton + th.sqrt(ton * ton + toff * toff))
    c = th.cos(theta)
    s = th.sin(theta)

    J[p,p] = c
    J[q,p] = s
    J[p,q] = -s
    J[q,q] = c
    ssum = s**2
    
    return J,ssum


def parjade(A, rank, solutions, thres=1.0E-12, maxiter=1000,verbose=2,seed=1):

    th.set_default_tensor_type('torch.cuda.FloatTensor')
    device = th.device('cuda:{}'.format(rank)) # select gpu
    A = A[rank].to(device) # take batch to device
    A,pad_flag = pad(A) # pad if necessary
    print('par shape of A loaded on gpu',A.shape)

    # init joint basis to identity
    m = A.shape[1] # shape of matrices, m by m
    V = th.eye(m,device=device)

    # init tournament table
    th.manual_seed(seed) # identical seed necessary for parallel processing
    tournament = th.randperm(m).reshape(2,m//2);

    # assign stopping criteria
    threshold = thres
    active = th.tensor(1,dtype=th.uint8)
    n_iter = 0

    while active == 1 and n_iter < maxiter:
        # matrix to set offdiag element to 0
        J,ssum = rotmat(A,tournament,device)

        # apply zeroing of offdiagonal
        A = J.transpose(1,0) @ A @ J
        # collect rotation
        V = V @ J

        # schedule successive tournament table
        tournament = scheduler(tournament)

        # evaluate stopping criteria
        n_iter += 1
        #active = ((th.sum(ssum))/m) > threshold

        # verbose monitoring:
        if verbose > 1 and n_iter % 10 == 0:
            print('iteration:',n_iter,' offdiag:', offdiag(A,device))

    if verbose > 0:
        print(n_iter,'of',maxiter,'iterations')
        print('reached threshold:',not(bool(active)))
        print('Frob of A:', offdiag(A,device))

    # undo zero padding, if flag is set
    A = A[:,:m-pad_flag,:m-pad_flag]
    V = V[:m-pad_flag,:m-pad_flag]

    solutions['As'].append(A.cpu())
    solutions['Vs'].append(V.cpu())
    solutions['n_iter'].append(n_iter) #seems to live on cpu already?

# process management
def init_processes(M,rank,size,solutions,backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    parjade(M, rank, solutions)

def distributed_jade(M,world_size):
    manager = Manager()
    solutions = manager.dict()
    solutions['As'] = manager.list()
    solutions['Vs'] = manager.list()
    solutions['n_iter'] = manager.list()

    processes = []
    for rank in range(world_size):
        p = Process(target=init_processes, args=(M,rank,world_size,solutions))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    A = th.cat([i for i in solutions['As']],dim=0)
    V = th.stack([i for i in solutions['Vs']])
    n_iter = [i for i in solutions['n_iter']]
    
    return A, V, n_iter

# working example
if __name__ == "__main__":
    world_size = 2 # number of gpus
    mat_shape = 100 # m x m matrix
    num_mats = 2000
    noise = 0 # variance, gaussian

    M = gentest(mat_shape,num_mats,noise)
    M = partition(M,world_size)

    Mdiag,V,n_iter = distributed_jade(M,world_size)

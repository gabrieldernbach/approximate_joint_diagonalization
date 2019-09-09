# Parallel Approximate Joint Diagonalization

GPU ready python implementations that scale to multiple GPUs.
By distributed calculation huge datasets can be computed in a scalable cloud environment.

## Getting Started

### Prerequisites

Installation of numpy, numba and pytorch.
Numpy and Numba versions are provided as base CPU implementation.

### Running the tests

For convenience appropriate tests are given in the main section of the modules


## Available methods

Available methods are JADE [1] and Phams [2].

[1] Cardoso, Jean-François, and Antoine Souloumiac. "Jacobi angles for simultaneous diagonalization." SIAM journal on matrix analysis and applications 17.1 (1996): 161-164.

[2] D. T. Pham, “Joint Approximate Diagonalization of Positive Definite Hermitian Matrices,” SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 4, pp. 1136–1152, Jan. 2001.

### CPU version
CPU implementations are based on numpy.
Parallelization and just in time compilation are realized with numba.
The code was modified to work with the library accordingly


### GPU versions
GPU versions with parallelization are written in pytorch.
Distribution among several GPUs are managed by the pytorch wrapper.

## Acknowledgments

* Dr. Andreas Ziehe for recommendation of base implementations and general support
* Panagiotis Karagiannis and Stefaan Hessmann for ongoing code review

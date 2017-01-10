CloverLeaf3D_ref
==============

The reference version of CloverLeaf3D

This repo is forked from the 2D CloverLeaf reference version at https://github.com/Warwick-PCAV/CloverLeaf_ref from version 1.1.


# Build Instructions

These remain the same as CloverLeaf:
In many case just typing make in the required software directory will work. This is the case if the mpif90 and mpicc wrappers are available on the system. This is true even for the Serial and OpenMP versions.

If the MPI compilers have different names then the build process needs to notified of this by defining two environment variables, `MPI_COMPILER` and `C_MPI_COMPILER`.

For example on some Intel systems:
```
make MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc
```
Or on Cray systems:
```
make MPI_COMPILER=ftn C_MPI_COMPILER=cc
```


##################################
# uberenv host-config
##################################
# chaos_5_x86_64_ib-gcc@4.9.3
##################################

# cmake from uberenv
# cmake exectuable path: /usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/cmake-3.8.2-cjuiep5cinfy7q6khr7hpsb3utzs34cd/bin/cmake

#######
# using gcc@4.9.3 compiler spec
#######

# c compiler used by spack
set("CMAKE_C_COMPILER" "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")

# cpp compiler used by spack
set("CMAKE_CXX_COMPILER" "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

# fortran compiler used by spack
set("ENABLE_FORTRAN" "ON" CACHE BOOL "")

set("CMAKE_Fortran_COMPILER" "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")

# Enable python module builds
set("ENABLE_PYTHON" "ON" CACHE BOOL "")

# python from uberenv
set("PYTHON_EXECUTABLE" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.11-eujx7frnxd5vpwolmye2fzq4tcylnbnv/bin/python" CACHE PATH "")

# sphinx from uberenv
set("SPHINX_EXECUTABLE" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.11-eujx7frnxd5vpwolmye2fzq4tcylnbnv/bin/sphinx-build" CACHE PATH "")

# OPENMP Support
set("ENABLE_OPENMP" "OFF" CACHE BOOL "")

# MPI Support
set("ENABLE_MPI" "ON" CACHE BOOL "")

set("MPI_C_COMPILER" "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set("MPI_CXX_COMPILER" "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set("MPI_Fortran_COMPILER" "/usr/local/tools/mvapich2-gnu-2.0/bin/mpif90" CACHE PATH "")

# CUDA support
set("ENABLE_CUDA" "ON" CACHE BOOL "")

set("CUDA_BIN_DIR" "/opt/cudatoolkit-8.0/bin" CACHE PATH "")

# conduit from uberenv
set("CONDUIT_DIR" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/conduit-0.2.1-nxbg2ahgeptrlrwcyifkdh2smekgrgvi" CACHE PATH "")


# hdf5 from uberenv
set("HDF5_DIR" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/hdf5-1.8.17-uftiwrdaelniei2m65p6shkjji66pxqp" CACHE PATH "")



# vtkm support

# tbb from uberenv
set("TBB_DIR" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/tbb-4.4.3-d6bja3hkb6ds7iuojibtwaxs5vn5cmyp" CACHE PATH "")

# vtkm from uberenv
set("VTKM_DIR" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/vtkm-kitware-gitlab-hyuwnlq4rqlk6dgf5yzfnuzutfxn4lsk" CACHE PATH "")

# icet from uberenv
set("ICET_DIR" "/usr/workspace/wsa/visit/alpine/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/icet-icet-master-p3efsnxqmcqxvrl7sykctgmowroczxrj" CACHE PATH "")


##################################
# end uberenv host-config
##################################

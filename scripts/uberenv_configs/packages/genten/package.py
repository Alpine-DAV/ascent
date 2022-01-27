from spack import *


class Genten(CMakePackage,CudaPackage):
    """Genten: Software for Generalized Tensor Decompositions by Sandia National Laboratories"""

    homepage = "https://gitlab.com/tensors/genten"

    maintainers = ['mclarsen']

    version('master',
#            git='https://gitlab.com/tensors/genten.git',
            git='https://github.com/mclarsen/genten.git',
            branch='higher-moments-interface',
            submodules=False,
            preferred=True)

    variant("shared", default=True, description="Build shared libs")
    variant("openmp", default=True, description="Build openmp support")
    variant("cuda", default=False, description="Build cuda")

    depends_on('blas', when='~cuda')
    depends_on('lapack', when='~cuda')
    depends_on('kokkos+openmp', when='+openmp~cuda')
    depends_on('kokkos+cuda+cuda_lambda', when='~openmp+cuda')
    depends_on('kokkos', when='~openmp~cuda')

    def cmake_args(self):
      args = []

      if '+shared' in self.spec:
          args.append('-DBUILD_SHARED_LIBS=ON')
      else:
          args.append('-DBUILD_SHARED_LIBS=OFF')

      args.append("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
      args.append("-DKOKKOS_PATH={0}".format(self.spec['kokkos'].prefix))
#      if '+cuda' in self.spec:
#        args.append("-DKokkos_ENABLE_CUDA=ON")

      if '~cuda' in self.spec:
        lapack_blas = self.spec['lapack'].libs + self.spec['blas'].libs
        lapack_blas_flags = '-DLAPACK_LIBS=%s' % ';'.join(lapack_blas.libraries)
        args.append(lapack_blas_flags)

      if '+cuda' in spec:
          cublas_lib = find_libraries("libcublas", spec['cuda'].libs.directories[0],
                                      shared='+shared' in spec, recursive=False)

          cusolver_lib = find_libraries("libcusolver", spec['cuda'].libs.directories[0],
                                        shared='+shared' in spec, recursive=False)
          #these are the names used in genten's cmake build system
          args.append('-DCUDA_CUBLAS_LIBRARIES="{0}'.format(cublas_lib))
          args.append('-DCUDA_cusolver_LIBRARY="{0}'.format(cusolver_lib))


      return args

    def cmake_install(self, spec, prefix):
        make()
        make('install')

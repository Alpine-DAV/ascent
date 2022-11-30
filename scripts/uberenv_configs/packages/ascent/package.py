# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import glob
import os
import shutil
import socket
import sys
from os import environ as env

import llnl.util.tty as tty

from spack import *


def cmake_cache_entry(name, value, vtype=None):
    """
    Helper that creates CMake cache entry strings used in
    'host-config' files.
    """
    if vtype is None:
        if value == "ON" or value == "OFF":
            vtype = "BOOL"
        else:
            vtype = "PATH"
    return 'set({0} "{1}" CACHE {2} "")\n\n'.format(name, value, vtype)


class Ascent(CMakePackage, CudaPackage, ROCmPackage):
    """Ascent is an open source many-core capable lightweight in situ
    visualization and analysis infrastructure for multi-physics HPC
    simulations."""

    homepage = "https://github.com/Alpine-DAV/ascent"
    git      = "https://github.com/Alpine-DAV/ascent.git"
    url      = "https://github.com/Alpine-DAV/ascent/releases/download/v0.5.1/ascent-v0.5.1-src-with-blt.tar.gz"

    maintainers = ['cyrush']

    version('develop',
            branch='develop',
            submodules=True)

    version('0.8.0',
            tag='v0.8.0',
            submodules=True,
            preferred=True)

    version('0.7.1',
            tag='v0.7.1',
            submodules=True)

    version('0.7.0',
            tag='v0.7.0',
            submodules=True)

    version('0.6.0',
            tag='v0.6.0',
            submodules=True)

    ###########################################################################
    # package variants
    ###########################################################################

    variant("shared", default=True, description="Build Ascent as shared libs")
    variant('test', default=True, description='Enable Ascent unit tests')

    variant("mpi", default=True, description="Build Ascent MPI Support")
    # set to false for systems that implicitly link mpi
    variant('blt_find_mpi', default=True, description='Use BLT CMake Find MPI logic')
    variant("serial", default=True, description="build serial (non-mpi) libraries")

    # variants for language support
    variant("python", default=False, description="Build Ascent Python support")
    variant("fortran", default=True, description="Build Ascent Fortran support")

    # variants for runtime features
    variant("vtkh", default=True,
            description="Build VTK-h filter and rendering support")

    variant("openmp", default=(sys.platform != 'darwin'),
            description="build openmp support")
    variant("mfem", default=False, description="Build MFEM filter support")
    variant("adios", default=False, description="Build Adios filter support")
    variant("dray", default=False, description="Build with Devil Ray support")
    variant("adios2", default=False, description="Build Adios2 filter support")
    variant("fides", default=False, description="Build Fides filter support")
    variant("genten", default=False, description="Build with GenTen support")
    variant("occa", default=False, description="Build with OCCA support")
    variant("raja", default=True, description="Build with RAJA support")
    variant("umpire", default=True, description="Build with Umpire support")
    variant("babelflow", default=False, description="Build with BabelFlow")

    # variants for dev-tools (docs, etc)
    variant("doc", default=False, description="Build Ascent's documentation")

    ##########################################################################
    # package dependencies
    ###########################################################################

    # Certain CMake versions have been found to break for our use cases
    depends_on("cmake@3.14.1:3.14.99,3.18.2:", type='build')
    # NOTE: With Old CONCRETIZER, dep on conduit with no variants
    # causes a conflict (since conduit defailts python o off)
    #depends_on("conduit")
    depends_on("conduit~python", when="~python")
    depends_on("conduit+python", when="+python")
    depends_on("conduit+fortran", when="+fortran")
    depends_on("conduit+mpi", when="+mpi")
    depends_on("conduit~mpi", when="~mpi")

    #######################
    # Python
    #######################
    # we need a shared version of python b/c linking with static python lib
    # causes duplicate state issues when running compiled python modules.
    with when('+python'):
        depends_on("python+shared")
        extends("python")
        depends_on("py-numpy", type=('build', 'run'))
        depends_on("py-pip", type=('build', 'run'))

    #######################
    # MPI
    #######################
    depends_on("mpi", when="+mpi")
    depends_on("py-mpi4py", when="+mpi+python")

    #############################
    # RAJA
    #############################
    depends_on("raja", when="+raja")

    depends_on("raja+rocm", when="+raja+rocm")

    # Propagate AMD GPU target to raja for +rocm
    for amdgpu_value in ROCmPackage.amdgpu_targets:
        depends_on("raja amdgpu_target=%s" % amdgpu_value, when="+raja+rocm amdgpu_target=%s" % amdgpu_value)

    # TODO: do we need all of these?
    depends_on("raja+cuda+shared", when="+cuda+shared")
    depends_on("raja+cuda~shared", when="+cuda~shared")
    depends_on("raja~cuda+shared", when="~cuda+shared")
    depends_on("raja~cuda~shared", when="~cuda~shared")


    #############################
    # Umpire
    #############################
    depends_on("umpire", when="+umpire")
    depends_on("umpire+rocm", when="+umpire+rocm")

    # Propagate AMD GPU target to umpire for +rocm
    for amdgpu_value in ROCmPackage.amdgpu_targets:
        depends_on("umpire amdgpu_target=%s" % amdgpu_value, when="+umpire+rocm amdgpu_target=%s" % amdgpu_value)

    # TODO: do we need all of these?
    depends_on("umpire+cuda+shared", when="+umpire+cuda+shared")
    depends_on("umpire+cuda~shared", when="+umpire+cuda~shared")
    depends_on("umpire~cuda+shared", when="+umpire~cuda+shared")
    depends_on("umpire~cuda~shared", when="+umpire~cuda~shared")

    #############################
    # HIP
    #############################
    depends_on("hip", when="+rocm")

    #############################
    # VTK-m + VTK-h
    #############################
    depends_on("raja", when="+raja")

    # ascent newer than 0.8.0 uses internal vtk-h
    # use vtk-m 1.8 for newer than ascent 0.8.0
    depends_on("vtk-m@1.8:", when="@0.8.1:")

    depends_on("vtk-m~tbb", when="@0.8.1: +vtkh")
    depends_on("vtk-m+openmp", when="@0.8.1: +vtkh+openmp")
    depends_on("vtk-m~openmp", when="@0.8.1: +vtkh~openmp")

    depends_on("vtk-m+openmp", when="@0.8.1: +vtkh+openmp")
    depends_on("vtk-m~openmp", when="@0.8.1: +vtkh~openmp")

    depends_on("vtk-m~cuda", when="@0.8.1: +vtkh~cuda")
    depends_on("vtk-m+cuda", when="@0.8.1: +vtkh+cuda")
    for _arch in CudaPackage.cuda_arch_values:
        depends_on("vtk-m+cuda cuda_arch={0}".format(_arch), when="@0.8.1: +cuda+openmp cuda_arch={0}".format(_arch))

    depends_on("vtk-m+fpic", when="@0.8.0: +vtkh")
    depends_on("vtk-m~shared+fpic", when="@0.8.0: +vtkh~shared")
    
    depends_on("vtk-m+rocm", when="+rocm")

    # Propagate AMD GPU target to vtk-m for +rocm
    for amdgpu_value in ROCmPackage.amdgpu_targets:
        depends_on("vtk-m amdgpu_target=%s" % amdgpu_value, when="+rocm amdgpu_target=%s" % amdgpu_value)

    # use external vtk-h for 0.8.0 and older
    depends_on("vtk-h",      when="@:0.8.0 +vtkh")
    depends_on("vtk-h~openmp",      when="@:0.8.0 +vtkh~openmp")
    depends_on("vtk-h+cuda+openmp", when="@:0.8.0 +vtkh+cuda+openmp")
    depends_on("vtk-h+cuda~openmp", when="@:0.8.0 +vtkh+cuda~openmp")

    depends_on("vtk-h~shared",             when="@:0.8.0 ~shared+vtkh")
    depends_on("vtk-h~shared~openmp",      when="@:0.8.0 ~shared+vtkh~openmp")
    depends_on("vtk-h~shared+cuda",        when="@:0.8.0 ~shared+vtkh+cuda")
    depends_on("vtk-h~shared+cuda~openmp", when="@:0.8.0 ~shared+vtkh+cuda~openmp")

    #############################
    # mfem
    #############################
    depends_on("mfem~threadsafe~openmp+shared+conduit", when="+shared+mfem")
    depends_on("mfem~threadsafe~openmp~shared+conduit", when="~shared+mfem")

    #######################
    # Devil Ray
    #######################
    # use external dray for 0.8.0 and older, its built in for newer ver of Ascent
    # devil ray variants with mpi
    # we have to specify both because mfem makes us
    depends_on("dray+mpi+shared+cuda",        when="@:0.8.0 +dray+mpi+cuda+shared")
    depends_on("dray+mpi+shared+openmp",      when="@:0.8.0 +dray+mpi+openmp+shared")
    depends_on("dray+mpi+shared~openmp~cuda", when="@:0.8.0 +dray+mpi~openmp~cuda+shared")

    depends_on("dray+mpi~shared+cuda",        when="@:0.8.0 +dray+mpi+cuda~shared")
    depends_on("dray+mpi~shared+openmp",      when="@:0.8.0 +dray+mpi+openmp~shared")
    depends_on("dray+mpi~shared~openmp~cuda", when="@:0.8.0 +dray+mpi~openmp~cuda~shared")

    # devil ray variants without mpi
    depends_on("dray~mpi+shared+cuda",        when="@:0.8.0 +dray~mpi+cuda+shared")
    depends_on("dray~mpi+shared+openmp",      when="@:0.8.0 +dray~mpi+openmp+shared")
    depends_on("dray~mpi+shared~openmp~cuda", when="@:0.8.0 +dray~mpi~openmp~cuda+shared")

    depends_on("dray~mpi~shared+cuda",        when="@:0.8.0 +dray~mpi+cuda~shared")
    depends_on("dray~mpi~shared+openmp",      when="@:0.8.0 +dray~mpi+openmp~shared")
    depends_on("dray~mpi~shared~openmp~cuda", when="@:0.8.0 +dray~mpi~openmp~cuda~shared")

    #######################
    # occa
    #######################
    # occa defaults to +cuda so we have to explicit tell it ~cuda
    depends_on("occa~cuda",        when="+occa~cuda")
    depends_on("occa~cuda~openmp", when="+occa~cuda~openmp")
    depends_on("occa+cuda+openmp", when="+occa+cuda+openmp")
    depends_on("occa+cuda~openmp", when="+occa+cuda~openmp")

    #############################
    # adios2
    #############################
    depends_on("adios2", when="+adios2")

    #############################
    # fides
    #############################
    depends_on("fides", when="+fides")

    #############################
    # genten
    #############################
    depends_on("genten", when="+genten")
    depends_on("genten+cuda~openmp", when="+genten+cuda~openmp")
    depends_on("genten+openmp~cuda", when="+genten+openmp~cuda")

    #######################
    # BabelFlow
    #######################
    depends_on('babelflow', when='+babelflow+mpi')
    depends_on('parallelmergetree', when='+babelflow+mpi')
    depends_on('talass', when='+babelflow+mpi')
    depends_on('streamstat', when='+babelflow+mpi')

    #######################
    # Documentation related
    #######################
    depends_on("py-sphinx", when="+python+doc", type='build')
    depends_on("py-sphinx-rtd-theme", when="+python+doc", type='build')

    ###########
    # Conflicts
    ###########
    conflicts("+shared", when="@:0.8.1 +cuda",
              msg="Ascent 0.8.0 and older need to be built with ~shared for CUDA builds.")

    ###################################
    # build phases used by this package
    ###################################
    phases = ['hostconfig', 'cmake', 'build', 'install']

    def setup_build_environment(self, env):
        env.set('CTEST_OUTPUT_ON_FAILURE', '1')

    ####################################################################
    # Note: cmake, build, and install stages are handled by CMakePackage
    ####################################################################

    # provide cmake args (pass host config as cmake cache file)
    def cmake_args(self):
        host_config = self._get_host_config_path(self.spec)
        options = []
        options.extend(['-C', host_config, "../spack-src/src/"])
        return options

    @run_after('install')
    @on_package_attributes(run_tests=True)
    def check_install(self):
        """
        Checks the spack install of ascent using ascents's
        using-with-cmake example
        """
        print("Checking Ascent installation...")
        spec = self.spec
        install_prefix = spec.prefix
        example_src_dir = join_path(install_prefix,
                                    "examples",
                                    "ascent",
                                    "using-with-cmake")
        print("Checking using-with-cmake example...")
        with working_dir("check-ascent-using-with-cmake-example",
                         create=True):
            cmake_args = ["-DASCENT_DIR={0}".format(install_prefix),
                          "-DCONDUIT_DIR={0}".format(spec['conduit'].prefix),
                          "-DVTKM_DIR={0}".format(spec['vtk-m'].prefix),
                          "-DVTKH_DIR={0}".format(spec['vtk-h'].prefix),
                          example_src_dir]
            cmake(*cmake_args)
            make()
            example = Executable('./ascent_render_example')
            example()
        print("Checking using-with-make example...")
        example_src_dir = join_path(install_prefix,
                                    "examples",
                                    "ascent",
                                    "using-with-make")
        example_files = glob.glob(join_path(example_src_dir, "*"))
        with working_dir("check-ascent-using-with-make-example",
                         create=True):
            for example_file in example_files:
                shutil.copy(example_file, ".")
            make("ASCENT_DIR={0}".format(install_prefix))
            example = Executable('./ascent_render_example')
            example()

    def _get_host_config_path(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        host_config_path = "{0}-{1}-{2}-ascent-{3}.cmake".format(socket.gethostname(),
                                                                  sys_type,
                                                                  spec.compiler,
                                                                  spec.dag_hash())
        dest_dir = spec.prefix
        host_config_path = os.path.abspath(join_path(dest_dir,
                                                     host_config_path))
        return host_config_path


    def hostconfig(self, spec, prefix):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build ascent.

        For more details about 'host-config' files see:
            http://ascent.readthedocs.io/en/latest/BuildingAscent.html

        """
        if not os.path.isdir(spec.prefix):
            os.mkdir(spec.prefix)

        #######################
        # Compiler Info
        #######################
        c_compiler = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        f_compiler = env["SPACK_FC"]

        #######################################################################
        # Directly fetch the names of the actual compilers to create a
        # 'host config' file that works outside of the spack install env.
        #######################################################################
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]

        ##############################################
        # Find and record what CMake is used
        ##############################################
        if "+cmake" in spec:
            cmake_exe = spec['cmake'].command.path
        else:
            cmake_exe = which("cmake")
            if cmake_exe is None:
                msg = 'failed to find CMake (and cmake variant is off)'
                raise RuntimeError(msg)
            cmake_exe = cmake_exe.path

        # get hostconfig name
        host_cfg_fname = self._get_host_config_path(spec)

        cfg = open(host_cfg_fname, "w")
        cfg.write("##################################\n")
        cfg.write("# spack generated host-config\n")
        cfg.write("##################################\n")
        cfg.write("# {0}-{1}\n".format(sys_type, spec.compiler))
        cfg.write("##################################\n\n")

        # Include path to cmake for reference
        cfg.write("# cmake from spack \n")
        cfg.write("# cmake executable path: %s\n\n" % cmake_exe)

        #######################
        # Compiler Settings
        #######################
        cfg.write("#######\n")
        cfg.write("# using %s compiler spec\n" % spec.compiler)
        cfg.write("#######\n\n")
        cfg.write("# c compiler used by spack\n")
        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER", c_compiler))
        cfg.write("# cpp compiler used by spack\n")
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER", cpp_compiler))

        cfg.write("# fortran compiler used by spack\n")
        if "+fortran" in spec:
            cfg.write(cmake_cache_entry("ENABLE_FORTRAN", "ON"))
            cfg.write(cmake_cache_entry("CMAKE_Fortran_COMPILER",
                                        f_compiler))
        else:
            cfg.write(cmake_cache_entry("ENABLE_FORTRAN", "OFF"))

        # shared vs static libs
        if "+shared" in spec:
            cfg.write(cmake_cache_entry("BUILD_SHARED_LIBS", "ON"))
        else:
            cfg.write(cmake_cache_entry("BUILD_SHARED_LIBS", "OFF"))

        # use global spack compiler flags
        cppflags = ' '.join(spec.compiler_flags['cppflags'])
        if cppflags:
            # avoid always ending up with ' ' with no flags defined
            cppflags += ' '
        cflags = cppflags + ' '.join(spec.compiler_flags['cflags'])
        if cflags:
            cfg.write(cmake_cache_entry("CMAKE_C_FLAGS", cflags))
        cxxflags = cppflags + ' '.join(spec.compiler_flags['cxxflags'])
        if cxxflags:
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS", cxxflags))
        fflags = ' '.join(spec.compiler_flags['fflags'])
        if self.spec.satisfies('%cce'):
            fflags += " -ef"
        if fflags:
            cfg.write(cmake_cache_entry("CMAKE_Fortran_FLAGS", fflags))

        #######################
        # Unit Tests
        #######################
        if "+test" in spec:
            cfg.write(cmake_cache_entry("ENABLE_TESTS", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_TESTS", "OFF"))

        #######################################################################
        # Core Dependencies
        #######################################################################

        #######################
        # Conduit
        #######################
        cfg.write("# conduit from spack \n")
        cfg.write(cmake_cache_entry("CONDUIT_DIR", spec['conduit'].prefix))

        #######################################################################
        # Optional Dependencies
        #######################################################################

        #######################
        # Python
        #######################
        cfg.write("# Python Support\n")

        if "+python" in spec and "+shared" in spec:
            cfg.write("# Enable python module builds\n")
            cfg.write(cmake_cache_entry("ENABLE_PYTHON", "ON"))
            cfg.write("# python from spack \n")
            cfg.write(cmake_cache_entry("PYTHON_EXECUTABLE",
                      spec['python'].command.path))
            try:
                cfg.write("# python module install dir\n")
                cfg.write(cmake_cache_entry("PYTHON_MODULE_INSTALL_PREFIX",
                          site_packages_dir))
            except NameError:
                # spack's  won't exist in a subclass
                pass
        else:
            cfg.write(cmake_cache_entry("ENABLE_PYTHON", "OFF"))

        if "+doc" in spec and "+python" in spec:
            cfg.write(cmake_cache_entry("ENABLE_DOCS", "ON"))

            cfg.write("# sphinx from spack \n")
            sphinx_build_exe = join_path(spec['py-sphinx'].prefix.bin,
                                         "sphinx-build")
            cfg.write(cmake_cache_entry("SPHINX_EXECUTABLE", sphinx_build_exe))
        else:
            cfg.write(cmake_cache_entry("ENABLE_DOCS", "OFF"))

        #######################
        # Serial
        #######################
        if "+serial" in spec:
            cfg.write(cmake_cache_entry("ENABLE_SERIAL", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_SERIAL", "OFF"))

        #######################
        # MPI
        #######################
        cfg.write("# MPI Support\n")

        if "+mpi" in spec:
            mpicc_path = spec['mpi'].mpicc
            mpicxx_path = spec['mpi'].mpicxx
            mpifc_path = spec['mpi'].mpifc
            # if we are using compiler wrappers on cray systems
            # use those for mpi wrappers, b/c  spec['mpi'].mpicxx
            # etc make return the spack compiler wrappers
            # which can trip up mpi detection in CMake 3.14
            if cpp_compiler == "CC":
                mpicc_path = "cc"
                mpicxx_path = "CC"
                mpifc_path = "ftn"
            cfg.write(cmake_cache_entry("ENABLE_MPI", "ON"))
            cfg.write(cmake_cache_entry("MPI_C_COMPILER", mpicc_path))
            cfg.write(cmake_cache_entry("MPI_CXX_COMPILER", mpicxx_path))
            cfg.write(cmake_cache_entry("MPI_Fortran_COMPILER", mpifc_path))
            mpiexe_bin = join_path(spec['mpi'].prefix.bin, 'mpiexec')
            if os.path.isfile(mpiexe_bin):
                # starting with cmake 3.10, FindMPI expects MPIEXEC_EXECUTABLE
                # vs the older versions which expect MPIEXEC
                if self.spec["cmake"].satisfies('@3.10:'):
                    cfg.write(cmake_cache_entry("MPIEXEC_EXECUTABLE",
                                                mpiexe_bin))
                else:
                    cfg.write(cmake_cache_entry("MPIEXEC",
                                                mpiexe_bin))
            # find mpi support
            if "+blt_find_mpi" in spec:
                cfg.write(cmake_cache_entry("ENABLE_FIND_MPI", "ON"))
            else:
                cfg.write(cmake_cache_entry("ENABLE_FIND_MPI", "OFF"))
            ###################################
            # BABELFLOW (also depends on mpi)
            ###################################
            if "+babelflow" in spec:
                cfg.write(cmake_cache_entry("BABELFLOW_DIR",
                                            spec['babelflow'].prefix))
                cfg.write(cmake_cache_entry("PMT_DIR",
                                            spec['parallelmergetree'].prefix))
                cfg.write(cmake_cache_entry("StreamStat_DIR",
                                            spec['streamstat'].prefix))
                cfg.write(cmake_cache_entry("TopoFileParser_DIR",
                                            spec['talass'].prefix))
        else:
            cfg.write(cmake_cache_entry("ENABLE_MPI", "OFF"))

        #######################
        # CUDA
        #######################
        cfg.write("# CUDA Support\n")

        if "+cuda" in spec:
            cfg.write(cmake_cache_entry("ENABLE_CUDA", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_CUDA", "OFF"))

        if "+openmp" in spec:
            cfg.write(cmake_cache_entry("ENABLE_OPENMP", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_OPENMP", "OFF"))

        ########################
        # ROCm
        #########################
        cfg.write("# ROCm Support\n")

        if "+rocm" in spec:
            cfg.write(cmake_cache_entry("ENABLE_HIP", "ON"))
            cfg.write(cmake_cache_entry("CMAKE_HIP_COMPILER", c_compiler))
            # NOTE: We need the root install of rocm for ROCM_PATH
            # There is no spack package named `rocm`, but rocminfo seems
            # to also point to the root of the rocm install
            cfg.write(cmake_cache_entry("ROCM_PATH", spec['rocminfo'].prefix))
            rocm_archs = ",".join(self.spec.variants['amdgpu_target'].value)
            cfg.write(cmake_cache_entry("CMAKE_HIP_ARCHITECTURES", rocm_archs))
        else:
            cfg.write(cmake_cache_entry("ENABLE_HIP", "OFF"))

        #######################
        # VTK-h (and deps)
        #######################
        cfg.write("# vtk-h support \n")

        if "+vtkh" in spec:
            cfg.write("# vtk-h\n")
            if self.spec.satisfies('@0.8.1:'):
                cfg.write(cmake_cache_entry("ENABLE_VTKH", "ON"))
            else:
                cfg.write(cmake_cache_entry("VTKH_DIR", spec['vtk-h'].prefix))

            cfg.write("# vtk-m from spack\n")
            cfg.write(cmake_cache_entry("VTKM_DIR", spec['vtk-m'].prefix))

            if "+cuda" in spec:
                cfg.write(cmake_cache_entry("VTKm_ENABLE_CUDA", "ON"))
                cfg.write(cmake_cache_entry("CMAKE_CUDA_HOST_COMPILER",
                          env["SPACK_CXX"]))
            else:
                cfg.write(cmake_cache_entry("VTKm_ENABLE_CUDA", "OFF"))

            if "+rocm" in spec:
                cfg.write(cmake_cache_entry("VTKm_ENABLE_KOKKOS ", "ON"))
                cfg.write(cmake_cache_entry("KOKKOS_DIR", spec['kokkos'].prefix))
            else:
                cfg.write("# vtk-m not using ROCm\n")


        else:
            if self.spec.satisfies('@0.8.1:'):
                cfg.write("# vtk-h\n")
                cfg.write(cmake_cache_entry("ENABLE_VTKH", "OFF"))
            else:
                cfg.write("# vtk-h not build by spack\n")

        #######################
        # MFEM
        #######################
        if "+mfem" in spec:
            cfg.write("# mfem from spack \n")
            cfg.write(cmake_cache_entry("MFEM_DIR", spec['mfem'].prefix))
            if "zlib" in spec:
                # MFEM depends on zlib
                cfg.write(cmake_cache_entry("ZLIB_DIR", spec["zlib"].prefix))
        else:
            cfg.write("# mfem not built by spack \n")

        #######################
        # Devil Ray
        #######################
        if "+dray" in spec:
            cfg.write("# devil ray\n")
            if self.spec.satisfies('@0.8.1:'):
                cfg.write(cmake_cache_entry("ENABLE_DRAY", "ON"))
                cfg.write(cmake_cache_entry("ENABLE_APCOMP", "ON"))
            else:
                cfg.write("# devil ray from spack \n")
                cfg.write(cmake_cache_entry("DRAY_DIR", spec['dray'].prefix))
        else:
            if self.spec.satisfies('@0.8.1:'):
                cfg.write("# devil ray\n")
                cfg.write(cmake_cache_entry("ENABLE_DRAY", "OFF"))
                cfg.write(cmake_cache_entry("ENABLE_APCOMP", "OFF"))
            else:
                cfg.write("# devil ray not build by spack\n")

        #######################
        # OCCA
        #######################
        if "+occa" in spec:
            cfg.write("# occa from spack \n")
            cfg.write(cmake_cache_entry("OCCA_DIR", spec['occa'].prefix))
        else:
            cfg.write("# occa not built by spack \n")

        #######################
        # RAJA
        #######################
        if "+raja" in spec:
            cfg.write("# RAJA from spack \n")
            cfg.write(cmake_cache_entry("RAJA_DIR", spec['raja'].prefix))
        else:
            cfg.write("# RAJA not built by spack \n")

        #######################
        # Umpire
        #######################
        if "+umpire" in spec:
            cfg.write("# umpire from spack \n")
            cfg.write(cmake_cache_entry("UMPIRE_DIR", spec['umpire'].prefix))
        else:
            cfg.write("# umpire not built by spack \n")

        #######################
        # Camp
        #######################
        if "+umpire" in spec or "+raja" in spec:
            cfg.write("# camp from spack \n")
            cfg.write(cmake_cache_entry("CAMP_DIR", spec['camp'].prefix))
        else:
            cfg.write("# camp not built by spack \n")

        #######################
        # Adios2
        #######################
        cfg.write("# adios2 support\n")

        if "+adios2" in spec:
            cfg.write(cmake_cache_entry("ADIOS2_DIR", spec['adios2'].prefix))
        else:
            cfg.write("# adios2 not built by spack \n")

        #######################
        # Fides
        #######################
        cfg.write("# Fides support\n")

        if "+fides" in spec:
            cfg.write(cmake_cache_entry("FIDES_DIR", spec['fides'].prefix))
        else:
            cfg.write("# fides not built by spack \n")

        #######################
        # GenTen
        #######################
        cfg.write("# GenTen support\n")
        if "+genten" in spec:
            cfg.write(cmake_cache_entry("GENTEN_DIR", spec['genten'].prefix))
        else:
            cfg.write("# genten not built by spack \n")

        #######################
        # Finish host-config
        #######################

        cfg.write("##################################\n")
        cfg.write("# end spack generated host-config\n")
        cfg.write("##################################\n")
        cfg.close()

        host_cfg_fname = os.path.abspath(host_cfg_fname)
        tty.info("spack generated ascent host-config file: " + host_cfg_fname)

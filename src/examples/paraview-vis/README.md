Insitu ParaView visualization using the Ascent Extract interface
================================================================

# 1. Installation instructions for Ascent git
* Install Ascent with mfem turned on so that proxies/laghos is built
   - `git clone --recursive https://github.com/alpine-dav/ascent.git`
   - `cd ascent`
   - `python scripts/uberenv/uberenv.py --install --prefix="build" --spec='+mfem'`
       `build/ascent-install` will contain the ascent installation and built examples
       and `build/spack` will contain the spack packages.
     Note that FORTRAN is required for the compiler chosen. Use `spack compilers`
     to see the compilers found by spack.
* Setup spack. Note spack modules are not setup
   - `. ascent/build/spack/share/spack/setup-env.sh`
* Install ParaView (use older matplotlib because newest requires python3.)
   - `spack install paraview +python ^py-matplotlib@2.2.3 ~tk`
      We use ParaView master because we need the fix in 0b349f3e18.
* Continue with `4. Common installation instructions`

# 2. Installation instructions for spack devel
* Install spack, modules and shell support.
  - Execute the following:  
  ```
  git clone https://github.com/spack/spack.git  
  cd spack  
  source share/spack/setup-env.sh  
  ```
  - Optional if `module` command does not exist:
    - instal environment-modules (or run `spack bootstrap`)
    - run `add.modules` to add the `module` command to your `.bashrc` file

# 3. Install ParaView and Ascent
* Optional: For MomentInvariants visualization patch paraview package
  - Download the patch [paraview-package-momentinvariants.patch](paraview-package-momentinvariants.patch)
  - Patch paraview: `patch -p1 < paraview-package-momentinvariants.patch`
* Install ParaView
  - `spack install paraview@develop+python3+mpi+osmesa~opengl2`
  - on mac use: `spack install paraview@develop+python3+mpi^python+shared`
    (ParaView OSMesa does not compile, conduit needs `^python+shared`)
  - for CUDA use: `spack install paraview@develop+python3+mpi+osmesa~opengl2+cuda`
* Install Ascent
  - `spack install ascent~vtkh^python@3.7.4`
     Make sure you match the python version used by ParaView
  - If you need ascent built with vtkh you can use
    `spack install ascent^python@3.7.4`. Note that the patch applied earlier
    select specific version of `vtkh` and `vtkm` that work with latest ascent.
    Those versions were backported from `ascent/scripts/uberenv/packages/`
    `vtkh/package.py` and `vtkm/package.py`.
* Load required modules
  - `spack load conduit;spack load py-numpy;spack load py-mpi4py;spack load paraview`
* Continue with `4. Common installation instructions`

# 4. Common installation instructions
* Test with available simulations - VTK files and images will be generated in the current directory.
   - Test `proxies/cloverleaf3d`
     - Go to a directory where to run cloverleaf, for summit use `cd $MEMBERWORK/csc340`.
     - `ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview_ascent_source.py`
     - Execute: `ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview-vis-cloverleaf3d-momentinvariants.py paraview-vis.py`
     for MomentInvariants visualization or `ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview-vis-cloverleaf3d.py paraview-vis.py`
     for surface visualization.
     - Execute:
     ```
     ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/ascent_actions.json  
     ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/expandingVortex.vti  
     ln -s $(spack location --install-dir ascent)/examples/ascent/proxies/cloverleaf3d/clover.in  
     ```
     - Set `paraview_path` in paraview-vis.py
         with the result of `echo $(spack location --install-dir paraview)/lib/python*/site-packages`
     - Run the simulation 
     `$(spack location --install-dir mpi)/bin/mpiexec -n 2 $(spack location --install-dir ascent)/examples/ascent/proxies/cloverleaf3d/cloverleaf3d_par > output.txt 2>&1`
     - examine the generated VTK files the images
   - Similarily test: `proxies/kripke`, `proxies/laghos`, `proxies/lulesh` (you need to create the links)
     - `$(spack location --install-dir mpi)/bin/mpiexec -np 8 ./kripke_par --procs 2,2,2 --zones 32,32,32 --niter 5 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4 > output.txt 2>&1`
     - `$(spack location --install-dir mpi)/bin/mpiexec -n 8 ./laghos_mpi -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.6 -visit -pa > output.txt 2>&1`
     - `$(spack location --install-dir mpi)/bin/mpiexec -np 8 ./lulesh_par -i 10 -s 32 > output.txt 2>&1`
   - Test noise from `synthetic/noise`
     - `$(spack location --install-dir mpi)/bin/mpiexec -np 8 ./noise_par  --dims=32,32,32 --time_steps=5 --time_delta=1 > output.txt 2>&1`

# 5. Compile and run on summit.olcf.ornl.gov
* Execute only `2. Installation instructions for spack devel` and
  do not continue with `3. Install ParaView and Ascent`
* Configure spack
  - add a file `~/.spack/packages.yaml` with the following content:
  ```
  packages:
   spectrum-mpi:
     modules:
       spectrum-mpi@10.3.0.1-20190611: spectrum-mpi/10.3.0.1-20190611
     buildable: False
   cuda:
     modules:
       cuda@10.1.168: cuda/10.1.168
     buildable: False
  ```
  - Load the correct compiler:
    ```
    module load gcc/7.4.0
    spack compiler add
    spack compiler remove gcc@4.8.5
    ```
* For busy summit I got `internal compiler error` when compiling `llvm` and `paraview`
    To fix this you could run `spack install` with `-j4` or even `-j1`. Even better
    move the spack installation on `$MEMBERWORK/csc340` and recompile everything
    on a compute node.
  - First login to a compute node (for an hour):
    `bsub -W 1:00 -nnodes 1 -P CSC340 -Is /bin/bash`
  - Install all spack packages with `-j80` option (there are 84 threads)
* Continue with `4. Common Installation Instructions` but do not run the simulation.
* Execute cloverleaf 
  `bsub $(spack location --install-dir ascent)/examples/ascent/paraview-vis/summit-moment-invariants.lsf`
  To check the results use `bjobs -a`

# Todo:
* Add testing
* Build the ParaView pipeline only once.

# Note:
Global extents are computed for uniform and rectilinear topologies but
they are not yet computed for a structured topology (lulesh). This
means that for lulesh and datasets that have a structured topology we
cannot save a correct parallel file that represents the whole dataset.

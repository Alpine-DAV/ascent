Insitu ParaView visualization using the Ascent Extract interface
================================================================

# Installation instructions for Ascent git
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
* Continue with `Common installation instructions`

# Installation instructions for spack devel
* Install spack, modules and shell support.
  - Execute the following:  
  ```
  git clone https://github.com/danlipsa/spack.git  
  cd spack  
  git checkout moment-invariants  
  source share/spack/setup-env.sh  
  spack bootstrap
  ```
* Optional: For MomentInvariants visualization patch paraview package
  - Download the patch [paraview-package-momentinvariants.patch](paraview-package-momentinvariants.patch)
  - Patch paraview: `patch -p1 < paraview-package-momentinvariants.patch`
* Install ParaView
  - `spack install paraview@develop+python3+mpi`
* Install Ascent
  - `spack install ascent~vtkh^python@3.7.4`
     Make sure you match the python version used by ParaView
* Load required modules
  - `spack load conduit;spack load py-numpy;spack load py-mpi4py;spack load paraview`
* Continue with `Common installation instructions`

# Common installation instructions
* Test with available simulations - VTK files and images will be generated in the current directory.
   - Test `proxies/cloverleaf3d`
     - `cd $(spack location --install-dir ascent)/examples/ascent/proxies/cloverleaf3d`
     - `ln -s ../../paraview-vis/paraview_ascent_source.py`
     - Execute: `ln -s ../../paraview-vis/paraview-vis-cloverleaf3d-momentinvariant.py paraview-vis.py`
     for MomentInvariants visualization or `ln -s ../../paraview-vis/paraview-vis-cloverleaf3d.py paraview-vis.py`
     for surface visualization.
     - Execute:
     ```
     mv ascent_actions.json ascent_actions_volume.json  
     ln -s ../../paraview-vis/ascent_actions.json  
     ln -s ../../paraview-vis/expandingVortex.vti  
     ```
     - replace paraview_path from paraview-vis-xxx.py
         with the result of `echo $(spack location --install-dir paraview)/lib/python*/site-packages` and
         scriptName with the correct path to 'paraview_ascent_source.py'
     - `$(spack location --install-dir mpi)/bin/mpiexec -n 2 cloverleaf3d_par > output.txt 2>&1`
     - examine the generated VTK files the images
   - Similarily test: `proxies/kripke`, `proxies/laghos`, `proxies/lulesh` (you need to create the links)
     - `$(spack location --install-dir mpi)/bin/mpiexec -np 8 kripke_par --procs 2,2,2 --zones 32,32,32 --niter 5 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4 > output.txt 2>&1`
     - `$(spack location --install-dir mpi)/bin/mpiexec -n 8 ./laghos_mpi -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.6 -visit -pa > output.txt 2>&1`
     - `$(spack location --install-dir mpi)/bin/mpiexec -np 8 lulesh_par -i 10 -s 32 > output.txt 2>&1`
   - Test noise from `synthetic/noise`
     - `$(spack location --install-dir mpi)/bin/mpiexec -np 8 noise_par  --dims=32,32,32 --time_steps=5 --time_delta=1 > output.txt 2>&1`

# Todo:
* Add testing
* Build the ParaView pipeline only once.

# Note:
Global extents are computed for uniform and rectilinear topologies but
they are not yet computed for a structured topology (lulesh). This
means that for lulesh and datasets that have a structured topology we
cannot save a correct parallel file that represents the whole dataset.

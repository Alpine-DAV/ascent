set(install_files
  ascent_actions.json
  expandingVortex.vti
  paraview_ascent_source.py
  paraview-package-momentinvariants.patch)

set(configure_files
  paraview-vis-cloverleaf3d-momentinvariants.py.in
  paraview-vis-cloverleaf3d.py.in
  paraview-vis-kripke.py.in
  paraview-vis-laghos.py.in
  paraview-vis-lulesh.py.in
  paraview-vis-noise.py.in
  paraview-vis-warpx.py.in)

set(configure_dest_files)
set(PARAVIEW_ASCENT_SOURCE
  ${CMAKE_INSTALL_PREFIX}/examples/ascent/paraview-vis/paraview_ascent_source.py)
foreach (cfin IN LISTS configure_files)
  string(REPLACE ".in" "" cfout ${cfin})
  list(APPEND configure_dest_files ${CMAKE_CURRENT_BINARY_DIR}/${cfout})
  configure_file(${cfin} ${cfout})
endforeach()

install(FILES ${install_files} ${configure_dest_files}
  DESTINATION examples/ascent/paraview-vis)

install(DIRECTORY tests
  DESTINATION examples/ascent/paraview-vis)

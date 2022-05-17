set(solver_third_party_libs mfem conduit)

################################################
# openmp support
################################################
if(ENABLE_OPENMP)
  list(APPEND solver_thirdparty_libs openmp)
endif()


################################################
# diffusion solver
################################################
blt_add_executable(
  NAME diffusion_solver
  SOURCES diffusion.cpp
  DEPENDS_ON ${solver_third_party_libs}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)


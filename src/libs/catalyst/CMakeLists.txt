# When implementing the Catalyst API, as against using
# it to invoke Catalyst, one must use the component
# ``SDK`` in ``find_package`` call. This ensures that all necessary
# libraries etc. are available.
if(ENABLE_MPI)
  add_compile_definitions(_USE_MPI)
endif()

set(BUILD_SHARED_LIBS ON)

catalyst_implementation(
  NAME    ascent
  TARGET  catalyst-ascent
  SOURCES AscentCatalyst.cxx
  LIBRARY_DESTINATION lib
  CATALYST_TARGET catalyst::catalyst)

if(ENABLE_MPI)
  target_link_libraries(catalyst-ascent PRIVATE ascent_mpi)
else()
  target_link_libraries(catalyst-ascent PRIVATE ascent)
endif()

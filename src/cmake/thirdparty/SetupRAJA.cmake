message(STATUS "Looking for RAJA in: ${RAJA_DIR}")
if (NOT RAJA_DIR)
  message(FATAL_ERROR "Must specify 'RAJA_DIR'")
endif()

set(RAJA_BASE_DIR ${RAJA_DIR})
find_dependency(RAJA REQUIRED
               NO_DEFAULT_PATH
               PATHS ${RAJA_DIR}/share/raja/cmake)

message(STATUS "Found RAJA")

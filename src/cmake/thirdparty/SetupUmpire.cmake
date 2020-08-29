message(STATUS "Looking for Umpire in: ${UMPIRE_DIR}")
if (NOT UMPIRE_DIR)
  message(FATAL_ERROR "Must specify 'UMPIRE_DIR'")
endif()

set(umpire_DIR ${UMPIRE_DIR}/share/umpire/cmake)
find_package(umpire REQUIRED)
message(STATUS "Found Umpire")

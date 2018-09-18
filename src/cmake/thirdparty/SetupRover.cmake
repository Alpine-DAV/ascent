if(NOT ROVER_DIR)
  message(FATAL_ERROR "Could not find Rover. Rover needs explicit ROVER_DIR to be set")
endif()

if(NOT EXISTS ${ROVER_DIR}/lib/RoverConfig.cmake)
  message(FATAL_ERROR "Could not find Rover CMake file: ${ROVER_DIR}/lib/RoverConifg.cmake")
endif()

include(${ROVER_DIR}/lib/RoverConfig.cmake)

set(ROVER_FOUND TRUE)
message(STATUS "Found Rover. Include dirs ${ROVER_INCLUDE_DIRS}")

blt_register_library(NAME rover 
                     INCLUDES ${ROVER_INCLUDE_DIRS}
                     LIBRARIES rover)

if (MPI_FOUND)
    blt_register_library(NAME rover_par
                         INCLUDES ${ROVER_INCLUDE_DIRS}
                         LIBRARIES rover_par)

endif()

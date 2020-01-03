
find_package(MPI REQUIRED)
find_package(BabelFlow REQUIRED)
find_package(PMT REQUIRED)

set(BabelFlow_INCLUDE_DIRS ${BabelFlow_DIR}/../../../include)
set(PMT_INCLUDE_DIRS ${PMT_DIR}/../../../include/PMT)


message(STATUS "FOUND BabelFlow at ${BabelFlow_DIR}")
message(STATUS "BabelFlow_INCLUDE_DIRS = ${BabelFlow_INCLUDE_DIRS}")

blt_register_library( NAME babelflow
                      INCLUDES ${BabelFlow_INCLUDE_DIRS}
                      LIBRARIES  babelflow babelflow_mpi)

message(STATUS "FOUND PMT at ${PMT_DIR}")
message(STATUS "PMT_INCLUDE_DIRS = ${PMT_INCLUDE_DIRS}")

blt_register_library( NAME pmt
                      INCLUDES ${PMT_INCLUDE_DIRS}
                      LIBRARIES  pmt)




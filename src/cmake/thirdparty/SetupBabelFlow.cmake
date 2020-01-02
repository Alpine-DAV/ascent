#include(${BabelFlow_DIR}/FindBabelFlow.cmake)

find_package(MPI REQUIRED)
find_package(BabelFlow REQUIRED)

set(BabelFlow_INCLUDE_DIRS ${BabelFlow_DIR}/../../../include)

#message("BABELFLOW_LIBRARIES ${BABELFLOW_LIBRARIES}")

#include_directories(${BABELFLOW_INCLUDE_DIR} ${MPI_C_INCLUDE_PATH})
#link_directories(${BABELFLOW_INCLUDE_DIR}/../lib)
#link_libraries(${BABELFLOW_LIBRARIES} ${MPI_CXX_LIBRARIES})


find_package(PMT REQUIRED)
set(PMT_INCLUDE_DIRS ${PMT_DIR}/../../../include/PMT)

#message("PMT_LIBRARIES ${PMT_LIBRARIES}")
#include_directories(${PMT_INCLUDE_DIR})
#link_directories(${PMT_INCLUDE_DIR}/../../../lib)
#link_libraries(PMT::pmt)

#set(PMT_DIR "" CACHE ${BabelFlow_DIR} "Parallel merge tree dir")
#if (PMT_DIR)
#    message("PMT_DIR=${PMT_DIR}")
#    set(PMT_INCLUDE_DIR ${PMT_DIR}/include)
#    set(PMT_LIBRARY_DIR ${PMT_DIR}/lib)
#    set(PMT_LIBRARIES libpmt.a)
#    include_directories(${PMT_INCLUDE_DIR})
#    link_directories(${PMT_LIBRARY_DIR})
#    link_libraries(${PMT_LIBRARIES})
#else()
#    message( WARNING "the variable PMT_DIR is not defined")
#endif ()

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




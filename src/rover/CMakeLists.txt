###############################################################################
##  Copyright (c) 2018, Lawrence Livermore National Security, LLC.
##
##  Produced at the Lawrence Livermore National Laboratory
##
##  LLNL-CODE-749865
##
##  All rights reserved.
##
##  This file is part of Rover.
##
##  Please also read rover/LICENSE
##
##  Redistribution and use in source and binary forms, with or without
##  modification, are permitted provided that the following conditions are met:
##
##  * Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the disclaimer below.
##
##  * Redistributions in binary form must reproduce the above copyright notice,
##    this list of conditions and the disclaimer (as noted below) in the
##    documentation and/or other materials provided with the distribution.
##
##  * Neither the name of the LLNS/LLNL nor the names of its contributors may
##    be used to endorse or promote products derived from this software without
##    specific prior written permission.
##
##  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
##  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
##  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
##  ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
##  LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
##  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
##  DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
##  OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
##  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
##  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
##  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
##  POSSIBILITY OF SUCH DAMAGE.
##
###############################################################################

set(rover_thirdparty_deps vtkh_lodepng vtkm vtkh)

set(rover_headers
    domain.hpp
    image.hpp
    partial_image.hpp
    rover_exports.h
    rover_exceptions.hpp
    rover_types.hpp
    rover.hpp
    scheduler.hpp
    scheduler_base.hpp
    static_scheduler.hpp
    # engines
    engine.hpp
    energy_engine.hpp
    volume_engine.hpp
    # ray generators headers
    ray_generators/ray_generator.hpp
    ray_generators/camera_generator.hpp
    ray_generators/visit_generator.hpp
    rover_exports.h
    vtkm_typedefs.hpp
    # utils headers
    utils/png_encoder.hpp
    utils/rover_logging.hpp
    utils/vtk_dataset_reader.hpp
   )

set(rover_sources
    domain.cpp
    image.cpp
    rover.cpp
    scheduler.cpp
    scheduler_base.cpp
    # engines
    energy_engine.cpp
    volume_engine.cpp
    # ray generators
    ray_generators/ray_generator.cpp
    ray_generators/camera_generator.cpp
    ray_generators/visit_generator.cpp
    # utils sources
    utils/png_encoder.cpp
    utils/rover_logging.cpp
    utils/vtk_dataset_reader.cpp
   )


  blt_add_library(
                  NAME rover
                  SOURCES ${rover_sources}
                  HEADERS ${rover_headers}
                  DEPENDS_ON ${rover_thirdparty_deps}
                 )

vtkm_add_target_information(rover DEVICE_SOURCES ${rover_sources})
##########################################
# Ensure we use c++11
##########################################

set_target_properties(rover PROPERTIES
                      CXX_STANDARD 11
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENTIONS NO)

set_target_properties(rover PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(rover PRIVATE ROVER_EXPORTS_FLAG)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty_builtin/diy2/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# PUBLIC keyword pushes these libs into RoverTargets.cmake
# and they are automatically pulled into downstream projects
target_link_libraries(rover PUBLIC ${VTKm_LIBRARIES})
target_include_directories(rover PUBLIC ${VTKm_INCLUDE_DIRS})

if(ENABLE_LOGGING)
  blt_add_target_compile_flags(TO rover FLAGS " -D ROVER_ENABLE_LOGGING=1")
endif()

if(OPENMP_FOUND)
  blt_add_target_compile_flags(TO rover FLAGS " -D ROVER_ENABLE_OPENMP=1")
endif()

# install target for mpi ver
install(TARGETS rover
        EXPORT  ascent
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)
###################################################
#   Build parallel version
##################################################

if(MPI_FOUND)

  set(rover_mpi_thirdparty_deps mpi vtkh_lodepng vtkm vtkh_mpi)

  blt_add_library(
                  NAME rover_mpi
                  SOURCES ${rover_sources}
                  HEADERS ${rover_headers}
                  DEPENDS_ON ${rover_mpi_thirdparty_deps}
                  )

  vtkm_add_target_information(rover_mpi DEVICE_SOURCES ${rover_sources})
  ##########################################
  # Ensure we use c++11
  ##########################################

  set_target_properties(rover_mpi PROPERTIES
                        CXX_STANDARD 11
                        CXX_STANDARD_REQUIRED YES
                        CXX_EXTENTIONS NO)

                      blt_add_target_compile_flags(TO rover_mpi FLAGS "-DROVER_PARALLEL")

  set_target_properties(rover_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_compile_definitions(rover_mpi PRIVATE ROVER_EXPORTS_FLAG)

  target_include_directories(rover_mpi PRIVATE ${MPI_INCLUDE_PATH})
  target_include_directories(rover_mpi PUBLIC ${VTKm_INCLUDE_DIRS})

  target_link_libraries(rover_mpi PUBLIC ${VTKm_LIBRARIES} ${MPI_CXX_LIBRARIES})

  install(TARGETS rover_mpi
          EXPORT  ascent
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION lib
  )
  if(ENABLE_LOGGING)
    blt_add_target_compile_flags(TO rover_mpi FLAGS " -DROVER_ENABLE_LOGGING=1")
  endif()
  if(OPENMP_FOUND)
    blt_add_target_compile_flags(TO rover FLAGS " -DROVER_ENABLE_OPENMP=1")
  endif()

endif()

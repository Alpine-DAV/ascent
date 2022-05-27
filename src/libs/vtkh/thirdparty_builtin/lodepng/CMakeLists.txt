#==============================================================================
# See License.txt
#==============================================================================
set(Headers
  lodepng.h
  )

set(Sources
  lodepng.cpp
  )




blt_add_library(
  NAME vtkh_lodepng
  SOURCES ${Sources}
  )

target_compile_definitions(vtkh_lodepng PRIVATE VTKH_COMPILING_FLAG)

  # add_library(vtkh_lodepng ${Sources})
# target_include_directories(vtkh_lodepng INTERFACE
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#   $<INSTALL_INTERFACE:${VTKh_INSTALL_INCLUDE_DIR}/vtkh/thirdparty_builtin/lodepng>
#   )


########
# TODO: lodepng can be an object lib? I don't think we need to install
#  b/c it won't be used in vtk-h interface by callers?
########

########
# TODO: We should have a macro for install targets + headers
########


# install target for vtkh_lodepng
install(TARGETS vtkh_lodepng
  EXPORT ${VTKh_EXPORT_NAME}
  ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
  LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR}
  )

# Install headers
install(FILES ${Headers}
  DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/thirdparty_builtin/lodepng
  )

# Install others files.
install(FILES License.txt README.md
  DESTINATION ${VTKh_INSTALL_SHARED_RESOURCES_DIR}/thirdparty_builtin/lodepng
  )

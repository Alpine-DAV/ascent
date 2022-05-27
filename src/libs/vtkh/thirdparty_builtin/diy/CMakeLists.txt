#==============================================================================
# See License.txt
#==============================================================================

########
# TODO: blt_register library support interface libs
########

add_library(vtkhdiy INTERFACE)
target_include_directories(vtkhdiy INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${VTKh_INSTALL_INCLUDE_DIR}/vtkh/thirdparty_builtin>  # <prefix>/vtkh/thirdparty_builtin/
  )


########
# This is our DIY. There are are many like it, but this one is ours
# This is completely internal to vtkh
#######

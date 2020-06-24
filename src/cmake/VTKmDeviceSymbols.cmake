function(source_objects binary_dir sources rover targets locations)
  #message(STATUS "Simple arguments: ${binary_dir}, followed by ${SOURCES}")
  #set(${REQUIRED_ARG} "From SIMPLE" PARENT_SCOPE)
  set(locations_on_disk "")

  foreach(source IN LISTS sources)
    #CMake name object files "name.<ext>.o" so that it can handle multiple
    #files with the same name but different ext in the same project
    if("${source}" MATCHES "cpp*")
      string(APPEND locations_on_disk "${binary_dir}/${source}.o ")
    endif()
    if("${source}" MATCHES "cxx*")
      string(APPEND locations_on_disk "${binary_dir}/${source}.o ")
    endif()
  endforeach()

  string(APPEND locations_on_disk "${rover} ")

  foreach(item IN LISTS targets)
    #message(STATUS "looking for location of target ${item}")
    set(props_to_search IMPORTED_LOCATION
                        IMPORTED_LOCATION_RELEASE
                        IMPORTED_LOCATION_MINSIZEREL
                        IMPORTED_LOCATION_DEBUG)
    foreach(prop IN LISTS props_to_search)
      #message(STATUS "looking for prop ${prop} of target ${item}")
      get_target_property(location ${item} ${prop})
      if(location)
        string(APPEND locations_on_disk "${location} ")
        #message(STATUS "**LOCATION ${location}")
        break()
      endif()
    endforeach()
  endforeach()

  set(${locations} ${locations_on_disk} PARENT_SCOPE)
endfunction()


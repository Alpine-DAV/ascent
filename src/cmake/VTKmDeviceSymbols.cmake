# function(source_objects binary_dir sources rover targets locations)
#   #message(STATUS "Simple arguments: ${binary_dir}, followed by ${SOURCES}")
#   #set(${REQUIRED_ARG} "From SIMPLE" PARENT_SCOPE)
#   set(locations_on_disk "")

#   foreach(source IN LISTS sources)
#     #CMake name object files "name.<ext>.o" so that it can handle multiple
#     #files with the same name but different ext in the same project
#     if("${source}" MATCHES "cpp*")
#       string(APPEND locations_on_disk "${binary_dir}/${source}.o ")
#     endif()
#     if("${source}" MATCHES "cxx*")
#       string(APPEND locations_on_disk "${binary_dir}/${source}.o ")
#     endif()
#   endforeach()

#   string(APPEND locations_on_disk "${rover} ")

#   foreach(item IN LISTS targets)
#     #message(STATUS "looking for location of target ${item}")
#     set(props_to_search IMPORTED_LOCATION
#                         IMPORTED_LOCATION_RELEASE
#                         IMPORTED_LOCATION_RELWITHDEBINFO
#                         IMPORTED_LOCATION_MINSIZEREL
#                         IMPORTED_LOCATION_DEBUG)
#     foreach(prop IN LISTS props_to_search)
#       #message(STATUS "looking for prop ${prop} of target ${item}")
#       get_target_property(location ${item} ${prop})
#       if(location)
#         string(APPEND locations_on_disk "${location} ")
#         #message(STATUS "**LOCATION ${location}")
#         break()
#       endif()
#     endforeach()
#   endforeach()

#   set(${locations} ${locations_on_disk} PARENT_SCOPE)
# endfunction()


function(source_objects)

    set(options)
    set(singleValueArgs BINARY_DIR RESULT)
    set(multiValueArgs  SOURCES TARGETS IMPORTED_TARGETS)

    # parse macro arguments
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    #message(STATUS "Simple arguments: ${binary_dir}, followed by ${SOURCES}")
    #set(${REQUIRED_ARG} "From SIMPLE" PARENT_SCOPE)
    set(locations_on_disk "")

    foreach(source IN LISTS arg_SOURCES)
        #message(STATUS "WORKING ON ${source}")  
        #CMake name object files "name.<ext>.o" so that it can handle multiple
        #files with the same name but different ext in the same project
        if("${source}" MATCHES "cpp*")
            string(APPEND locations_on_disk "${arg_BINARY_DIR}/${source}.o ")
        endif()
        if("${source}" MATCHES "cxx*")
            string(APPEND locations_on_disk "${arg_BINARY_DIR}/${source}.o ")
        endif()
    endforeach()

    foreach(target IN LISTS arg_TARGETS)
        #message(STATUS "WORKING ON ${target}")  
        string(APPEND locations_on_disk "${target} ")
    endforeach()

    foreach(item IN LISTS arg_IMPORTED_TARGETS)
        #message(STATUS "looking for location of target ${item}")
        set(props_to_search IMPORTED_LOCATION
                            IMPORTED_LOCATION_RELEASE
                            IMPORTED_LOCATION_RELWITHDEBINFO
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

    set(${arg_RESULT} ${locations_on_disk} PARENT_SCOPE)
endfunction()


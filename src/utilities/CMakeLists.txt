###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# file: src/utilites/CMakeLists.txt
#
###############################################################################
add_subdirectory(about)
add_subdirectory(replay)
add_subdirectory(actions_conversions)
add_subdirectory(holo_compare)


# install visit scripts
install(DIRECTORY visit_session_converters
        DESTINATION utilities/ascent/visit_session_converters)

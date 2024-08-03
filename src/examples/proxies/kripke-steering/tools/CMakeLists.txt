
include_directories(.)

add_library(kripketools STATIC "testKernels.cpp")

set(KRIPKE_LIBS ${KRIPKE_LIBS} kripketools)

add_executable(kripke "kripke.cpp")
target_link_libraries(kripke  ${KRIPKE_LIBS} ${KRIPKE_LIBS})


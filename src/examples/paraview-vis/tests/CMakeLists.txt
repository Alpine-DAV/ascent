cmake_minimum_required(VERSION 3.10)
project(AscentParaViewTests)

include(CTest)
add_custom_target(
  BuildAndRunSim
  ALL
  COMMAND ${CMAKE_SOURCE_DIR}/build_and_run_sim_with_docker.sh 4 -j40 ^mpich | tee build_and_run_sim.log 2>&1
  DEPENDS build_and_run_sim_with_docker.sh build_and_run_sim_inside_docker.sh
    build_and_run_sim.sh
  BYPRODUCTS build_and_run_sim.log
  )
add_test(
  NAME BuildAndRunSim
  COMMAND cat build_and_run_sim.log
  )
set_tests_properties(BuildAndRunSim
  PROPERTIES PASS_REGULAR_EXPRESSION
  "All Ascent ParaView tests passed;Already up to date;No new package installed, no need to run tests"
  )

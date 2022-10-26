//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_expressions.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent_expression_eval.hpp>
#include <expressions/ascent_blueprint_architect.hpp>
#include <runtimes/expressions/ascent_memory_manager.hpp>
#include <runtimes/expressions/ascent_conduit_reductions.hpp>
#include <runtimes/expressions/ascent_execution.hpp>

#include <cmath>
#include <iostream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

index_t EXAMPLE_MESH_SIDE_DIM = 5;



//-----------------------------------------------------------------------------
TEST(ascent_conduit_reductions, go)
{
  conduit::Node input, res;

  std::cout << "[max]" << std::endl;
  std::cout << "input" << std::endl;
  input["values"].parse("[0,1,2,3,4,5]","yaml");
  input.print();
  res = ascent::runtime::expressions::array_max(input["values"],ExecutionManager::execution());
  std::cout << "RESULT:" << std::endl;
  res.print();
  EXPECT_EQ(res.to_float64(), 5.0);

  std::cout << "[min]" << std::endl;
  std::cout << "input" << std::endl;
  input["values"].parse("[0,1,2,3,4,5]","yaml");
  input.print();
  res = ascent::runtime::expressions::array_min(input["values"],ExecutionManager::execution());
  std::cout << "RESULT:" << std::endl;
  res.print();
  EXPECT_EQ(res.to_float64(), 0.0);

  std::cout << "[sum]" << std::endl;
  std::cout << "input" << std::endl;
  input["values"].parse("[0,1,2,3,4,5]","yaml");
  input.print();
  res = ascent::runtime::expressions::array_sum(input["values"],ExecutionManager::execution());
  std::cout << "RESULT:" << std::endl;
  res.print();
  EXPECT_EQ(res.to_float64(), 0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0);

}

//-----------------------------------------------------------------------------
int
main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  // this is normally set in ascent::Initialize, but we
  // have to set it here so that we do the right thing with
  // device pointers
  AllocationManager::set_conduit_mem_handlers();

  // allow override of the data size via the command line
  if(argc == 2)
  {
    EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
  }

  result = RUN_ALL_TESTS();
  return result;
}


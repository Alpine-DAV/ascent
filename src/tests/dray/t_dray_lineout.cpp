// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>
#include <dray/queries/lineout.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

using namespace dray;
//---------------------------------------------------------------------------//
bool
mfem_enabled()
{
#ifdef DRAY_MFEM_ENABLED
    return true;
#else
    return false;
#endif
}

TEST (dray_scalar_renderer, dray_scalars)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  
  std::string output_path = prepare_output_dir ();

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";

  Collection collection = dray::BlueprintReader::load (root_file);

  Lineout lineout;

  lineout.samples(10);
  lineout.add_var("density");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.5f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.5f}};
  lineout.add_line(start, end);

  Lineout::Result res = lineout.execute(collection);
  for(int i = 0; i < res.m_values[0].size(); ++i)
  {
    std::cout<<"Value "<<i<<" "<<res.m_values[0].get_value(i)<<"\n";
  }

}

TEST (dray_locate_2d, dray_locate)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  
  std::string output_path = prepare_output_dir ();

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green_2d.cycle_000050.root";

  Collection collection = dray::BlueprintReader::load (root_file);

  Lineout lineout;

  lineout.samples(10);
  lineout.add_var("density");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.0f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.0f}};
  lineout.add_line(start, end);

  Lineout::Result res = lineout.execute(collection);
  for(int i = 0; i < res.m_values[0].size(); ++i)
  {
    std::cout<<"Value "<<i<<" "<<res.m_values[0].get_value(i)<<"\n";
  }
}

TEST (dray_locate_2d, dray_lineout_partial_failure)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  
  std::string output_path = prepare_output_dir ();

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green_2d.cycle_000050.root";

  Collection collection = dray::BlueprintReader::load (root_file);

  Lineout lineout;

  lineout.samples(10);
  lineout.add_var("density");
  lineout.add_var("bananas");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.0f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.0f}};
  lineout.add_line(start, end);

  Lineout::Result res = lineout.execute(collection);
}

TEST (dray_locate_2d, dray_lineout_failure)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  
  std::string output_path = prepare_output_dir ();

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green_2d.cycle_000050.root";

  Collection collection = dray::BlueprintReader::load (root_file);

  Lineout lineout;

  lineout.samples(10);
  lineout.add_var("bananas");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.0f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.0f}};
  lineout.add_line(start, end);

  try
  {
    Lineout::Result res = lineout.execute(collection);
  }
  catch(std::exception &e)
  {
    std::cout<<e.what()<<"\n";
  }

}

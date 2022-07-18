//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_vtkh_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"
#include <iostream>
#include <vtkm/cont/DataSet.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>

typedef vtkm::cont::ArrayHandleUniformPointCoordinates UniformCoords;

//-----------------------------------------------------------------------------
TEST(vtkh_smoke, about_vtkh)
{
    std::cout << vtkh::AboutVTKH() << std::endl;
}


//-----------------------------------------------------------------------------
TEST(vtkh_smoke, vtkh_simple_dset)
{
  vtkm::Vec<vtkm::Float32,3> origin;
  origin[0] = 0.0;
  origin[1] = 0.0;
  origin[2] = 0.0;

  vtkm::Vec<vtkm::Float32,3> spacing(1.f, 1.f, 1.f);

  vtkm::Id3 point_dims;
  point_dims[0] = 11;
  point_dims[1] = 11;
  point_dims[2] = 11;

  vtkm::Id3 cell_dims;
  cell_dims[0] = point_dims[0] - 1;
  cell_dims[1] = point_dims[1] - 1;
  cell_dims[2] = point_dims[2] - 1;

  vtkm::cont::DataSet data_set;

  UniformCoords point_handle(point_dims,
                             origin,
                             spacing);

  vtkm::cont::CoordinateSystem coords("coords", point_handle);
  data_set.AddCoordinateSystem(coords);

  vtkm::cont::CellSetStructured<3> cell_set;
  cell_set.SetPointDimensions(point_dims);
  data_set.SetCellSet(cell_set);

  vtkh::DataSet res;

  res.AddDomain(data_set, 0);

  res.AddConstantPointField(42.0, "myfield");

  vtkm::cont::ArrayHandle<vtkm::Range> range = res.GetGlobalRange("myfield");
  EXPECT_EQ(1, range.GetNumberOfValues());

  EXPECT_EQ(range.ReadPortal().Get(0).Min,42.0);
  EXPECT_EQ(range.ReadPortal().Get(0).Max,42.0);

}


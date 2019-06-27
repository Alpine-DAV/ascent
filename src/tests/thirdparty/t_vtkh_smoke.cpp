//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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

  vtkm::cont::CellSetStructured<3> cell_set("cells");
  cell_set.SetPointDimensions(point_dims);
  data_set.AddCellSet(cell_set);

  vtkh::DataSet res;

  res.AddDomain(data_set, 0);

  res.AddConstantPointField(42.0, "myfield");

  vtkm::cont::ArrayHandle<vtkm::Range> range = res.GetGlobalRange("myfield");
  EXPECT_EQ(1, range.GetPortalControl().GetNumberOfValues());

  EXPECT_EQ(range.GetPortalControl().Get(0).Min,42.0);
  EXPECT_EQ(range.GetPortalControl().Get(0).Max,42.0);

}


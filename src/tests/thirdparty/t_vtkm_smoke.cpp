//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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
/// file: vtkm_smoke.cpp
///
//-----------------------------------------------------------------------------
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL

#include <iostream>
#include "gtest/gtest.h"

#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/testing/MakeTestDataSet.h>
#include <vtkm/rendering/Actor.h>
//#include <vtkm/cont/DeviceAdapter.h>
#include <iostream>


//-----------------------------------------------------------------------------
TEST(vtkm_smoke, headers_work)
{
    vtkm::cont::DataSet *res = NULL;
    EXPECT_EQ(1, 1);
}

//-----------------------------------------------------------------------------
TEST(vtkm_smoke, basic_use)
{
    vtkm::cont::testing::MakeTestDataSet maker;
    vtkm::cont::DataSet data = maker.Make3DExplicitDataSet2();
    
    std::vector<vtkm::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    vtkm::cont::Field scalarField("some_field",
                                  vtkm::cont::Field::ASSOC_POINTS,
                                  "cell_set",
                                  scalars);
    const vtkm::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    std::cout<<"before actor\n";
    data.PrintSummary(std::cout);
    vtkm::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);
    std::cout<<"After actor\n";
    vtkm::Bounds coordsBounds; // Xmin,Xmax,Ymin..
    coordsBounds = actor.GetSpatialBounds();

    //should be [0,1,0,1,0,1];
    
    std::cout <<  coordsBounds.X.Min << " " <<
                  coordsBounds.X.Max << " " <<
                  coordsBounds.Y.Min << " " <<
                  coordsBounds.Y.Max << " " << 
                  coordsBounds.Z.Min << " " << 
                  coordsBounds.Z.Max << std::endl;
                      
    EXPECT_NEAR(coordsBounds.X.Min, 0.0, 1e-3 );
    EXPECT_NEAR(coordsBounds.X.Max, 1.0, 1e-3 );
    
    EXPECT_NEAR(coordsBounds.Y.Min, 0.0, 1e-3 );
    EXPECT_NEAR(coordsBounds.Y.Max, 1.0, 1e-3 );
    
    EXPECT_NEAR(coordsBounds.Z.Min, 0.0, 1e-3 );
    EXPECT_NEAR(coordsBounds.Z.Max, 1.0, 1e-3 );

}




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_vtkm_smoke.cpp
///
//-----------------------------------------------------------------------------
#include <iostream>
#include "gtest/gtest.h"

#include <vtkm/cont/DataSet.h>
#include <vtkm/rendering/Actor.h>
#include <iostream>
#include "t_vtkm_test_utils.hpp"

//-----------------------------------------------------------------------------
TEST(vtkm_smoke, headers_work)
{
    vtkm::cont::DataSet *res;
    res = NULL;
    EXPECT_EQ(1, 1);
}

//-----------------------------------------------------------------------------
TEST(vtkm_smoke, basic_use_serial)
{
    vtkm::cont::RuntimeDeviceTracker &device_tracker
      = vtkm::cont::GetRuntimeDeviceTracker();
    device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagSerial());

    vtkm::cont::DataSet data = Make3DExplicitDataSet2();
    //
    // work around for a problem adding scalar fields of size 1
    // to Actors
    //
    std::vector<vtkm::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    vtkm::cont::Field scalarField = vtkm::cont::make_Field("some_field",
                                                           vtkm::cont::Field::Association::Cells,
                                                           scalars,
                                                           vtkm::CopyFlag::On);

    const vtkm::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    vtkm::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);

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
#ifdef VTKH_FORCE_OPENMP
TEST(vtkm_smoke, basic_use_openmp)
{
    vtkm::cont::RuntimeDeviceTracker &device_tracker
      = vtkm::cont::GetRuntimeDeviceTracker();
    device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagOpenMP());

    vtkm::cont::DataSet data = Make3DExplicitDataSet2();
    //
    // work around for a problem adding scalar fields of size 1
    // to Actors
    //
    std::vector<vtkm::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    vtkm::cont::Field scalarField = vtkm::cont::make_Field("some_field",
                                                           vtkm::cont::Field::Association::Cells,
                                                           scalars,
                                                           vtkm::CopyFlag::Off);

    const vtkm::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    vtkm::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);

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
#endif

#ifdef VTKH_FORCE_CUDA
TEST(vtkm_smoke, basic_use_cuda)
{
    vtkm::cont::RuntimeDeviceTracker &device_tracker
      = vtkm::cont::GetRuntimeDeviceTracker();
    device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagCuda());

    vtkm::cont::DataSet data = Make3DExplicitDataSet2();
    //
    // work around for a problem adding scalar fields of size 1
    // to Actors
    //
    std::vector<vtkm::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    vtkm::cont::Field scalarField = vtkm::cont::make_Field("some_field",
                                                           vtkm::cont::Field::Association::Cells,
                                                           scalars,
                                                           vtkm::CopyFlag::Off);

    const vtkm::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    vtkm::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);

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
#endif

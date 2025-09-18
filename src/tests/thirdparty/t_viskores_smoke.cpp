//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_viskores_smoke.cpp
///
//-----------------------------------------------------------------------------
#include <iostream>
#include "gtest/gtest.h"

#include <viskores/cont/DataSet.h>
#include <viskores/rendering/Actor.h>
#include <iostream>
#include "t_viskores_test_utils.hpp"

//-----------------------------------------------------------------------------
TEST(viskores_smoke, headers_work)
{
    viskores::cont::DataSet *res;
    res = NULL;
    EXPECT_EQ(1, 1);
}

//-----------------------------------------------------------------------------
TEST(viskores_smoke, basic_use_serial)
{
    viskores::cont::RuntimeDeviceTracker &device_tracker
      = viskores::cont::GetRuntimeDeviceTracker();
    device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagSerial());

    viskores::cont::DataSet data = Make3DExplicitDataSet2();
    //
    // work around for a problem adding scalar fields of size 1
    // to Actors
    //
    std::vector<viskores::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    viskores::cont::Field scalarField = viskores::cont::make_Field("some_field",
                                                           viskores::cont::Field::Association::Cells,
                                                           scalars,
                                                           viskores::CopyFlag::On);

    const viskores::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    viskores::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);

    viskores::Bounds coordsBounds; // Xmin,Xmax,Ymin..
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
TEST(viskores_smoke, basic_use_openmp)
{
    viskores::cont::RuntimeDeviceTracker &device_tracker
      = viskores::cont::GetRuntimeDeviceTracker();
    device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagOpenMP());

    viskores::cont::DataSet data = Make3DExplicitDataSet2();
    //
    // work around for a problem adding scalar fields of size 1
    // to Actors
    //
    std::vector<viskores::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    viskores::cont::Field scalarField = viskores::cont::make_Field("some_field",
                                                           viskores::cont::Field::Association::Cells,
                                                           scalars,
                                                           viskores::CopyFlag::Off);

    const viskores::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    viskores::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);

    viskores::Bounds coordsBounds; // Xmin,Xmax,Ymin..
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
TEST(viskores_smoke, basic_use_cuda)
{
    viskores::cont::RuntimeDeviceTracker &device_tracker
      = viskores::cont::GetRuntimeDeviceTracker();
    device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda());

    viskores::cont::DataSet data = Make3DExplicitDataSet2();
    //
    // work around for a problem adding scalar fields of size 1
    // to Actors
    //
    std::vector<viskores::Float32> scalars;
    scalars.push_back(0);
    scalars.push_back(1);
    viskores::cont::Field scalarField = viskores::cont::make_Field("some_field",
                                                           viskores::cont::Field::Association::Cells,
                                                           scalars,
                                                           viskores::CopyFlag::Off);

    const viskores::cont::CoordinateSystem coords = data.GetCoordinateSystem();
    viskores::rendering::Actor actor(data.GetCellSet(),
                                 data.GetCoordinateSystem(),
                                 scalarField);

    viskores::Bounds coordsBounds; // Xmin,Xmax,Ymin..
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

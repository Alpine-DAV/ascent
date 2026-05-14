//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_viskores_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_threshold, vtkh_serial_threshold)
{
#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::Threshold thresher;
  thresher.SetInput(&data_set);
  thresher.SetField("point_data_Float64");

  double upper_bound = (float)base_size * (float)num_blocks * 0.5f;
  double lower_bound = 0;

  thresher.SetFieldUpperThreshold(upper_bound);
  thresher.SetFieldLowerThreshold(lower_bound);
  thresher.Update();
  vtkh::DataSet *output = thresher.GetOutput();
  viskores::Bounds bounds = output->GetGlobalBounds();

  viskores::rendering::Camera camera;
  camera.SetPosition(viskores::Vec<viskores::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *output,
                                         "threshold");
  vtkh::RayTracer tracer;
  tracer.SetInput(output);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();

  delete output;
}

//----------------------------------------------------------------------------
TEST(vtkh_threshold, vtkh_threshold_point_mesh_coincident)
{
#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif

  // Create a point mesh with two coincident particles.
  // Without the fix, CleanGrid's MergePoints merges the coincident
  // pair, reducing coordinates by one but leaving both cells,
  // producing GetNumberOfCells() > GetNumberOfPoints().

  const int num_particles = 4;
  double x_vals[num_particles] = {0.0, 1.0, 1.0, 2.0};
  double y_vals[num_particles] = {0.0, 1.0, 1.0, 2.0};
  double z_vals[num_particles] = {0.0, 1.0, 1.0, 2.0};
  double field_vals[num_particles] = {1.0, 2.0, 2.0, 3.0};

  viskores::cont::DataSet ds;

  viskores::cont::ArrayHandle<viskores::Vec3f_64> coords_handle;
  coords_handle.Allocate(num_particles);
  auto coords_portal = coords_handle.WritePortal();
  for(int i = 0; i < num_particles; ++i)
  {
    coords_portal.Set(i, viskores::Vec3f_64(x_vals[i], y_vals[i], z_vals[i]));
  }
  ds.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords", coords_handle));

  viskores::cont::ArrayHandle<viskores::Id> conn;
  conn.Allocate(num_particles);
  auto conn_portal = conn.WritePortal();
  for(int i = 0; i < num_particles; ++i)
  {
    conn_portal.Set(i, i);
  }
  viskores::cont::CellSetSingleType<> cellset;
  cellset.Fill(num_particles, viskores::CELL_SHAPE_VERTEX, 1, conn);
  ds.SetCellSet(cellset);

  auto field_handle = viskores::cont::make_ArrayHandle<viskores::Float64>(
      {1.0, 2.0, 2.0, 3.0});
  ds.AddCellField("phase", field_handle);

  vtkh::DataSet data_set;
  data_set.AddDomain(ds, 0);

  // Threshold: keep particles with phase in [1.5, 3.5]
  // This keeps 3 particles, including the coincident pair.
  vtkh::Threshold thresher;
  thresher.SetInput(&data_set);
  thresher.SetField("phase");
  thresher.SetFieldLowerThreshold(1.5);
  thresher.SetFieldUpperThreshold(3.5);
  thresher.Update();

  vtkh::DataSet *output = thresher.GetOutput();

  // Verify that for every domain, cells <= points.
  // For point topology, cells > points is invalid and causes
  // out-of-bounds access in the sphere ray tracer.
  for(int i = 0; i < output->GetNumberOfDomains(); ++i)
  {
    viskores::cont::DataSet out_ds;
    viskores::Id domain_id;
    output->GetDomain(i, out_ds, domain_id);
    int nc = out_ds.GetCellSet().GetNumberOfCells();
    int np = out_ds.GetCoordinateSystem().GetNumberOfPoints();
    EXPECT_LE(nc, np) << "Point mesh after threshold has more cells ("
                      << nc << ") than points (" << np
                      << ") on domain " << domain_id;
  }

  delete output;
}

//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include "t_test_utils.hpp"

#include <iostream>
#include <mpi.h>


//----------------------------------------------------------------------------
TEST(vtkh_raytracer, vtkh_serial_render)
{
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  vtkh::DataSet data_set;

  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
  }

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "multi_par");
  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("cell_data_Float64");

  vtkm::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, .1);
  color_map.AddPointAlpha(1.0, .3);

  vtkh::VolumeRenderer v_tracer;
  v_tracer.SetColorTable(color_map);
  v_tracer.SetInput(&data_set);
  v_tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&v_tracer);
  scene.AddRenderer(&tracer);
  scene.Render();

  delete iso_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_multi_par, vtkh_batch)
{
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  vtkh::DataSet data_set;

  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
  }

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "multi_par");
  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("cell_data_Float64");

  vtkm::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, .1);
  color_map.AddPointAlpha(1.0, .3);

  vtkh::VolumeRenderer v_tracer;
  v_tracer.SetColorTable(color_map);
  v_tracer.SetInput(&data_set);
  v_tracer.SetField("point_data_Float64");

  const int num_images = 11;
  std::vector<vtkh::Render> renders;
  for(int i = 0; i < num_images; ++i)
  {
    vtkh::Render tmp = render.Copy();
    camera.Azimuth(float(i));
    tmp.SetCamera(camera);
    std::stringstream name;
    name << "par_batch_"<<i;
    tmp.SetImageName(name.str());
    renders.push_back(tmp);
  }

  vtkh::Scene scene;
  scene.SetRenderBatchSize(5);
  scene.SetRenders(renders);
  scene.AddRenderer(&v_tracer);
  scene.AddRenderer(&tracer);
  scene.Render();

  delete iso_output;
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}



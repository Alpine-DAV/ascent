//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <vtkh_marching_cubes.hpp>
#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include <rendering/vtkh_renderer_volume.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_raytracer, vtkh_serial_render)
{
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPIComm(MPI_COMM_WORLD);

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
  marcher.SetField("point_data"); 

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data");
  marcher.AddMapField("cell_data");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender<vtkh::RayTracer>(512, 
                                                          512, 
                                                          camera, 
                                                          data_set, 
                                                          "multi_par");  
  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.AddRender(render);
  tracer.SetField("cell_data"); 
  // composite only on the last renderer 
  tracer.SetDoComposite(false); 
  tracer.Update();

  vtkm::rendering::ColorTable color_map("cool2warm"); 
  color_map.AddAlphaControlPoint(0.0, .01);
  color_map.AddAlphaControlPoint(1.0, .01);

  vtkh::VolumeRenderer v_tracer;
  v_tracer.SetColorTable(color_map);
  v_tracer.AddRender(render);
  v_tracer.SetInput(&data_set);
  v_tracer.SetField("point_data"); 

  v_tracer.Update();

  delete iso_output; 
  MPI_Finalize();
}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/dray.hpp>
#include <dray/filters/redistribute.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <mpi.h>

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


TEST (dray_redistribute, redistribute)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
    conduit::utils::join_file_path (output_path, "redistribute");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  int rank = dray::dray::mpi_rank();
  int size = dray::dray::mpi_size();


  int domains = dataset.size();
  int local_domains = dataset.local_size();

  std::vector<int32> dom_counts;
  dom_counts.resize(size);
  MPI_Allgather(&local_domains, 1, MPI_INT, &dom_counts[0], 1, MPI_INT, comm);

  std::vector<int32> dom_offsets;
  dom_offsets.resize(size);
  dom_offsets[0] = 0;

  for(int i = 1; i < size; ++i)
  {
    dom_offsets[i] = dom_offsets[i-1] + dom_counts[i-1];
  }

  // create a round robin schedule
  std::vector<int> src_list(domains);
  std::vector<int> dest_list(domains);

  int src = 0;
  for(int i = 0; i < domains; ++i)
  {
    if((src != size -1 ) && dom_offsets[src+1] <= i )
    {
      src += 1;
    }

    int dest = (src + 1) % size;
    src_list[i] = src;
    dest_list[i] = dest;
    if(rank == 0)
    {
      std::cout<<"src "<<src<<" dest "<<dest<<"\n";
    }
  }

  dray::Redistribute redist;
  dray::Collection res = redist.execute(dataset, src_list, dest_list);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(res);

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(20);
  camera.elevate(10);

  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  if(dray::dray::mpi_rank() == 0)
  {
    fb.composite_background();
    fb.save (output_file);
    // note: dray diff tolerance was 0.2f prior to import
    EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
  }
}

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}

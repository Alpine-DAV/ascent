// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "t_utils.hpp"
#include "t_config.hpp"

#include <stdio.h>

#include <dray/GridFunction/grid_function.hpp>
#include <dray/aabb.hpp>
#include <dray/filters/attractor_map.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/utils/png_encoder.hpp>


const int grid_depth_2d = 10; // 1024x1024
const int c_width = 1 << grid_depth_2d;
const int c_height = 1 << grid_depth_2d;

const int grid_depth_3d = 6; // 64x64x64
const int v_width = 1 << grid_depth_3d;
const int v_height = 1 << grid_depth_3d;
const int v_depth = 1 << grid_depth_3d;

const int num_frames = 24;

#include <fstream>
#include <iostream>
void write_attractor_vtk_image (const char *output_name,
                                const int nx,
                                const int ny,
                                const int nz,
                                const dray::Vec<float, 3> *solutions,
                                const dray::int32 *iterations);

void write_attractor_vtk_image (const int nx,
                                const int ny,
                                const int nz,
                                const dray::Vec<float, 3> *solutions,
                                const dray::int32 *iterations)
{
  write_attractor_vtk_image ("attractors.vtk", nx, ny, nz, solutions, iterations);
}

void write_attractor_vtk_image (const char *output_name,
                                const int nx,
                                const int ny,
                                const int nz,
                                const dray::Vec<float, 3> *solutions,
                                const dray::int32 *iterations)
{
  const int num_cells = nx * ny * nz;

  const double space_x = (nx > 1 ? 1.0 / (nx - 1) : 1);
  const double space_y = (ny > 1 ? 1.0 / (ny - 1) : 1);
  const double space_z = (nz > 1 ? 1.0 / (nz - 1) : 1);

  std::ofstream file;
  file.open (output_name);
  file << "# vtk DataFile Version 3.0\n";
  file << "attractors\n";
  file << "ASCII\n";
  file << "DATASET STRUCTURED_POINTS\n";
  file << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << "\n";
  file << "ORIGIN 0 0 0\n";
  file << "SPACING " << space_x << " " << space_y << " " << space_z << "\n";

  file << "CELL_DATA " << num_cells << "\n";

  file << "SCALARS iterations int 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int i = 0; i < num_cells; ++i)
  {
    file << iterations[i] << "\n";
  }

  file << "SCALARS edge_dist float 1\n";
  file << "LOOKUP_TABLE default\n";
  for (int i = 0; i < num_cells; ++i)
  {
    file << dray::AttractorMapShader::edge_dist (solutions[i]) << "\n";
  }

  file << "VECTORS attractor float\n";
  file << "LOOKUP_TABLE default\n";
  for (int i = 0; i < num_cells; ++i)
  {
    file << solutions[i][0] << " " << solutions[i][1] << " " << solutions[i][2] << "\n";
  }

  file.close ();
}


//
// sample_path_linear()
//
// Note: If num_samples is >= 2, then a path is stored. If num_samples == 1, then path[0] = start.
//
void sample_path_linear (const dray::Vec<float, 3> &start,
                         const dray::Vec<float, 3> &end,
                         const int num_samples,
                         dray::Vec<float, 3> *path);


TEST (dray_attractors, dray_attractors_2d)
{
  std::string file_name = std::string (ASCENT_T_DATA_DIR) + "warbly_cube/warbly_cube";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "warbly_cube_attractors");
  remove_test_image (output_file);

  // Get mesh/cell.
  auto dataset = dray::MFEMReader::load (file_name);

  /// // What coordinates are actually inside the first cell?
  /// {
  ///   dray::AABB<3> cell_bounds;
  ///   dataset.get_mesh().access_host_mesh().get_elem(0).get_bounds(cell_bounds);
  ///   std::cout << "First element bounds: " << cell_bounds << "\n";
  /// }

  const int el_id = 0; // Use one cell for all queries/guesses.

  // Define query point.
  const dray::Vec<float, 3> query_point ({ 1.0, 1.0, 1.0 });
  /// const dray::Vec<float,3> query_point({14.0/23.0, .5, .5});

  /// // Query point produced from a point that is definitely inside or on the element.
  /// dray::Vec<float,3> query_point;
  /// dray::Vec<dray::Vec<float,3>,3> unused_deriv;
  /// dataset.get_mesh().access_host_mesh().get_elem(el_id).eval({1.0, 1.0, 1.0}, query_point, unused_deriv);

  // Define collection of sample initial guesses.
  const dray::Array<dray::RefPoint<3>> sample_guesses =
  dray::AttractorMap::domain_grid_slice_xy (grid_depth_2d, grid_depth_2d, 0.5, el_id);

  // Other outputs (for vtk file).
  dray::Array<dray::Vec<float, 3>> solutions;
  dray::Array<int> iterations;

  // Get image.
  dray::AttractorMap attractor_map_filter;
  dray::Array<dray::Vec<dray::float32, 4>> color_buffer =
  attractor_map_filter.execute (true, query_point, sample_guesses, solutions,
                                iterations, dataset);

  // Encode image.
  dray::PNGEncoder png_encoder;
  png_encoder.encode ((float *)color_buffer.get_host_ptr (), c_width, c_height);
  png_encoder.save (output_file + ".png");

  /// // If one or more solutions were found inside the element, what was the first one?
  /// const int solutions_size = solutions.size();
  /// const dray::Vec<float,3> *solutions_ptr = solutions.get_host_ptr();
  /// int sidx;
  /// for (sidx = 0; sidx < solutions_size; sidx++)
  /// {
  ///   if (dray::MeshElem<float,3>::is_inside(solutions_ptr[sidx]))
  ///     break;
  /// }
  /// if (sidx < solutions_size)
  ///   std::cout << "Solution found: sidx==" << sidx << " \t ref_coords==" << solutions_ptr[sidx] << "\n";

  // Dump VTK file of (solutions, iterations).
  write_attractor_vtk_image ("attractors-2d.vtk", c_width, c_height,
                             1, // 2D slab image.
                             solutions.get_host_ptr_const (),
                             iterations.get_host_ptr_const ());

  // Check against benchmark.
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}


TEST (dray_attractors, dray_attractors_3d)
{
  std::string file_name = std::string (ASCENT_T_DATA_DIR) + "warbly_cube/warbly_cube";

  // Get mesh/cell.
  auto dataset = dray::MFEMReader::load (file_name);

  const int el_id = 0; // Use one cell for all queries/guesses.

  // Define collection of sample initial guesses.
  const dray::Array<dray::RefPoint<3>> sample_guesses =
  dray::AttractorMap::domain_grid_3d (grid_depth_3d, grid_depth_3d, grid_depth_3d, el_id);

  // Outputs for vtk file.
  dray::Array<dray::Vec<float, 3>> solutions;
  dray::Array<int> iterations;

  // Path of query points.
  dray::Vec<float, 3> query_points[num_frames];
  /// sample_path_linear({0.0, 0.27, 1.000}, {1.0, 0.27, 0.600}, num_frames, query_points);
  sample_path_linear ({ 0.0, 0.5, 0.5 }, { 1.0, 0.5, 0.5 }, num_frames, query_points);

  // Create the .visit file to represent a time series.
  std::ofstream visit_file;
  visit_file.open ("attractors-3d.visit");
  for (int t = 0; t < num_frames; t++)
  {
    char outfilename[] = "attractors-3d-t000000.vtk";
    char *frame_num_ptr = outfilename + 15;
    snprintf (frame_num_ptr, 6 + 5, "%06d.vtk", t);
    visit_file << outfilename << "\n";
  }
  visit_file.close ();

  dray::AttractorMap attractor_map_filter;

  for (int t = 0; t < num_frames; t++)
  {
    char outfilename[] = "attractors-3d-t000000.vtk";
    char *frame_num_ptr = outfilename + 15;
    snprintf (frame_num_ptr, 6 + 5, "%06d.vtk", t);

    // Get results.
    attractor_map_filter.execute (false, query_points[t], sample_guesses,
                                  solutions, iterations, dataset);

    // Dump VTK file of (solutions, iterations).
    write_attractor_vtk_image (outfilename, v_width, v_height,
                               v_depth, // 3D block.
                               solutions.get_host_ptr_const (),
                               iterations.get_host_ptr_const ());
  }
}


//
// sample_path_linear()
//
void sample_path_linear (const dray::Vec<float, 3> &start,
                         const dray::Vec<float, 3> &end,
                         const int num_samples,
                         dray::Vec<float, 3> *path)
{
  const int divisor = (num_samples > 1 ? num_samples - 1 : 1);
  for (int i = 0; i <= num_samples - 1; i++)
  {
    double blend = 1.0 * i / divisor;
    path[i] = start * (1.0 - blend) + end * blend;
  }
}

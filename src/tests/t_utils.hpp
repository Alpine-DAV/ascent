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
/// file: t_ascent_test_utils.hpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef T_ASCENT_DATA
#define T_ASCENT_DATA
//-----------------------------------------------------------------------------

#include <iostream>
#include <math.h>

#include "t_config.hpp"
#include <ascent.hpp>
#include <utils/ascent_png_compare.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
void
remove_test_image(const std::string &path, const std::string num = "100")
{
    if(conduit::utils::is_file(path + num + ".png"))
    {
        conduit::utils::remove_file(path + num + ".png");
    }

    if(conduit::utils::is_file(path + num + ".pnm"))
    {
        conduit::utils::remove_file(path + num + ".pnm");
    }

}

//-----------------------------------------------------------------------------
void
remove_test_file(const std::string &path)
{
    if(conduit::utils::is_file(path))
    {
        conduit::utils::remove_file(path);
    }
}

//-----------------------------------------------------------------------------
std::string
prepare_output_dir()
{
    string output_path = ASCENT_T_BIN_DIR;

    output_path = conduit::utils::join_file_path(output_path,"_output");

    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    return output_path;
}

//----------------------------------------------------------------------------
std::string
output_dir()
{
    return conduit::utils::join_file_path(ASCENT_T_BIN_DIR,"_output");;
}

//-----------------------------------------------------------------------------
bool
check_test_image(const std::string &path, const float tolerance = 0.001f, std::string num = "100")
{
    Node info;
    std::string png_path = path + num + ".png";
    // for now, just check if the file exists.
    bool res = conduit::utils::is_file(png_path);
    info["test_file/path"] = png_path;
    if(res)
    {
      info["test_file/exists"] = "true";
    }
    else
    {
      info["test_file/exists"] = "false";
    }

    std::string file_name;
    std::string path_b;

    conduit::utils::rsplit_file_path(png_path,
                                     file_name,
                                     path_b);

    string baseline_dir = conduit::utils::join_file_path(ASCENT_T_SRC_DIR,"baseline_images");
    string baseline = conduit::utils::join_file_path(baseline_dir,file_name);

    info["baseline_file/path"] = baseline;
    if(conduit::utils::is_file(baseline))
    {
      info["baseline_file/exists"] = "true";
    }
    else
    {
      info["baseline_file/exists"] = "false";
    }

    ascent::PNGCompare compare;

    res &= compare.Compare(png_path, baseline, info, tolerance);

    if(!res)
    {
      info.print();
    }
    std::string info_fpath = path + num + "_img_compare_results.json";
    info.save(info_fpath,"json");

    return res;
}

bool
check_test_file(const std::string &path)
{
    // for now, just check if the file exists.
    return conduit::utils::is_file(path);
}


//-----------------------------------------------------------------------------
// create an example 2d rectilinear grid with two variables.
//-----------------------------------------------------------------------------
void
create_2d_example_dataset(Node &data,
                          int par_rank=0,
                          int par_size=1)
{
    const float64 PI_VALUE = 3.14159265359;

    // if( (par_size > 1)  && ((par_size % par_rank) != 0))
    // {
    //     ASCENT_ERROR("par_size ("  << par_size << ") " <<
    //                    "must must divide evenly into " <<
    //                    "par_rank (" << par_rank << ")");
    // }

    int size = 20;

    int nx = size;
    int ny = size;

    float64 dx = 1;
    float64 dy = 1;

    index_t npts = (nx+1)*(ny+1);
    index_t nele = nx*ny;


    data["state/time"]   = (float64)3.1415;
    data["state/domain_id"] = (uint64) par_rank;
    data["state/cycle"]  = (uint64) 100;

    data["coordsets/coords/type"] = "rectilinear";
    data["coordsets/coords/values/x"].set(DataType::float64(nx+1));
    data["coordsets/coords/values/y"].set(DataType::float64(ny+1));

    data["topologies/mesh/type"] = "rectilinear";
    data["topologies/mesh/coordset"] = "coords";

    data["fields/radial_vert/type"] = "scalar";
    data["fields/radial_vert/topology"] = "mesh";
    data["fields/radial_vert/association"] = "vertex";
    data["fields/radial_vert/values"].set(DataType::float64(npts));

    data["fields/radial_ele/type"] = "scalar";
    data["fields/radial_ele/topology"] = "mesh";
    data["fields/radial_ele/association"] = "element";
    data["fields/radial_ele/values"].set(DataType::float64(nele));


    float64 *x_vals =  data["coordsets/coords/values/x"].value();
    float64 *y_vals =  data["coordsets/coords/values/y"].value();

    float64 *point_scalar   = data["fields/radial_vert/values"].value();
    float64 *element_scalar = data["fields/radial_ele/values"].value();

    float64 start = 0.0 - (float64)(size) / 2.0;

    for (int i = 0; i < nx+1; ++i)
        x_vals[i] = start + i * dx;
    for (int j = 0; j < ny+1; ++j)
        y_vals[j] = start + j * dy;

    index_t idx = 0;

    float64 fsize      = (float64) size;
    float64 fhalf_size = .5 * fsize;

    for (int i = 0; i < ny + 1; ++i)
    {
        float64 cy = y_vals[i];

        for(int k = 0; k < nx +1; ++k)
        {
            float64 cx = x_vals[k];
            point_scalar[idx] = sin( (2 * PI_VALUE * cx) / fhalf_size) +
                                sin( (2 * PI_VALUE * cy) / fsize );
            idx++;
        }

    }


    dx = fsize / float64(nx-1);
    dy = fsize / float64(ny-1);

    idx = 0;
    for(int i = 0; i < ny ; ++i)
    {
        float64 cy = y_vals[i];
        for(int k = 0; k < nx; ++k)
        {
            float64 cx = (i * dx) + -fhalf_size;
            float64 cv = fhalf_size * sqrt( cx*cx + cy*cy );

            element_scalar[idx] = cv;

            idx++;
        }
    }
}

//-----------------------------------------------------------------------------
// create an example 3d rectilinear grid with two variables.
//-----------------------------------------------------------------------------
void
create_3d_example_dataset(Node &data,
                          int cell_dim,
                          int par_rank,
                          int par_size)
{
    // if( (par_size > 1)  && ((par_size % par_rank) != 0))
    // {
    //     ASCENT_ERROR("par_size ("  << par_size << ") " <<
    //                    "must must divide evenly into " <<
    //                    "par_rank (" << par_rank << ")");
    // }

    int cellsPerRank = cell_dim;
    int size = par_size * cellsPerRank;

    int nx = size / par_size;
    int ny = size;
    int nz = size;

    float64 dx = 1;
    float64 dy = 1;
    float64 dz = 1;

    index_t npts = (nx+1)*(ny+1)*(nz+1);
    index_t nele = nx*ny*nz;


    data["state/time"]   = (float64)3.1415;
    data["state/domain_id"] = (uint64) par_rank;
    data["state/cycle"]  = (uint64) 100;
    data["coordsets/coords/type"] = "rectilinear";

    data["coordsets/coords/values/x"].set(DataType::float64(nx+1));
    data["coordsets/coords/values/y"].set(DataType::float64(ny+1));
    data["coordsets/coords/values/z"].set(DataType::float64(nz+1));

    data["topologies/mesh/type"] = "rectilinear";
    data["topologies/mesh/coordset"] = "coords";

    data["fields/radial_vert/association"] = "vertex";
    data["fields/radial_vert/topology"] = "mesh";
    data["fields/radial_vert/values"].set(DataType::float64(npts));

    data["fields/radial_ele/association"] = "element";
    data["fields/radial_ele/topology"] = "mesh";
    data["fields/radial_ele/values"].set(DataType::float64(nele));

    data["fields/rank_ele/association"] = "element";
    data["fields/rank_ele/topology"] = "mesh";
    data["fields/rank_ele/values"].set(DataType::float64(nele));


    float64 *x_vals =  data["coordsets/coords/values/x"].value();
    float64 *y_vals =  data["coordsets/coords/values/y"].value();
    float64 *z_vals =  data["coordsets/coords/values/z"].value();

    float64 *point_scalar   = data["fields/radial_vert/values"].value();
    float64 *element_scalar = data["fields/radial_ele/values"].value();

    float64 *rank_scalar = data["fields/rank_ele/values"].value();

    for(int i=0;i < nele;i++)
    {
        rank_scalar[i] = (float64)par_rank;
    }

    float64 start = 0.0 - (float64)(size) / 2.0;
    float64 rank_offset = start + (float)(par_rank * nx);

    for (int i = 0; i < nx+1; ++i)
        x_vals[i] = rank_offset + i * dx;
    for (int j = 0; j < ny+1; ++j)
        y_vals[j] = start + j * dy;
    for (int k = 0; k < nz + 1; ++k)
        z_vals[k] = start / 2.f + k * dz;

    index_t idx = 0;
    for (int j = 0; j < nz + 1; ++j)
    {
        float64 cz = z_vals[j];
        for (int i = 0; i < ny + 1; ++i)
        {
            float64 cy = y_vals[i];
            for(int k = 0; k < nx +1; ++k)
            {
                float64 cx = x_vals[k];
                point_scalar[idx] = 10.0 * sqrt( cx*cx + cy*cy + cz*cz);
                idx++;
            }

        }
    }

    dx = (float64)(size) / float64(nx-1);
    dy = (float64)(size) / float64(ny-1);
    dz = (float64)(size) / float64(nz-1);

    idx = 0;
    for(int j = 0; j < nz ; ++j)
    {
        float64 cz = z_vals[j];
        for(int i = 0; i < ny ; ++i)
        {
            float64 cy = y_vals[i];
            for(int k = 0; k < nx; ++k)
            {
                float64 cx = x_vals[k];
                float64 cv = 10.0 *sqrt( cx*cx + cy*cy + cz*cz);
                element_scalar[idx] = cv;
                idx++;
            }
        }
    }
}

void add_interleaved_vector(conduit::Node &dset)
{
  int dims = dset["fields/vel/values"].number_of_children();
  if(dims != 2 && dims != 3)
  {
    return;
  }

  Node &in_field = dset["fields/vel/values"];

  int nvals = in_field["u"].dtype().number_of_elements();

  index_t stride = sizeof(conduit::float64) * dims;
  Schema s;
  index_t size = sizeof(conduit::float64);
  s["u"].set(DataType::float64(nvals,0,stride));
  s["v"].set(DataType::float64(nvals,size,stride));
  if(dims == 3)
  {
    s["w"].set(DataType::float64(nvals,size*2,stride));
  }

  Node &res = dset["fields/vel_interleaved/values"];
  dset["fields/vel_interleaved/association"] = dset["fields/vel/association"];
  dset["fields/vel_interleaved/topology"] = dset["fields/vel/topology"];
  // init the output
  res.set(s);

  float64_array u_a = res["u"].value();
  float64_array v_a = res["v"].value();
  float64_array w_a;

  float64_array u_in = in_field["u"].value();
  float64_array v_in = in_field["v"].value();
  float64_array w_in;
  if(dims == 3)
  {
    w_a = res["w"].value();
    w_in = in_field["w"].value();
  }

  for(index_t i=0;i<nvals;i++)
  {
      u_a[i] = u_in[i];
      v_a[i] = v_in[i];
      if(dims == 3)
      {
        w_a[i] = w_in[i];
      }
  }
}

// Macro to save ascent actions file
#define ASCENT_ACTIONS_DUMP(actions,name) \
  std::string actions_str = actions.to_yaml(); \
  std::ofstream out; \
  out.open(name+".yaml"); \
  out<<actions_str; \
  out.close();


//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------


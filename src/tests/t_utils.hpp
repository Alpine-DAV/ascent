//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
#include <png_utils/ascent_png_compare.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
inline void
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
inline void
remove_test_image_direct(const std::string &path)
{
    return remove_test_image(path,"");
}


//-----------------------------------------------------------------------------
inline void
remove_test_file(const std::string &path)
{
    if(conduit::utils::is_file(path))
    {
        conduit::utils::remove_file(path);
    }
}

//-----------------------------------------------------------------------------
inline std::string
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
inline std::string
output_dir()
{
    return conduit::utils::join_file_path(ASCENT_T_BIN_DIR,"_output");;
}

//-----------------------------------------------------------------------------
inline std::string
test_data_file(const std::string &file_name)
{
    string data_dir = conduit::utils::join_file_path(ASCENT_T_SRC_DIR,"_test_data");
    string file = conduit::utils::join_file_path(data_dir,file_name);
    return file;
}


inline std::string
dray_baselines_dir()
{
    string res = conduit::utils::join_file_path(ASCENT_T_SRC_DIR,"_baseline_images");
    return conduit::utils::join_file_path(res,"dray");
}


// NOTE: Devil Ray diff tolerance was 0.2f at time of great amalgamation 

//-----------------------------------------------------------------------------
inline bool
check_test_image(const std::string &path,
                 const std::string &baseline_dir,
                 const float tolerance = 0.001f)
{
    Node info;
    std::string png_path = path + ".png";
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
      res = false;
    }

    std::string file_name;
    std::string path_b;

    conduit::utils::rsplit_file_path(png_path,
                                     file_name,
                                     path_b);

    string baseline = conduit::utils::join_file_path(baseline_dir,file_name);

    info["baseline_file/path"] = baseline;
    if(conduit::utils::is_file(baseline))
    {
      info["baseline_file/exists"] = "true";
    }
    else
    {
      info["baseline_file/exists"] = "false";
      res = false;
    }

    if(res)
    {
      ascent::PNGCompare compare;

      res &= compare.Compare(png_path, baseline, info, tolerance);
    }

    if(!res)
    {
      info.print();
    }
    std::string info_fpath = path + "_img_compare_results.json";
    info.save(info_fpath,"json");

    return res;
}

//-----------------------------------------------------------------------------
inline bool
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
      res = false;
    }

    std::string file_name;
    std::string path_b;

    conduit::utils::rsplit_file_path(png_path,
                                     file_name,
                                     path_b);

    string baseline_dir = conduit::utils::join_file_path(ASCENT_T_SRC_DIR,"_baseline_images");
    string baseline = conduit::utils::join_file_path(baseline_dir,file_name);

    info["baseline_file/path"] = baseline;
    if(conduit::utils::is_file(baseline))
    {
      info["baseline_file/exists"] = "true";
    }
    else
    {
      info["baseline_file/exists"] = "false";
      res = false;
    }

    if(res)
    {
      ascent::PNGCompare compare;

      res &= compare.Compare(png_path, baseline, info, tolerance);
    }

    if(!res)
    {
      info.print();
    }
    std::string info_fpath = path + num + "_img_compare_results.json";
    info.save(info_fpath,"json");

    return res;
}


inline bool
check_test_file(const std::string &path)
{
    // for now, just check if the file exists.
    return conduit::utils::is_file(path);
}

//-----------------------------------------------------------------------------
// create an example 2d rectilinear grid with two variables.
//-----------------------------------------------------------------------------
inline void
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
inline void
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

    data["fields/ones_ele/association"] = "element";
    data["fields/ones_ele/topology"] = "mesh";
    data["fields/ones_ele/values"].set(DataType::float64(nele));

    data["fields/ones_vert/association"] = "vertex";
    data["fields/ones_vert/topology"] = "mesh";
    data["fields/ones_vert/values"].set(DataType::float64(npts));

    float64_array ones_ele_vals =  data["fields/ones_ele/values"].value();
    ones_ele_vals.fill(1.0);

    float64_array ones_vert_vals =  data["fields/ones_vert/values"].value();
    ones_vert_vals.fill(1.0);

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

inline void
add_interleaved_vector(conduit::Node &dset)
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

void append_ghosts(conduit::Node &data,
                   const int size,
                   const std::string ghost_name,
                   const std::string topo_name)
{
  std::vector<double> ghosts;
  ghosts.resize(size);
  const int garbage = 2;
  const int actual = 1;
  const int real = 0;

  assert(size > 3);

  for(int i = 0; i < size; ++i)
  {
    int value;
    if(i == 0)
    {
      value = garbage;
    }
    else if(i == 1)
    {
      value = actual;
    }
    else
    {
      value = real;
    }
    ghosts[i] = value;
  }

  data["fields/"+ghost_name+"/values"].set(ghosts);
  data["fields/"+ghost_name+"/association"] = "element";
  data["fields/"+ghost_name+"/topology"] = topo_name;
}

// outputs a mutli domain(size 1) multiple topo data
// set
void build_multi_topo(Node &data, const int dims)
{
  Node verify_info;
  Node &dom = data.append();

  conduit::blueprint::mesh::examples::braid("uniform",
                                            dims,
                                            dims,
                                            dims,
                                            dom);

  Node point_data;
  conduit::blueprint::mesh::examples::braid("points",
                                            dims,
                                            dims,
                                            dims,
                                            point_data);

  dom["state/domain_id"] = (int)0;

  dom["topologies/point_mesh"] = point_data["topologies/mesh"];
  dom["topologies/point_mesh/coordset"] = "point_coords";
  dom["coordsets/point_coords"] = point_data["coordsets/coords"];
  dom["fields/point_braid"] = point_data["fields/braid"];
  dom["fields/point_braid/topology/"] = "point_mesh";
  dom["fields/point_radial"] = point_data["fields/radial"];
  dom["fields/point_radial/topology/"] = "point_mesh";
  const int elements = (dims - 1) * (dims - 1) * (dims - 1);
  const int points= (dims) * (dims) * (dims);
  append_ghosts(dom, points, "point_ghosts", "point_mesh");
  append_ghosts(dom, elements, "cell_ghosts", "mesh");
  //data.print();
}


//-----------------------------------------------------------------------------
// create an example multi domain multi topo dataset
// where one topo only lives on rank 0
//-----------------------------------------------------------------------------
inline void
create_example_multi_domain_multi_topo_dataset(Node &data,
                                               int par_rank=0,
                                               int par_size=1)
{
    int dims = 5;
    data.reset();
    Node &mesh = data.append();

    // rank zero will also include braid points
    if(par_rank == 0)
    {
        conduit::blueprint::mesh::examples::braid("points",
                                                  dims,
                                                  dims,
                                                  1,
                                                  mesh);
    }

    // create the coordinate set
    mesh["coordsets/ucoords/type"] = "uniform";
    mesh["coordsets/ucoords/dims/i"] = 3;
    mesh["coordsets/ucoords/dims/j"] = 3;

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/ucoords/origin/x"] = -10.0;
    mesh["coordsets/ucoords/origin/y"] = -10.0;
    mesh["coordsets/ucoords/spacing/dx"] = 10.0;
    mesh["coordsets/ucoords/spacing/dy"] = 10.0;

    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/utopo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/utopo/coordset"] = "ucoords";

    // add a simple element-associated field
    mesh["fields/ele_example/association"] =  "element";
    // reference the topology this field is defined on by name
    mesh["fields/ele_example/topology"] =  "utopo";
    // set the field values, for this case we have 4 elements
    mesh["fields/ele_example/values"].set(DataType::float64(4));

    float64 *ele_vals_ptr = mesh["fields/ele_example/values"].value();

    for(int i=0;i<4;i++)
    {
        ele_vals_ptr[i] = float64(i);
    }

    // std::cout  << mesh.to_yaml() << std::endl;
}

//-----------------------------------------------------------------------------
void
add_matset_to_spiral(Node &n_mesh, const int ndomains)
{
    // Add a matset to each domain
    for (index_t domain_id = 0; domain_id < n_mesh.number_of_children(); domain_id ++)
    {
        Node &domain = n_mesh[domain_id];
        const auto num_elements = blueprint::mesh::topology::length(domain["topologies/topo"]);
        Node &matset = domain["matsets/matset"];
        // add a matset to it
        matset["topology"].set("topo");

        // Uni buffer requires material map
        for(index_t i = 0; i < ndomains; i ++)
        {
            const std::string mat_name("mat" + std::to_string(i));
            matset["material_map"][mat_name].set((int32) i);
        }

        Node &mat_ids = matset["material_ids"];
        mat_ids.set_dtype(DataType::index_t(num_elements));
        index_t_array ids = mat_ids.value();
        for (index_t i = 0; i < ids.number_of_elements(); i++)
        {
            ids[i] = domain_id;
        }

        Node &mat_vfs = matset["volume_fractions"];
        mat_vfs.set_dtype(DataType::c_float(num_elements));
        float_array data = mat_vfs.value();
        for (index_t i = 0; i < data.number_of_elements(); i++)
        {
            data[i] = 1.f;
        }
    }
}


// Macro to save ascent actions file
#define ASCENT_ACTIONS_DUMP(actions,name,msg) \
  std::string actions_str = actions.to_yaml(); \
  std::ofstream out; \
  out.open(name+"100"+".yaml"); \
  out<<"#"<<msg<<"\n"; \
  out<<actions_str; \
  out.close();


//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_htg_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_htg_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_metadata.hpp>
#include <ascent_mpi_utils.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_runtime_param_check.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// std includes
#include <limits>
#include <set>

using namespace std;
using namespace conduit;
using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::detail --
//-----------------------------------------------------------------------------
namespace detail
{


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::detail --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helper used by io save
//-----------------------------------------------------------------------------
bool
verify_htg_params(const conduit::Node &params,
                  conduit::Node &info)
{
    bool res = true;

    if( !params.has_child("path") )
    {
        info["errors"].append() = "missing required entry 'path'";
        res = false;
    }
    else if(!params["path"].dtype().is_string())
    {
        info["errors"].append() = "'path' must be a string";
        res = false;
    }
    else if(params["path"].as_string().empty())
    {
        info["errors"].append() = "'path' is an empty string";
        res = false;
    }

    if( !params.has_child("blank_value") )
    {
        info["errors"].append() = "missing required entry 'blank_value'";
        res = false;
    }
    else if(!params["blank_value"].dtype().is_float())
    {
        info["errors"].append() = "'blank_value' must be a float";
        res = false;
    }

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("path");
    valid_paths.push_back("fields");
    valid_paths.push_back("blank_value");
    ignore_paths.push_back("fields");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
        info["errors"].append() = surprises;
        res = false;
    }

    return res;
}

//-----------------------------------------------------------------------------
float htg_create(const float *var_in,
                 float *var_out,
                 int *mask,
                 int n_levels,
                 int nx,
                 int blank_value,
                 int level,
                 int *offsets,
                 int i_start,
                 int j_start,
                 int k_start)
{
    if (level < n_levels - 1)
    {
        // 
        // Recurse over the 8 sub-blocks.
        //
        int nx_sub_box = 1 << (n_levels - level - 1);
        int offset = offsets[level];
        offsets[level] = offsets[level] + 8;

        var_out[offset]   = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start, j_start, k_start);
        var_out[offset+1] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start + nx_sub_box, j_start, k_start);
        var_out[offset+2] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start, j_start + nx_sub_box, k_start);
        var_out[offset+3] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start + nx_sub_box, j_start + nx_sub_box, k_start);
        var_out[offset+4] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start, j_start, k_start + nx_sub_box);
        var_out[offset+5] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start + nx_sub_box, j_start, k_start + nx_sub_box);
        var_out[offset+6] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start, j_start + nx_sub_box, k_start + nx_sub_box);
        var_out[offset+7] = htg_create(var_in, var_out, mask, n_levels, nx,
            blank_value, level+1, offsets,
            i_start + nx_sub_box, j_start + nx_sub_box, k_start + nx_sub_box);
        mask[offset]   = var_out[offset]   == blank_value ? 1 : 0;
        mask[offset+1] = var_out[offset+1] == blank_value ? 1 : 0;
        mask[offset+2] = var_out[offset+2] == blank_value ? 1 : 0;
        mask[offset+3] = var_out[offset+3] == blank_value ? 1 : 0;
        mask[offset+4] = var_out[offset+4] == blank_value ? 1 : 0;
        mask[offset+5] = var_out[offset+5] == blank_value ? 1 : 0;
        mask[offset+6] = var_out[offset+6] == blank_value ? 1 : 0;
        mask[offset+7] = var_out[offset+7] == blank_value ? 1 : 0;

        //
        // Calculate and return the average.
        //
        float ave = 0.;
        int n_val = 0;
        for (int l = 0; l < 8; l++)
        {
            if (var_out[offset+l] != blank_value)
            {
                n_val++;
                ave += var_out[offset+l];
            }
        }
        if (n_val)
            ave /= float(n_val);
        else
            ave = blank_value;
        return ave;
    }
    else
    {
        //
        // Base case. Reorder and store the values as the finest
        // resolution.
        //
        int offset = offsets[level];
        offsets[level] = offsets[level] + 8;

        int index  = (k_start)   * nx * nx + (j_start)   * nx + (i_start);
        int index2 = (k_start)   * nx * nx + (j_start)   * nx + (i_start+1);
        int index3 = (k_start)   * nx * nx + (j_start+1) * nx + (i_start);
        int index4 = (k_start)   * nx * nx + (j_start+1) * nx + (i_start+1);
        int index5 = (k_start+1) * nx * nx + (j_start)   * nx + (i_start);
        int index6 = (k_start+1) * nx * nx + (j_start)   * nx + (i_start+1);
        int index7 = (k_start+1) * nx * nx + (j_start+1) * nx + (i_start);
        int index8 = (k_start+1) * nx * nx + (j_start+1) * nx + (i_start+1);
        var_out[offset]   = var_in[index];
        var_out[offset+1] = var_in[index2];
        var_out[offset+2] = var_in[index3];
        var_out[offset+3] = var_in[index4];
        var_out[offset+4] = var_in[index5];
        var_out[offset+5] = var_in[index6];
        var_out[offset+6] = var_in[index7];
        var_out[offset+7] = var_in[index8];
        mask[offset]   = var_out[offset]   == blank_value ? 1 : 0;
        mask[offset+1] = var_out[offset+1] == blank_value ? 1 : 0;
        mask[offset+2] = var_out[offset+2] == blank_value ? 1 : 0;
        mask[offset+3] = var_out[offset+3] == blank_value ? 1 : 0;
        mask[offset+4] = var_out[offset+4] == blank_value ? 1 : 0;
        mask[offset+5] = var_out[offset+5] == blank_value ? 1 : 0;
        mask[offset+6] = var_out[offset+6] == blank_value ? 1 : 0;
        mask[offset+7] = var_out[offset+7] == blank_value ? 1 : 0;

        //
        // Calculate and return the average.
        //
        float ave = 0.;
        int n_val = 0;
        for (int l = 0; l < 8; l++)
        {
            if (var_out[offset+l] != blank_value)
            {
                n_val++;
                ave += var_out[offset+l];
            }
        }
        if (n_val)
            ave /= float(n_val);
        else
            ave = blank_value;
        return ave;
    }
}

void htg_write_file(const string &stem,
                    const double *bounds,
                    int n_levels,
                    int n_vertices,
                    int n_descriptor,
                    int descriptor_min,
                    int descriptor_max,
                    const int *descriptor,
                    int nb_vertices_by_level_max,
                    const int *nb_vertices_by_level,
                    int n_mask,
                    int mask_min,
                    int mask_max,
                    const int *mask,
                    double var_min,
                    double var_max,
                    const float *var)
{
    //
    // Write out the HTG VTK file. It is in ASCII format, which is the
    // least efficient, but it's the simplest and was great for developing
    // the algorithm. This should probably be improved at some point.
    //
    std::string filename = stem + ".htg";
    ofstream *ofile = new ofstream(filename.c_str());

    *ofile << "<VTKFile type=\"HyperTreeGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">" << endl;
    *ofile << "  <HyperTreeGrid BranchFactor=\"2\" TransposedRootIndexing=\"0\" Dimensions=\"2 2 2\">" << endl;
    *ofile << "    <Grid>" << endl;
    *ofile << "      <DataArray type=\"Float64\" Name=\"XCoordinates\" NumberOfTuples=\"2\" format=\"ascii\" RangeMin=\"" << bounds[0] << "\" RangeMax=\"" << bounds[1] << "\">" << endl;
    *ofile << "        " << bounds[0] << " " << bounds[1] << endl;
    *ofile << "      </DataArray>" << endl;
    *ofile << "      <DataArray type=\"Float64\" Name=\"YCoordinates\" NumberOfTuples=\"2\" format=\"ascii\" RangeMin=\"" << bounds[2] << "\" RangeMax=\"" << bounds[3] << "\">" << endl;
    *ofile << "        " << bounds[2] << " " << bounds[3] << endl;
    *ofile << "      </DataArray>" << endl;
    *ofile << "      <DataArray type=\"Float64\" Name=\"ZCoordinates\" NumberOfTuples=\"2\" format=\"ascii\" RangeMin=\"" << bounds[4] << "\" RangeMax=\"" << bounds[5] << "\">" << endl;
    *ofile << "        " << bounds[4] << " " << bounds[5] << endl;
    *ofile << "      </DataArray>" << endl;
    *ofile << "    </Grid>" << endl;
    *ofile << "    <Trees>" << endl;
    *ofile << "      <Tree Index=\"0\" NumberOfLevels=\"" << n_levels << "\" NumberOfVertices=\"" << n_vertices << "\">" << endl;
    *ofile << "        <DataArray type=\"Bit\" Name=\"Descriptor\" NumberOfTuples=\"" << n_descriptor << "\" format=\"ascii\" RangeMin=\"" << descriptor_min << "\" RangeMax=\"" << descriptor_max << "\">" << endl;
    for (int i = 0; i < n_descriptor; i += 6)
    {
        *ofile << "          ";
        int jmax = (i + 6 < n_descriptor) ? i + 6 : n_descriptor;
        for (int j = i; j < jmax - 1; j++)
            *ofile << descriptor[j] << " ";
        *ofile << descriptor[jmax-1] << endl;;
    }
    *ofile << "        </DataArray>" << endl;
    *ofile << "        <DataArray type=\"Int64\" Name=\"NbVerticesByLevel\" NumberOfTuples=\"" << n_levels << "\" format=\"ascii\" RangeMin=\"1\" RangeMax=\"" << nb_vertices_by_level_max << "\">" << endl;
    *ofile << "          ";
    for (int i = 0; i < n_levels - 1; i++)
        *ofile << nb_vertices_by_level[i] << " ";
    *ofile << nb_vertices_by_level[n_levels-1] << endl;;
    *ofile << "        </DataArray>" << endl;
    *ofile << "        <DataArray type=\"Bit\" Name=\"Mask\" NumberOfTuples=\"" << n_mask << "\" format=\"ascii\" RangeMin=\"" << mask_min << "\" RangeMax=\"" << mask_max << "\">" << endl;
    for (int i = 0; i < n_mask; i += 6)
    {
        *ofile << "          ";
        int jmax = (i + 6 < n_mask) ? i + 6 : n_mask;
        for (int j = i; j < jmax - 1; j++)
            *ofile << mask[j] << " ";
        *ofile << mask[jmax-1] << endl;;
    }
    *ofile << "        </DataArray>" << endl;
    *ofile << "        <CellData>" << endl;
    *ofile << "          <DataArray type=\"Float64\" Name=\"u\" NumberOfTuples=\"" << n_vertices << "\" format=\"ascii\" RangeMin=\"" << var_min << "\" RangeMax=\"" << var_max << "\">" << endl;
    for (int i = 0; i < n_vertices; i += 6)
    {
        *ofile << "          ";
        int jmax = (i + 6 < n_vertices) ? i + 6 : n_vertices;
        for (int j = i; j < jmax - 1; j++)
            *ofile << var[j] << " ";
        *ofile << var[jmax-1] << endl;;
    }
    *ofile << "          </DataArray>" << endl;
    *ofile << "        </CellData>" << endl;
    *ofile << "      </Tree>" << endl;
    *ofile << "    </Trees>" << endl;
    *ofile << "  </HyperTreeGrid>" << endl;
    *ofile << "</VTKFile>" << endl;
}

void htg_write(const std::string &path,
               float blank_value,
               int nx,
               const double *bounds,
               const float *value)
{
    //
    // Determine the number of levels.
    //
    int n_levels = 1;
    int dim = nx;
    while (dim > 1)
    {
        n_levels++;
        dim = dim / 2;
    }

    //
    // Calculate the total number of vertices.
    //
    int n_vertices = 0;
    for (int i = 0; i < n_levels; i++)
        n_vertices += 1 << i * 3;

    //
    // Calculate min and max for the variable. We only need to do
    // the input array, since the output will contain the input and
    // averages of the input. We exclude any blank values. If the
    // array contains all blank values we throw an exception.
    //
    int nvals = nx * nx * nx;

    //
    // Find the first non blank value.
    //
    int i_real;
    for (i_real = 0; i_real < nvals && value[i_real] == blank_value; i_real++)
        /* Do nothing */;

    if (i_real == nvals)
    {
        ASCENT_ERROR("htg extract: the variable only had blank values."<<endl);
    }

    float var_min = value[i_real];
    float var_max = value[i_real];
    for (int i = i_real + 1; i < nvals; i++)
    {
        if (value[i] != blank_value)
        {
            var_min = value[i] < var_min ? value[i] : var_min;
            var_max = value[i] > var_max ? value[i] : var_max;
        }
    }

    //
    // Set the number of vertices in each level.
    //
    int *nb_vertices_by_level = new int[n_levels];
    for (int i = 0; i < n_levels; i++)
        nb_vertices_by_level[i] = 1 << i * 3;

    //
    // Create the HTG, specifically the output variable and the mask.
    //
    int *mask = new int[n_vertices];
    float *var = new float[n_vertices];

    int i_level = 1;
    int *offsets = new int[n_levels];
    offsets[0] = 0;
    for (int i = 1; i < n_levels; i++)
       offsets[i] = offsets[i-1] + nb_vertices_by_level[i-1];
    int i_start = 0;
    int j_start = 0;
    int k_start = 0;
    var[0] = htg_create(value, var, mask, n_levels, nx, blank_value, i_level,
        offsets, i_start, j_start, k_start);
    mask[0] = var[0] == blank_value ? 1 : 0;
    delete [] offsets;

    //
    // Compress the output variable based on the mask variable.
    //
    int *mask2 = new int[n_vertices];
    int n_vertices2 = 9;
    int i_var = 9;
    int i_mask = 1;
    for (int i = 0; i < 9; i++)
        mask2[i] = mask[i];
    for (int i = 1; i < n_levels - 1; i++)
    {
        int n_bits = 1 << i * 3;
        int nb_vertices = 0;
        for (int j = 0; j < n_bits; j++)
        {
             if (mask[i_mask] == 0)
             {
                 nb_vertices += 8;
                 for (int k = 0; k < 8; k++)
                 {
                     mask2[n_vertices2] = mask[i_var];
                     var[n_vertices2] = var[i_var];
                     n_vertices2++;
                     i_var++;
                 }
             }
             else
             {
                 i_var += 8;
             }
             i_mask++;
        }
        nb_vertices_by_level[i+1] = nb_vertices;
    }
    n_vertices = n_vertices2;
    delete [] mask;
    mask = mask2;

    int nb_vertices_by_level_max = nb_vertices_by_level[n_levels-1];

    //
    // Determine the size of the mask variable. Remove any trailing zeros.
    //
    int last_zero = -1;
    int last_one = -1;
    for (int i = 0; i < n_vertices; i++)
    {
        if (mask[i] == 0)
            last_zero = i;
        else
            last_one = i;
    }
    int n_mask = 1;
    int mask_min = 0;
    int mask_max = 0;
    if (last_one != -1)
    {
        mask_max = 1;
        n_mask = last_one + 1;
    }

    //
    // Create the descriptor variable. It is the opposite of the mask
    // variable for all but the last level.
    //
    int n_descriptor = 0;
    for (int i = 0; i < n_levels-1; i++)
        n_descriptor += nb_vertices_by_level[i];

    int *descriptor = new int[n_descriptor];
    for (int i = 0; i < n_descriptor; i++)
        descriptor[i] = (mask[i] == 0) ? 1 : 0;

    //
    // Determine the size of the descriptor variable. Remove any trailing zeros.
    //
    last_zero = -1;
    last_one = -1;
    for (int i = 0; i < n_descriptor; i++)
    {
        if (descriptor[i] == 0)
            last_zero = i;
        else
            last_one = i;
    }
    n_descriptor = 1;
    int descriptor_min = 0;
    int descriptor_max = 0;
    if (last_zero == -1)
        descriptor_min = 1;
    if (last_one != -1)
    {
        descriptor_max = 1;
        n_descriptor = last_one + 1;
    }

    //
    // Replace any blank_value with zero.
    //
    for (int i = 0; i <n_vertices; i++)
    {
        if (var[i] == blank_value)
            var[i] = 0.;
    }

    htg_write_file(path, bounds, n_levels, n_vertices, n_descriptor,
                   descriptor_min, descriptor_max, descriptor,
                   nb_vertices_by_level_max, nb_vertices_by_level,
                   n_mask, mask_min, mask_max, mask, var_min, var_max, var);
}

void htg_save(const Node &data,
              const Node &fields,
              const std::string &path,
              float blank_value)
{
    if (data.number_of_children() != 1)
    {
        ASCENT_ERROR("htg extract requires a single domain."<<endl);
    }

    const conduit::Node &dom = data.child(0);

    //
    // Determine the fields. If the fields node is empty then use all
    // the fields.
    //
    int nfields = fields.number_of_children();
    std::vector<std::string> fnames;
    if (nfields == 0 && dom.has_path("fields"))
    {
        fnames = dom["fields"].child_names();
        nfields = fnames.size();
    }
    else
    {
        fnames = fields.child_names();
    }

    //
    // Loop over the fields.
    //
    for(int f = 0; f < nfields; ++f)
    {
        const std::string fname = fnames[f];
        if(dom.has_path("fields/" + fname))
        {
            const std::string fpath = "fields/" + fname;
            const std::string topo = dom[fpath + "/topology"].as_string();
            const std::string tpath = "topologies/" + topo;
            const std::string coords = dom[tpath + "/coordset"].as_string();
            const std::string cpath = "coordsets/" + coords;

            if(dom[fpath + "/association"].as_string() != "element")
            {
                ASCENT_INFO(fname<<": htg extract requires an element association, skipping."<<endl);
                continue;
            }
            if(dom[cpath + "/type"].as_string() != "uniform")
            {
                ASCENT_INFO(fname<<": htg extract requires a uniform mesh, skipping."<<endl);
                continue;
            }
            if (!dom.has_path(cpath + "/dims/k"))
            {
                ASCENT_INFO(fname<<": htg extract requires a 3d mesh, skipping."<<endl);
                continue;
            }

            int nx, ny, nz;
            nx = dom[cpath + "/dims/i"].to_int32();
            ny = dom[cpath + "/dims/j"].to_int32();
            nz = dom[cpath + "/dims/k"].to_int32();
            if (nx != ny && ny != nz)
            {
                ASCENT_INFO(fname<<": htg extract requires the dimensions to be equal, skipping."<<endl);
                continue;
            }
            nx = nx - 1;
            if (nx == 0 || ((nx & (nx - 1)) != 0))
            {
                ASCENT_INFO(fname<<": htg extract requires the grid dimension to be a power of 2, skipping."<<endl);
                continue;
            }

            double x0, y0, z0, dx, dy, dz;
            double bounds[6];
            x0 = dom[cpath + "/origin/x"].to_float64();
            y0 = dom[cpath + "/origin/y"].to_float64();
            z0 = dom[cpath + "/origin/z"].to_float64();
            dx = dom[cpath + "/spacing/dx"].to_float64();
            dy = dom[cpath + "/spacing/dy"].to_float64();
            dz = dom[cpath + "/spacing/dz"].to_float64();
            bounds[0] = x0;
            bounds[1] = x0 + dx * double(nx);
            bounds[2] = y0;
            bounds[3] = y0 + dy * double(nx);
            bounds[4] = z0;
            bounds[5] = z0 + dz * double(nx);
            conduit::Node res;
            if (dom[fpath + "/values"].dtype().is_float() &&
                dom[fpath + "/values"].dtype().is_compact())
            {
                res.set_external(dom[fpath + "/values"]);
            }
            else
            {
                dom[fpath + "/values"].to_float_array(res);
            }
            const float *values = res.value();

            htg_write(path, blank_value, nx, bounds, values);
        }
    }
}


//-----------------------------------------------------------------------------
HTGIOSave::HTGIOSave()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
HTGIOSave::~HTGIOSave()
{
// empty
}

//-----------------------------------------------------------------------------
void
HTGIOSave::declare_interface(Node &i)
{
    i["type_name"]   = "htg_io_save";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
HTGIOSave::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    return verify_htg_params(params,info);
}

//-----------------------------------------------------------------------------
void
HTGIOSave::execute()
{
  
#if ASCENT_MPI_ENABLED
    ASCENT_ERROR("htg extract only supports serial execution"<<endl);  
#endif
    std::string path;
    path = params()["path"].as_string();
    path = output_dir(path);

    float blank_value = params()["blank_value"].as_float32();

    if(!input("in").check_type<DataObject>())
    {
        ASCENT_ERROR("htg extract requires a DataObject input"<<endl);
    }

    DataObject *data_object  = input<DataObject>("in");
    if(!data_object->is_valid())
    {
      return;
    }
    std::shared_ptr<Node> n_input = data_object->as_node();

    Node *in = n_input.get();

    Node fields;
    if(params().has_path("fields"))
    {
      fields = params()["fields"];
    }

    htg_save(*in, fields, path, blank_value);

    // add this to the extract results in the registry
    if(!graph().workspace().registry().has_entry("extract_list"))
    {
      conduit::Node *extract_list = new conduit::Node();
      graph().workspace().registry().add<Node>("extract_list",
                                               extract_list,
                                               -1); // TODO keep forever?
    }

    conduit::Node *extract_list = graph().workspace().registry().fetch<Node>("extract_list");

    Node &einfo = extract_list->append();
    einfo["type"] = "htg";
    einfo["path"] = path;
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

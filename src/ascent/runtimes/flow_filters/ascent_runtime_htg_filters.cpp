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
#include <ascent_file_system.hpp>
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
void htg_vtk_save(const Node &data,
                  const std::string &path)
{
    int nx = data["coordsets/coords/values/x"].dtype().number_of_elements();
    int ny = data["coordsets/coords/values/y"].dtype().number_of_elements();
    int nz = data["coordsets/coords/values/z"].dtype().number_of_elements();
    const double *xcoord = data["coordsets/coords/values/x"].as_double_ptr();
    const double *ycoord = data["coordsets/coords/values/y"].as_double_ptr();
    const double *zcoord = data["coordsets/coords/values/z"].as_double_ptr();
    int nlevels = data["topologies/mesh/nlevels"].as_int();

    FILE *f = fopen(path.c_str(), "w");

    // Write the header.
    fprintf(f, "<VTKFile type=\"HyperTreeGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n");
    fprintf(f, "  <HyperTreeGrid BranchFactor=\"2\" TransposedRootIndexing=\"0\" Dimensions=\"%d %d %d\">\n", nx, ny, nz);

    // Write the grid.
    fprintf(f, "    <Grid>\n");
    fprintf(f, "      <DataArray type=\"Float64\" Name=\"XCoordinates\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%g\" RangeMax=\"%g\">\n", nx, xcoord[0], xcoord[nx-1]);
    fprintf(f, "       ");
    for (int i = 0; i < nx; ++i)
    {
        fprintf(f, " %g", xcoord[i]);
    }
    fprintf(f, "\n");
    fprintf(f, "      </DataArray>\n");

    fprintf(f, "      <DataArray type=\"Float64\" Name=\"YCoordinates\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%g\" RangeMax=\"%g\">\n", ny, ycoord[0], ycoord[ny-1]);
    fprintf(f, "       ");
    for (int i = 0; i < ny; ++i)
    {
        fprintf(f, " %g", ycoord[i]);
    }
    fprintf(f, "\n");
    fprintf(f, "      </DataArray>\n");

    fprintf(f, "      <DataArray type=\"Float64\" Name=\"ZCoordinates\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%g\" RangeMax=\"%g\">\n", nz, zcoord[0], zcoord[nz-1]);
    fprintf(f, "       ");
    for (int i = 0; i < nz; ++i)
    {
        fprintf(f, " %g", zcoord[i]);
    }
    fprintf(f, "\n");
    fprintf(f, "      </DataArray>\n");
    fprintf(f, "    </Grid>\n");

    // Write the trees.
    fprintf(f, "    <Trees>\n");

    int ntrees = data["topologies/mesh/trees"].number_of_children();
    for (int itree = 0; itree < ntrees; ++itree)
    {
        const Node &tree = data["topologies/mesh/trees"][itree];
        int nvertices = tree["nvertices"].as_int();
        int descriptor_ntuples = tree["descriptor_ntuples"].as_int();
        const int *descriptor_range = tree["descriptor_range"].as_int_ptr();
        const int *descriptor_values = tree["descriptor_values"].as_int_ptr();
        int nbverticesbylevel_ntuples = tree["nbverticesbylevel_ntuples"].as_int();
        const int *nbverticesbylevel_range = tree["nbverticesbylevel_range"].as_int_ptr();
        const int *nbverticesbylevel_values = tree["nbverticesbylevel_values"].as_int_ptr();
        int mask_ntuples = tree["mask_ntuples"].as_int();
        const int *mask_range = tree["mask_range"].as_int_ptr();
        const int *mask_values = tree["mask_values"].as_int_ptr();
        const double *var_range = tree["var_range"].as_double_ptr();
        const double *var_values = tree["var_values"].as_double_ptr();

        fprintf(f, "      <Tree Index=\"%d\" NumberOfLevels=\"%d\" NumberOfVertices=\"%d\">\n", itree, nlevels, nvertices);
        fprintf(f, "        <DataArray type=\"Bit\" Name=\"Descriptor\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%d\" RangeMax=\"%d\">\n", descriptor_ntuples, descriptor_range[0], descriptor_range[1]);
        fprintf(f, "         ");
        for (int i = 0; i < descriptor_ntuples; ++i)
        {
            fprintf(f, " %d", descriptor_values[i]);
        }
        fprintf(f, "\n");
        fprintf(f, "        </DataArray>\n");
        fprintf(f, "        <DataArray type=\"Int64\" Name=\"NbVerticesByLevel\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%d\" RangeMax=\"%d\">\n", nbverticesbylevel_ntuples, nbverticesbylevel_range[0], nbverticesbylevel_range[1]);
        fprintf(f, "         ");
        for (int i = 0; i < nbverticesbylevel_ntuples; ++i)
        {
            fprintf(f, " %d", nbverticesbylevel_values[i]);
        }
        fprintf(f, "\n");
        fprintf(f, "        </DataArray>\n");
        fprintf(f, "        <DataArray type=\"Bit\" Name=\"Mask\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%d\" RangeMax=\"%d\">\n", mask_ntuples, mask_range[0], mask_range[1]);
        fprintf(f, "         ");
        for (int i = 0; i < mask_ntuples; ++i)
        {
            fprintf(f, " %d", mask_values[i]);
        }
        fprintf(f, "\n");
        fprintf(f, "        </DataArray>\n");
        fprintf(f, "        <CellData>\n");
        fprintf(f, "          <DataArray type=\"Float64\" Name=\"u_measure\" NumberOfTuples=\"%d\" format=\"ascii\" RangeMin=\"%.16f\" RangeMax=\"%.16f\">\n", nvertices, var_range[0], var_range[1]);
        for (int i = 0; i < nvertices; i+=6)
        {
            fprintf(f, "           ");
            int jmax = i + 6 < nvertices ? i + 6 : nvertices;
            for (int j = i; j < jmax; ++j)
            {
                fprintf(f, " %.16f", var_values[j]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "          </DataArray>\n");
        fprintf(f, "        </CellData>\n");
        fprintf(f, "      </Tree>\n");
    }

    fprintf(f, "    </Trees>\n");

    //Write the trailer.
    fprintf(f, "  </HyperTreeGrid>\n");
    fprintf(f, "</VTKFile>\n");
    fclose(f);
}

//-----------------------------------------------------------------------------
HTGSave::HTGSave()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
HTGSave::~HTGSave()
{
// empty
}

//-----------------------------------------------------------------------------
void
HTGSave::declare_interface(Node &i)
{
    i["type_name"]   = "htg_save";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
HTGSave::verify_params(const conduit::Node &params,
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

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("path");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}


//-----------------------------------------------------------------------------
void
HTGSave::execute()
{
    std::string path;
    path = params()["path"].as_string();
    path = output_dir(path);

    if(!input("in").check_type<DataObject>())
    {
        // error
        ASCENT_ERROR("htg_save requires a DataObject input");
    }

    DataObject *data_object  = input<DataObject>("in");
    if(!data_object->is_valid())
    {
      return;
    }
    std::shared_ptr<Node> n_input = data_object->as_node();

    Node *in = n_input.get();

    // We are only saving the first domain.
    htg_vtk_save((*in)[0], path);

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






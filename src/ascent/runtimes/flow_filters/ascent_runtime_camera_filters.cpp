//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory //
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
/// file: ascent_runtime_camera_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_camera_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>
#include <ascent_data_object.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/filters/Log.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/filters/Slice.hpp>
#include <vtkh/filters/Statistics.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>
#include <vtkh/filters/Histogram.hpp>
#include <vtkh/filters/HistSampling.hpp>
#include <vtkm/cont/DataSet.h>

#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <stdio.h>

using namespace conduit;
using namespace std;

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
// -- begin ascent::runtime::filters::detail --
//-----------------------------------------------------------------------------

AutoCamera::AutoCamera()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
AutoCamera::~AutoCamera()
{
// empty
}

//-----------------------------------------------------------------------------
void
AutoCamera::declare_interface(Node &i)
{
    i["type_name"]   = "auto_camera";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
AutoCamera::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    bool has_values = check_numeric("iso_values",params, info, false);
    bool has_levels = check_numeric("levels",params, info, false);

    if(!has_values && !has_levels)
    {
        info["errors"].append() = "Missing required numeric parameter. Contour must"
                                  " specify 'iso_values' or 'levels'.";
        res = false;
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("levels");
    valid_paths.push_back("iso_values");
    valid_paths.push_back("use_contour_tree");
    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
AutoCamera::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_vector_magnitude input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      ASCENT_ERROR("Unknown field '"<<field_name<<"'");
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkh::MarchingCubes marcher;

    marcher.SetInput(&data);
    marcher.SetField(field_name);

    if(params().has_path("iso_values"))
    {
      const Node &n_iso_vals = params()["iso_values"];

      // convert to contig doubles
      Node n_iso_vals_dbls;
      n_iso_vals.to_float64_array(n_iso_vals_dbls);

      marcher.SetIsoValues(n_iso_vals_dbls.as_double_ptr(),
                           n_iso_vals_dbls.dtype().number_of_elements());
    }
    else
    {
      marcher.SetLevels(params()["levels"].to_int32());
      if(params().has_path("use_contour_tree"))
      {
        std::string use = params()["use_contour_tree"].as_string();
        if(use == "true")
        {
          ;//marcher.SetUseContourTree(true);
        }
      }
    }

    marcher.Update();

    vtkh::DataSet *iso_output = marcher.GetOutput();
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*iso_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete iso_output;
    set_output<DataObject>(res);
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

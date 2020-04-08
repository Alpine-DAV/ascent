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
/// file: ascent_runtime_blueprint_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_blueprint_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------


#include <cstring>

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <runtimes/ascent_data_object.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/DataSet.h>
#include <ascent_vtkh_data_adapter.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/DataSet.hpp>
#endif

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
BlueprintVerify::BlueprintVerify()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BlueprintVerify::~BlueprintVerify()
{
// empty
}

//-----------------------------------------------------------------------------
void
BlueprintVerify::declare_interface(Node &i)
{
    i["type_name"]   = "blueprint_verify";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BlueprintVerify::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("protocol") ||
       ! params["protocol"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'protocol'";
    }

    return res;
}


//-----------------------------------------------------------------------------
void
BlueprintVerify::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_verify input must be a DataObject");
    }

    std::string protocol = params()["protocol"].as_string();

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    // some MPI tasks may not have data, that is fine
    // but blueprint verify will fail, so if the
    // input node is empty skip verify
    int local_verify_ok = 0;
    if(!n_input->dtype().is_empty())
    {
        if(!conduit::blueprint::verify(protocol,
                                       *n_input,
                                       v_info))
        {
            n_input->schema().print();
            v_info.print();
            ASCENT_ERROR("blueprint verify failed for protocol"
                          << protocol << std::endl
                          << "details:" << std::endl
                          << v_info.to_json());
        }
        else
        {
            local_verify_ok = 1;
        }
    }

    // make sure some MPI task actually had bp data
#ifdef ASCENT_MPI_ENABLED
    int global_verify_ok = 0;
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    MPI_Allreduce((void *)(&local_verify_ok),
                (void *)(&global_verify_ok),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);
    local_verify_ok = global_verify_ok;
#endif

    if(local_verify_ok == 0)
    {
        ASCENT_ERROR("blueprint verify failed: published data is empty");
    }


    set_output<DataObject>(d_input);
}

//-----------------------------------------------------------------------------
BlueprintAMRMask::BlueprintAMRMask()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BlueprintAMRMask::~BlueprintAMRMask()
{
// empty
}

//-----------------------------------------------------------------------------
void
BlueprintAMRMask::declare_interface(Node &i)
{
    i["type_name"]   = "blueprint_amr_mask";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

void calc_mask(const conduit::Node &nest,
               const conduit::Node &topo,
               int dims[3],
               conduit::Node &mask)
{
  std::cout<<"************************\n";
  std::cout<<"dims "<<dims[0]<<" "<<dims[1]<<"  "<<dims[2]<<"\n";
  nest.print();
  topo.print();

  int size = dims[0];
  if(dims[1] != 0)
  {
    size *= dims[1];
  }
  if(dims[2] != 0)
  {
    size *= dims[2];
  }

  mask.set(DataType::int32(size));
  int *mask_ptr = mask.value();

  std::memset(mask_ptr, 0, sizeof(int) * size);

  if(nest.has_path("windows"))
  {
    const int num_wins = nest["windows"].number_of_children();
    for(int i = 0; i < num_wins; ++i)
    {
      const conduit::Node &win = nest["windows"].child(i);
      if(win["domain_type"].as_string() == "parent")
      {
        // we care about zones covered by children
        continue;
      }

      int cdims[3] = {0,0,0};
      int corig[3] = {0,0,0};

      cdims[0] = win["dims/i"].to_int32();
      corig[0] = win["origin/i"].to_int32();

      if(win["origin"].has_path("j"))
      {
        cdims[1] = win["dims/j"].to_int32();
        corig[1] = win["origin/j"].to_int32();
      }
      if(win["origin"].has_path("k"))
      {
        cdims[2] = win["dims/k"].to_int32();
        corig[2] = win["origin/k"].to_int32();
      }

      bool is_2d = dims[2] == 0;
      int overlap_start[3] = {0,0,0};
      int overlap_end[3] = {0,0,0};
      int origin[3] = {0,0,0}; // TODO: get origin

      for(int d = 0; d < 3; ++d)
      {
        overlap_start[d] = std::max(origin[d], corig[d]);
        overlap_end[d] = std::min(origin[d] + dims[d], corig[d] + cdims[d]) - 1;
        std::cout<<"overlap ("<<d<<") "<<overlap_start[d]<<" - "<<overlap_end[d]<<"\n";
      }

      //if(is_2d)
      //{
      //  const int y_size = overlap_end[1] - overlap_start[1] + 1;
      //  for(int y = 0; y <
      //}

    }
  }


}

void topology_dims(const conduit::Node &dom,
                   const std::string topo_name,
                   int dims[3])
{
  const conduit::Node &topo = dom["topologies/"+topo_name];
  const std::string coord_name = topo["coordset"].as_string();
  const conduit::Node &coords = dom["coordsets/"+coord_name];

  const std::string topo_type = topo["type"].as_string();
  dims[0] = 0;
  dims[1] = 0;
  dims[2] = 0;

  if(topo_type == "structured")
  {
    dims[0] = topo["elements/i"].to_int32();
    if(topo["elements"].has_path("j"))
    {
      dims[1] = topo["elements/j"].to_int32();
    }
    if(topo["elements"].has_path("k"))
    {
      dims[2] = topo["elements/k"].to_int32();
    }
  }
  else if(topo_type == "uniform")
  {
    dims[0] = coords["dims/i"].to_int32() - 1;
    if(coords["dims"].has_path("j"))
    {
      dims[1] = coords["dims/j"].to_int32() - 1;
    }
    if(coords["dims"].has_path("k"))
    {
      dims[2] = coords["dims/k"].to_int32() - 1;
    }
  }
  else if(topo_type == "rectilinear")
  {
    dims[0] = coords["values/x"].dtype().number_of_elements() - 1;
    if(coords["values"].has_path("y"))
    {
      dims[1] = coords["values/y"].dtype().number_of_elements() - 1;
    }
    if(coords["values"].has_path("z"))
    {
      dims[2] = coords["values/z"].dtype().number_of_elements() - 1;
    }
  }
  else
  {
    ASCENT_ERROR("NO "<<topo_type<<"\n");
  }

}

//-----------------------------------------------------------------------------
void
BlueprintAMRMask::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_amr_mask input must be a DataObject");
    }

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    const int num_doms = n_input->number_of_children();

    for(int i = 0; i < num_doms; ++i)
    {
      conduit::Node &dom = n_input->child(i);
      bool has_amr = dom.has_child("nestsets");
      if(has_amr)
      {
        const int num_nests = dom["nestsets"].number_of_children();
        for (int n = 0; n < num_nests; ++n)
        {
          const conduit::Node &nest = dom["nestsets"].child(n);
          if(!nest.has_path("topology"))
          {
            ASCENT_ERROR("verify should have caught this");
          }
          const std::string topo_name = nest["topology"].as_string();
          const conduit::Node &topo = dom["topologies/" + topo_name];
          int dims[3];
          topology_dims(dom, topo_name, dims);
          conduit::Node mask; //TODO get this  from dom
          calc_mask(nest, topo, dims, mask);
        }

      }

    }

    set_output<DataObject>(d_input);
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






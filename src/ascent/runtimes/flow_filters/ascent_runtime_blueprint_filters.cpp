//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/DataSet.h>
#include <ascent_vtkh_data_adapter.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
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

    if(!input(0).check_type<Node>())
    {
        ASCENT_ERROR("blueprint_verify input must be a conduit node");
    }


    std::string protocol = params()["protocol"].as_string();

    Node v_info;
    Node *n_input = input<Node>(0);
    if(!conduit::blueprint::verify(protocol,
                                   *n_input,
                                   v_info))
    {
        n_input->print();
        v_info.print();
        ASCENT_ERROR("blueprint verify failed for protocol"
                      << protocol << std::endl
                      << "details:" << std::endl
                      << v_info["message"].as_string());
                      //<< v_info.to_json());
    }
    
    set_output<Node>(n_input);
}


//-----------------------------------------------------------------------------
EnsureLowOrder::EnsureLowOrder()
:Filter(),
 m_refinement_level(2)
{
// empty
}

//-----------------------------------------------------------------------------
EnsureLowOrder::~EnsureLowOrder()
{
// empty
}

//-----------------------------------------------------------------------------
void 
EnsureLowOrder::declare_interface(Node &i)

{
    i["type_name"]   = "ensure_low_order";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

bool 
EnsureLowOrder::is_high_order(const conduit::Node &doms)
{
  // treat everything as a multi-domain data set 
  const int num_domains = doms.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = doms.child(i);      
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      const int num_fields= fields.number_of_children();
      for(int t = 0; t < num_fields; ++t)
      {
        const conduit::Node &field = fields.child(t);      
        if(field.has_path("basis")) return true;
      }
      
    }
  }

  return false;
}
//-----------------------------------------------------------------------------
bool
EnsureLowOrder::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();   
    bool res = true;
    
    return res;
}


//-----------------------------------------------------------------------------
void 
EnsureLowOrder::execute()
{

    if(!input(0).check_type<Node>())
    {
        ASCENT_ERROR("ensure_low order input must be a conduit node");
    }

    Node *n_input = input<Node>(0);

    if(is_high_order(*n_input))
    {
#if defined(ASCENT_MFEM_ENABLED)
      MFEMDomains *domains = MFEMDataAdapter::BlueprintToMFEMDataSet(*n_input);
      conduit::Node *lo_dset = new conduit::Node;
      MFEMDataAdapter::Linearize(domains, *lo_dset, m_refinement_level);
      //lo_dset->print();
      delete domains;
      set_output<Node>(lo_dset);

      // add a second registry entry for the output so it can be zero copied.
      const std::string key = "low_mesh_key";
      graph().workspace().registry().add(key, lo_dset, 1);
#else
      ASCENT_ERROR("Unable to convert high order mesh when MFEM is not enabled");
#endif
    }
    else
    {
      // already low order just pass it through
      set_output<Node>(n_input);
    }

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






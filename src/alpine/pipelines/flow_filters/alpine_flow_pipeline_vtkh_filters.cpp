//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: alpine_flow_pipeline_vtkh_filters.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow_pipeline_vtkh_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// alpine includes
//-----------------------------------------------------------------------------
#include <alpine_logging.hpp>
#include <alpine_flow_graph.hpp>
#include <alpine_flow_workspace.hpp>

#if defined(ALPINE_VTKM_ENABLED)
#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include <vtkm/cont/DataSet.h>

#endif

using namespace conduit;
using namespace std;

using namespace alpine::flow;

//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

//-----------------------------------------------------------------------------
// -- begin alpine::pipeline --
//-----------------------------------------------------------------------------
namespace pipeline
{

//-----------------------------------------------------------------------------
// -- begin alpine::pipeline::flow --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// -- begin alpine::pipeline::flow::filters --
//-----------------------------------------------------------------------------
namespace filters
{


//-----------------------------------------------------------------------------
EnsureVTKH::EnsureVTKH()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
EnsureVTKH::~EnsureVTKH()
{
// empty
}

//-----------------------------------------------------------------------------
void 
EnsureVTKH::declare_interface(Node &i)
{
    i["type_name"]   = "ensure_vtkh";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void 
EnsureVTKH::execute()
{

    
    if(input(0).check_type<vtkm::cont::DataSet>())
    {
        vtkm::cont::DataSet *vtkm_data = input<vtkm::cont::DataSet>(0);
        vtkh::vtkhDataSet   *res = new  vtkh::vtkhDataSet;
        // should be MPI_TASK
        res->AddDomain(*vtkm_data,0);
        
        set_output<vtkh::vtkhDataSet>(res);
    }
    else if(input(0).check_type<vtkh::vtkhDataSet>())
    {
        // ok
        set_output(input(0));
    }
    else
    {
        ALPINE_ERROR("ensure_vtkh input must be a vtk-m or vtk-h dataset");
    }
}


//-----------------------------------------------------------------------------
VTKHRayTracer::VTKHRayTracer()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHRayTracer::~VTKHRayTracer()
{
// empty
}

//-----------------------------------------------------------------------------
void 
VTKHRayTracer::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_raytracer";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
VTKHRayTracer::verify_params(const conduit::Node &params,
                                   conduit::Node &info)
{
    info.reset();   
    bool res = true;
    // TODO
    return res;
}


//-----------------------------------------------------------------------------
void 
VTKHRayTracer::execute()
{
    if(!input(0).check_type<vtkh::vtkhDataSet>())
    {
        ALPINE_ERROR("vtkh_raytracer input must be a vtk-h dataset");
    }
 
    ALPINE_INFO("Doing the render!");
    
    vtkh::vtkhDataSet *data = input<vtkh::vtkhDataSet>(0);
    vtkh::vtkhRayTracer ray_tracer;  
    ray_tracer.SetInput(data);
    ray_tracer.SetField("braid"); 
    ray_tracer.Update();
    
}


//-----------------------------------------------------------------------------
VTKHMarchingCubes::VTKHMarchingCubes()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHMarchingCubes::~VTKHMarchingCubes()
{
// empty
}

//-----------------------------------------------------------------------------
void 
VTKHMarchingCubes::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_marchingcubes";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHMarchingCubes::verify_params(const conduit::Node &params,
                                   conduit::Node &info)
{
    info.reset();   
    bool res = true;
    // TODO
    return res;
}


//-----------------------------------------------------------------------------
void 
VTKHMarchingCubes::execute()
{

    ALPINE_INFO("Marching the cubes!");
    
    if(!input(0).check_type<vtkh::vtkhDataSet>())
    {
        ALPINE_ERROR("vtkh_marchingcubes input must be a vtk-h dataset");
    }
    
    // TODO!
    set_output(input(0));
}




//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::pipeline::flow::filters --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::pipeline::flow --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::pipeline --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------






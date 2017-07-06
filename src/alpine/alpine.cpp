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
/// file: alpine.cpp
///
//-----------------------------------------------------------------------------

#include <alpine.hpp>
#include <alpine_pipeline.hpp>

#include <pipelines/alpine_empty_pipeline.hpp>
#include <pipelines/alpine_flow_pipeline.hpp>

#if defined(ALPINE_VTKM_ENABLED)
    #include <pipelines/alpine_vtkm_pipeline.hpp>
#endif

#if defined(ALPINE_HDF5_ENABLED)
    #include <pipelines/alpine_blueprint_hdf5_pipeline.hpp>
#endif

using namespace conduit;
//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Alpine::Alpine()
: m_pipeline(NULL)
{
}

//-----------------------------------------------------------------------------
Alpine::~Alpine()
{

}

//-----------------------------------------------------------------------------
void
Alpine::Open()
{
    Node opts;
    Open(opts);
}

//-----------------------------------------------------------------------------
void
CheckForJSONFile(std::string file_name, conduit::Node &node)
{
    if(!conduit::utils::is_file(file_name))
    {
        return;
    }
    
    conduit::Node file_node; 
    file_node.load(file_name, "json");
    node.update(file_node);
}

//-----------------------------------------------------------------------------
void
Alpine::Open(const conduit::Node &options)
{
    Node processed_opts(options);
    CheckForJSONFile("alpine_options.json", processed_opts); 
    if(m_pipeline != NULL)
    {
        ALPINE_ERROR("Alpine Pipeline already exists.!");
    }
    
    Node cfg;
    alpine::about(cfg);
    
    std::string pipeline_type = cfg["default_pipeline"].as_string();
    
    if(processed_opts.has_path("pipeline"))
    {
        if(processed_opts.has_path("pipeline/type"))
        {
            pipeline_type = processed_opts["pipeline/type"].as_string();
        }
    }

    if(pipeline_type == "empty")
    {
        m_pipeline = new EmptyPipeline();
    }
    else if(pipeline_type == "vtkm")
    {
#if defined(ALPINE_VTKM_ENABLED)
        m_pipeline = new VTKMPipeline();
#else
        ALPINE_ERROR("Alpine was not built with VTKm support");
#endif
    }
    else if(pipeline_type == "blueprint_hdf5")
    {
    #if defined(ALPINE_HDF5_ENABLED)
        m_pipeline = new BlueprintHDF5Pipeline();
    #else
        ALPINE_ERROR("Alpine was not built with HDF5 support");
    #endif
    }
    else if(pipeline_type == "flow")
    {
        m_pipeline = new FlowPipeline();
    }
    else
    {
        ALPINE_ERROR("Unsupported Pipeline type " 
                       << "\"" << pipeline_type << "\""
                       << " passed via 'pipeline' open option.");
    }
    
    m_pipeline->Initialize(processed_opts);
}

//-----------------------------------------------------------------------------
void
Alpine::Publish(const conduit::Node &data)
{
    m_pipeline->Publish(data);
}

//-----------------------------------------------------------------------------
void
Alpine::Execute(const conduit::Node &actions)
{
    Node processed_actions(actions);
    CheckForJSONFile("alpine_actions.json", processed_actions);
    m_pipeline->Execute(processed_actions);
}

//-----------------------------------------------------------------------------
void
Alpine::Close()
{
    if(m_pipeline != NULL)
    {
        m_pipeline->Cleanup();
        delete m_pipeline;
        m_pipeline = NULL;
    }
}

//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    alpine::about(n);
    return n.to_json();
}

//---------------------------------------------------------------------------//
void
about(conduit::Node &n)
{
    n.reset();
    n["version"] = "0.1.0";

#if defined(ALPINE_VTKM_ENABLED)
    n["pipelines/vtkm/status"] = "enabled";
    
    n["pipelines/vtkm/backends/serial"] = "enabled";
    
#ifdef ALPINE_VTKM_USE_TBB
    n["pipelines/vtkm/backends/tbb"] = "enabled";
#else
    n["pipelines/vtkm/backends/tbb"] = "disabled";
#endif

#ifdef ALPINE_VTKM_USE_CUDA
    n["pipelines/vtkm/backends/cuda"] = "enabled";
#else
    n["pipelines/vtkm/backends/cuda"] = "disabled";
#endif    
    
#else
    n["pipelines/vtkm/status"] = "disabled";
#endif

#if defined(ALPINE_HDF5_ENABLED)
    n["pipelines/blueprint_hdf5/status"] = "enabled";
#else
    n["pipelines/blueprint_hdf5/status"] = "disabled";
#endif

//
// Select default pipeline based on what is available.
//
#if defined(ALPINE_VTKM_ENABLED)
    n["default_pipeline"] = "vtkm";
#elif defined(ALPINE_HDF5_ENABLED)    
    n["default_pipeline"] = "blueprint_hdf5";
#else
    n["default_pipeline"] = "empty";    
#endif

}


//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------



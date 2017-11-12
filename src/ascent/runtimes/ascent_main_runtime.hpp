//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
/// file: ascent_ascent_runtime.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_ASCENT_RUNTIME_HPP
#define ASCENT_ASCENT_RUNTIME_HPP

#include <ascent.hpp>
#include <ascent_runtime.hpp>
#include <ascent_web_interface.hpp>
#include <flow.hpp>



//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class AscentRuntime : public Runtime
{
public:
    
    // Creation and Destruction
    AscentRuntime();
    virtual ~AscentRuntime();

    // Main runtime interface methods used by the ascent interface.
    void  Initialize(const conduit::Node &options);

    void  Publish(const conduit::Node &data);
    void  Execute(const conduit::Node &actions);
    
    void  Info(conduit::Node &out);
    
    void  Cleanup();

private:
    // holds options passed to initialize
    conduit::Node     m_runtime_options;
    // conduit node that (externally) holds the data from the simulation
    conduit::Node     m_data; 
    conduit::Node     m_connections; 
    conduit::Node     m_scene_connections;
    
    conduit::Node     m_info;

    WebInterface      m_web_interface;
    
    void              ResetInfo();

    flow::Workspace w;
    std::string CreateDefaultFilters();
    void ConvertToFlowGraph(const conduit::Node &pipeline,
                            const std::string pipeline_name);
    void ConvertPlotToFlow(const conduit::Node &plot,
                           const std::string plot_name,
                           bool composite);
    void ConvertExtractToFlow(const conduit::Node &plot,
                              const std::string extract_name);
    void CreatePipelines(const conduit::Node &pipelines);
    void CreateExtracts(const conduit::Node &extracts);
    void CreatePlots(const conduit::Node &plots);
    std::vector<std::string> GetPipelines(const conduit::Node &plots);
    void CreateScenes(const conduit::Node &scenes);
    void ConvertSceneToFlow(const conduit::Node &scenes);
    void ConnectGraphs();
    void ExecuteGraphs();
    std::string GetDefaultImagePrefix(const std::string scene);
    
    void FindRenders(const conduit::Node &info, conduit::Node &out);
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------



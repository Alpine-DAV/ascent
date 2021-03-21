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
/// file: ascent_ascent_runtime.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_ASCENT_RUNTIME_HPP
#define ASCENT_ASCENT_RUNTIME_HPP

#include <ascent.hpp>
#include <ascent_exports.h>
#include <ascent_runtime.hpp>
#include <ascent_data_object.hpp>
#include <ascent_web_interface.hpp>
#include <flow.hpp>



//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class ASCENT_API AscentRuntime : public Runtime
{
public:

    // Creation and Destruction
    AscentRuntime();
    virtual ~AscentRuntime();

    // Main runtime interface methods used by the ascent interface.
    void  Initialize(const conduit::Node &options) override;

    void  Publish(const conduit::Node &data) override;
    void  Execute(const conduit::Node &actions) override;

    void  Info(conduit::Node &out) override;

    void  Cleanup() override;

    void DisplayError(const std::string &msg) override;

    template <class FilterType>
    static void register_filter_type(const std::string &role_path = "",
                                     const std::string &api_name  = "")
    {
        flow::Workspace::register_filter_type<FilterType>();
        std::string filter_type_name = flow::Workspace::filter_type_name<FilterType>();
        RegisterFilterType(role_path, api_name, filter_type_name);
    }

private:
    // holds options passed to initialize
    conduit::Node     m_runtime_options;
    // DataObject that (externally) holds the data from the simulation
    conduit::Node     m_source;
    DataObject        m_data_object;
    conduit::Node     m_connections;
    conduit::Node     m_scene_connections;

    conduit::Node     m_info;
    conduit::Node     m_previous_actions;

    WebInterface      m_web_interface;
    int               m_refinement_level;
    int               m_rank;
    conduit::Node     m_ghost_fields; // a list of strings
    std::string       m_default_output_dir;
    std::string       m_session_name;

    bool              m_field_filtering;
    std::set<std::string> m_field_list;

    void              ResetInfo();

    flow::Workspace w;
    conduit::Node CreateDefaultFilters();
    void ConvertPipelineToFlow(const conduit::Node &pipeline,
                               const std::string pipeline_name);
    void ConvertPlotToFlow(const conduit::Node &plot,
                           const std::string plot_name);
    void ConvertExtractToFlow(const conduit::Node &extract,
                              const std::string extract_name);
    void ConvertTriggerToFlow(const conduit::Node &trigger,
                              const std::string trigger_name);
    void ConvertQueryToFlow(const conduit::Node &trigger,
                            const std::string trigger_name,
                            const std::string prev_name);
    void CreatePipelines(const conduit::Node &pipelines);
    void CreateExtracts(const conduit::Node &extracts);
    void CreateTriggers(const conduit::Node &triggers);
    void CreateQueries(const conduit::Node &queries);
    void CreatePlots(const conduit::Node &plots);
    std::vector<std::string> GetPipelines(const conduit::Node &plots);
    void CreateScenes(const conduit::Node &scenes);
    void ConvertSceneToFlow(const conduit::Node &scenes);
    void ConnectSource();
    void ConnectGraphs();
    void SourceFieldFilter();
    void PaintNestsets();
    void VerifyGhosts();

    void BuildGraph(const conduit::Node &actions);
    void EnsureDomainIds();
    void PopulateMetadata();

    std::string GetDefaultImagePrefix(const std::string scene);

    void FindRenders(conduit::Node &image_params, conduit::Node &image_list);

    // internal reg helper
    static void RegisterFilterType(const std::string &role_path,
                                   const std::string &api_name,
                                   const std::string &filter_type_name);

    // internal reg filter tracking
    // use const method for access, to avoid adding to the tree
    static const conduit::Node &registered_filter_types()
                                    {return s_reged_filter_types;}

    static conduit::Node s_reged_filter_types;

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



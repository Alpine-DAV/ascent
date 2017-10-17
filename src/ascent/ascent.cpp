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
/// file: ascent.cpp
///
//-----------------------------------------------------------------------------

#include <ascent.hpp>
#include <ascent_runtime.hpp>

#include <ascent_empty_runtime.hpp>
#include <ascent_flow_runtime.hpp>

#if defined(ASCENT_VTKH_ENABLED)
    #include <runtimes/ascent_main_runtime.hpp>
    #include <vtkh/vtkh.hpp>
#endif

using namespace conduit;
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
void  
quiet_handler(const std::string &,
              const std::string &,
              int )
{
}

//-----------------------------------------------------------------------------
Ascent::Ascent()
: m_runtime(NULL)
{
}

//-----------------------------------------------------------------------------
Ascent::~Ascent()
{

}

//-----------------------------------------------------------------------------
void
Ascent::open()
{
    Node opts;
    open(opts);
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
Ascent::open(const conduit::Node &options)
{
    Node processed_opts(options);
    CheckForJSONFile("ascent_options.json", processed_opts); 
    if(m_runtime != NULL)
    {
        ASCENT_ERROR("Ascent Runtime already exists.!");
    }

    bool quiet_output = true; 
    if(options.has_path("ascent_info"))
    {
        if(options["ascent_info"].as_string() == "verbose")
        {
            quiet_output = false;
        }
    }

    if(quiet_output)
    {
        conduit::utils::set_info_handler(quiet_handler);
    }

    Node cfg;
    ascent::about(cfg);
    
    std::string runtime_type = cfg["default_runtime"].as_string();
    
    if(processed_opts.has_path("runtime"))
    {
        if(processed_opts.has_path("runtime/type"))
        {
            runtime_type = processed_opts["runtime/type"].as_string();
        }
    }

    ASCENT_INFO("Runtime Type = " << runtime_type);

    if(runtime_type == "empty")
    {
        m_runtime = new EmptyRuntime();
    }
    else if(runtime_type == "ascent")
    {
#if defined(ASCENT_VTKH_ENABLED)
        m_runtime = new AscentRuntime();
        if(processed_opts.has_path("runtime/backend"))
        {
          std::string backend = processed_opts["runtime/backend"].as_string();
          if(backend == "serial")
          {
            vtkh::ForceSerial();
          }
          else if(backend == "tbb")
          {
            vtkh::ForceTBB();
          }
          else if(backend == "cuda")
          {
            vtkh::ForceCUDA();
          }
          else
          {
            ASCENT_ERROR("Ascent unrecognized backend "<<backend);
          }
        }
#else
        ASCENT_ERROR("Ascent runtime is disabled. "
                     "Ascent was not built with vtk-h support");
#endif
    }
    else if(runtime_type == "flow")
    {
        m_runtime = new FlowRuntime();
    }
    else
    {
        ASCENT_ERROR("Unsupported Runtime type " 
                       << "\"" << m_runtime << "\""
                       << " passed via 'runtime' open option.");
    }
     
    m_runtime->Initialize(processed_opts);
}

//-----------------------------------------------------------------------------
void
Ascent::publish(const conduit::Node &data)
{
    m_runtime->Publish(data);
}

//-----------------------------------------------------------------------------
void
Ascent::execute(const conduit::Node &actions)
{
    Node processed_actions(actions);
    CheckForJSONFile("ascent_actions.json", processed_actions);
    m_runtime->Execute(processed_actions);
}

//-----------------------------------------------------------------------------
void
Ascent::close()
{
    if(m_runtime != NULL)
    {
        m_runtime->Cleanup();
        delete m_runtime;
        m_runtime = NULL;
    }
}

//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    ascent::about(n);

    
    std::string ASCENT_MASCOT = "\n"
    "                                       \n"
    "         &&&&&&&&&&&                   \n"
    "       &&%&&@@@@@&&&%&&                \n"
    "      &%%&@       #@&%%&@              \n"
    "      %%%@           @&%%%%&           \n"
    "      %%%,            &%%%%%%%&        \n"
    "      &%#%*          &%##%######%%%%%  \n"
    "       @%###%&       %####%&%%%#####   \n"
    "          %###############&    @%##    \n"
    "                %%@*%((((##      &     \n"
    "                   ,#(((((#            \n"
    "                   @(////(%            \n"
    "                   &(/////#            \n"
    "                   &(/***/#            \n"
    "       #&&%%%#####%&(****/(            \n"
    "   %(////***********(*****(            \n"
    "  (********,,,,,,,**(*,,,*(            \n"
    " #**,,,*************%*,,,*(            \n"
    " (*,,,*/(((((((((#% %*,,,*%            \n"
    " /,.,*#              /,.,*             \n"
    " (,..,(             ,(,..,             \n"
    " %,..,(              (,..,             \n"
    "  ,..,/              #,..,             \n"
    "  *..,(              %...,             \n"
    "  /..,(              ..  ,             \n"
    "  @. ,#               .  .             \n"
    "  (..,#               .  .             \n"
    "\n\n"
    "Ascent Mascot ASCII Art is licensed under the: \n"
    " Creative Commons - Attribution - Share Alike license.\n"
    "  https://creativecommons.org/licenses/by-sa/3.0/\n"
    "\n"
    " Derived from:\n"
    "  https://www.thingiverse.com/thing:5340\n";
    
    return n.to_json() + "\n" + ASCENT_MASCOT;
    
}

//---------------------------------------------------------------------------//
void
about(conduit::Node &n)
{
    n.reset();
    n["version"] = "0.Z.0";

#if defined(ASCENT_VTKH_ENABLED)
     n["runtimes/ascent/status"] = "enabled";
     if(vtkh::IsSerialEnabled())
     {
       n["runtimes/ascent/backends/serial"] = "enabled";
     }
     else
     {
       n["runtimes/ascent/backends/serial"] = "disabled";
     }

     if(vtkh::IsTBBEnabled())
     {
       n["runtimes/ascent/backends/tbb"] = "enabled";
     }
     else
     {
       n["runtimes/ascent/backends/tbb"] = "disabled";
     }

     if(vtkh::IsCUDAEnabled())
     {
       n["runtimes/ascent/backends/cuda"] = "enabled";
     }
     else
     {
       n["runtimes/ascent/backends/cuda"] = "disabled";
     }
#else
     n["runtimes/ascent/status"] = "disabled";
#endif

    n["runtimes/flow/status"] = "enabled";

#if defined(ASCENT_VTKM_ENABLED)
    n["runtimes/vtkm/status"] = "enabled";
    
    n["runtimes/vtkm/backends/serial"] = "enabled";
    
#ifdef ASCENT_VTKM_USE_TBB
    n["runtimes/vtkm/backends/tbb"] = "enabled";
#else
    n["runtimes/vtkm/backends/tbb"] = "disabled";
#endif

#ifdef ASCENT_VTKM_USE_CUDA
    n["runtimes/vtkm/backends/cuda"] = "enabled";
#else
    n["runtimes/vtkm/backends/cuda"] = "disabled";
#endif    
    
#else
    n["runtimes/vtkm/status"] = "disabled";
#endif

//
// Select default runtime based on what is available.
//
#if defined(ASCENT_VTKH_ENABLED)
    n["default_runtime"] = "ascent";
#else
    n["default_runtime"] = "flow";
#endif
}


//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------



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
/// file: ascent.cpp
///
//-----------------------------------------------------------------------------

#include <ascent.hpp>
#include <ascent_license.hpp>
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
: m_runtime(NULL),
  m_verbose_msgs(false),
  m_forward_exceptions(false)
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
    try
    {
        if(m_runtime != NULL)
        {
            ASCENT_ERROR("Ascent Runtime already exists!");
        }

        Node processed_opts(options);
        CheckForJSONFile("ascent_options.json", processed_opts); 

        if(options.has_path("messages") && 
           options["messages"].dtype().is_string() )
        {
            std::string msgs_opt = options["messages"].as_string();
            if( msgs_opt == "verbose")
            {
                m_verbose_msgs = true;
            }
            else if(msgs_opt == "quiet")
            {
                m_verbose_msgs = false;
            }
        }

        if(options.has_path("exceptions") && 
           options["exceptions"].dtype().is_string() )
        {
            std::string excp_opt = options["exceptions"].as_string();
            if( excp_opt == "catch")
            {
                m_forward_exceptions = false;
            }
            else if(excp_opt == "forward")
            {
                m_forward_exceptions = true;
            }
        }
        
        // don't print info messages unless we are using verbose
        if(!m_verbose_msgs)
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
                           << "\"" << runtime_type << "\""
                           << " passed via 'runtime' open option.");
        }
     
        m_runtime->Initialize(processed_opts);
    }
    catch(conduit::Error &e)
    {
        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            // NOTE: CONDUIT_INFO could be muted, so we use std::cout
            std::cout << "[Error] Ascent::open " 
                      << e.message() << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::publish(const conduit::Node &data)
{
    try
    {
        m_runtime->Publish(data);
    }
    catch(conduit::Error &e)
    {
        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            // NOTE: CONDUIT_INFO could be muted, so we use std::cout
            std::cout << "[Error] Ascent::publish " 
                      << e.message() << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::execute(const conduit::Node &actions)
{
    try
    {
        Node processed_actions(actions);
        CheckForJSONFile("ascent_actions.json", processed_actions);
        m_runtime->Execute(processed_actions);
    }
    catch(conduit::Error &e)
    {
        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            // NOTE: CONDUIT_INFO could be muted, so we use std::cout
            std::cout << "[Error] Ascent::execute " 
                      << e.message() << std::endl;
        }
    }
}


//-----------------------------------------------------------------------------
void
Ascent::info(conduit::Node &info_out)
{
    try
    {
        if(m_runtime != NULL)
        {
            m_runtime->Info(info_out);
        }
    }
    catch(conduit::Error &e)
    {
        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            // NOTE: CONDUIT_INFO could be muted, so we use std::cout
            std::cout << "[Error] Ascent::info " 
                      << e.message() << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::close()
{
    try
    {
        if(m_runtime != NULL)
        {
            m_runtime->Cleanup();
            delete m_runtime;
            m_runtime = NULL;
        }
    }
    catch(conduit::Error &e)
    {
        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            // NOTE: CONDUIT_INFO could be muted, so we use std::cout
            std::cout << "[Error] Ascent::close " 
                      << e.message() << std::endl;
        }
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

    n["version"] = ASCENT_VERSION;

#ifdef ASCENT_GIT_SHA1
    n["git_sha1"] = CONDUIT_GIT_SHA1;
#endif
    
    n["compilers/cpp"] = ASCENT_CPP_COMPILER;
#ifdef ASCENT_FORTRAN_COMPILER
    n["compilers/fortran"] = ASCENT_FORTRAN_COMPILER;
#endif

#if   defined(ASCENT_PLATFORM_WINDOWS)
    n["platform"] = "windows";
#elif defined(ASCENT_PLATFORM_APPLE)
    n["platform"] = "apple";
#else 
    n["platform"] = "linux";
#endif
    
    n["system"] = ASCENT_SYSTEM_TYPE;
    n["install_prefix"] = ASCENT_INSTALL_PREFIX;
    n["license"] = ASCENT_LICENSE_TEXT;

#if defined(ASCENT_MPI_ENABLED)
    n["mpi"] = "enabled";
#else
    n["mpi"] = "disabled";
#endif

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



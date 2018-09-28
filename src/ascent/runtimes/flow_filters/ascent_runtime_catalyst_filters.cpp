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
/// file: ascent_runtime_catalyst_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_catalyst_filters.hpp"

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

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_CATALYST_ENABLED)

#include "vtkCommunicator.h"
#include "vtkCPAdaptorAPI.h"
#include "vtkCPDataDescription.h"
#include "vtkCPInputDataDescription.h"
#include "vtkCPProcessor.h"
#include "vtkNew.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkInformationDoubleKey.h"
#include "vtkFieldData.h"
#include "vtkSmartPointer.h"
#include "vtkTypeInt64Array.h"
#include "vtkPVConfig.h"
#ifdef PARAVIEW_ENABLE_PYTHON
#  include "vtkCPPythonScriptPipeline.h"
#endif

#include <ascent_vtk_data_adapter.hpp>

#endif

#include <stdlib.h> // for atexit

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
CatalystPythonScript::CatalystPythonScript()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
CatalystPythonScript::~CatalystPythonScript()
{
// empty
}

//-----------------------------------------------------------------------------
void
CatalystPythonScript::declare_interface(Node &i)
{
    i["type_name"]   = "catalyst_python_script";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
CatalystPythonScript::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
  info.reset();
#ifdef PARAVIEW_ENABLE_PYTHON
  bool res = true;

  if (
    ! params.has_child("script") ||
    ! params["script"].dtype().is_string())
  {
    info["errors"].append() = "Missing required string parameter 'script'";
    res = false;
  }
#else
  bool res = false;
  info["errors"].append() = "Catalyst was compiled without python support.";
#endif

  return res;
}

//-----------------------------------------------------------------------------
void
CatalystPythonScript::execute()
{
#ifdef PARAVIEW_ENABLE_PYTHON
  ASCENT_INFO("Running Catalyst Python script!");

  // I. Get data in a VTK format (if it is not already)
  vtkDataObject* data = nullptr;
  if(input(0).check_type<Node>())
  {
    // We have been passed a conduit dataset.
    const Node* node = input<Node>(0);
    conduit::Node info;

    // Verify the conduit schema is valid
    bool success = conduit::blueprint::verify("mesh",*node,info);
    if(!success)
    {
      info.print();
      ASCENT_ERROR("conduit::Node input to EnsureBlueprint is non-conforming");
      return;
    }

    data = VTKDataAdapter::BlueprintToVTKMultiBlock(
      *node, /*zero_copy*/ true, /*topo_name*/ "");
  }
  else if (input(0).check_type<vtkDataObject>())
  {
    data = input<vtkDataObject>(0);
  }
  else
  {
    ASCENT_ERROR("catalyst_python_script input must be a conduit Node or a vtk dataset");
  }

  std::string script_name = params()["script"].as_string();

  static bool once = false;
  static vtkSmartPointer<vtkCPPythonScriptPipeline> pythonPipeline;
  static vtkSmartPointer<vtkCPDataDescription> dataDesc;
  constexpr const char* meshName = "simulation";
  vtkCPProcessor* proc;
  if (!once)
  {
    // Now initialize, run, and finalize the catalyst pipeline.
    vtkCPAdaptorAPI::CoProcessorInitialize();
    pythonPipeline = vtkSmartPointer<vtkCPPythonScriptPipeline>::New();
    pythonPipeline->Initialize(script_name.c_str());
    proc = vtkCPAdaptorAPI::GetCoProcessor();
    proc->AddPipeline(pythonPipeline);
    dataDesc = vtkSmartPointer<vtkCPDataDescription>::New();
    dataDesc->AddInput(meshName);
    once = true;
  }
  else
  {
    proc = vtkCPAdaptorAPI::GetCoProcessor();
  }

  // Add data to catalyst "description":
  double time = vtkDataObject::DATA_TIME_STEP()->Get(data->GetInformation());
  auto fields = data ? data->GetFieldData() : nullptr;
  vtkTypeInt64Array* tsarr = fields ? vtkTypeInt64Array::SafeDownCast(fields->GetArray("cycle")) : nullptr;
  vtkTypeInt64 timeStep = (tsarr && tsarr->GetNumberOfTuples() > 0 ? tsarr->GetValue(0) : -1);

  dataDesc->SetTimeData(time, timeStep);
  // For now, just handle 1 data object:
  vtkCPInputDataDescription* inDesc = dataDesc->GetInputDescriptionByName(meshName);

  inDesc->SetGrid(data);
  // TODO: Set whole extent of **inDesc** if **data** is structured.
  proc->CoProcess(dataDesc);

  static bool onceAtExit = false;
  if (!onceAtExit)
  {
    onceAtExit = true;
    // Finalize catalyst each timestep
    // FIXME: This causes problems because atexit() is called after MPI_Finalize.
    //        But we do not know when CatalystPythonScript::execute is being
    //        called for the last time and CatalystPythonScript instances are
    //        created new each timestep, so putting it in the destructor will
    //        do no good.
    atexit(vtkCPAdaptorAPI::CoProcessorFinalize);
  }

  set_output<vtkDataObject>(data); // pass-through to allow other scripts, etc.
#endif // PARAVIEW_ENABLE_PYTHON
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

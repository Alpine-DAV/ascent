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
/// file: ascent_runtime_adios2_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_adios2_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_runtime_utils.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_logging.hpp>
#include <ascent_file_system.hpp>
#include <ascent_data_object.hpp>

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <ascent_runtime_vtkh_utils.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif


#include <adios2.h>
#include <set>
#include <cstring>
#include <limits>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <fides/DataSetReader.h>
#include <fides/DataSetWriter.h>

using namespace std;
using namespace conduit;
using namespace flow;

template <class T>
inline std::ostream& operator<<(ostream& os, const vector<T>& v)
{
    os<<"[";
    auto it = v.begin();
    for ( ; it != v.end(); ++it)
        os<<" "<< *it;
    os<<"]";
    return os;
}

template <class T>
inline ostream& operator<<(ostream& os, const set<T>& s)
{
    os<<"{";
    auto it = s.begin();
    for ( ; it != s.end(); ++it)
        os<<" "<< *it;
    os<<"}";
    return os;
}

static fides::io::DataSetAppendWriter *writer = NULL;

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
ADIOS2::ADIOS2()
    :Filter()
{
}

//-----------------------------------------------------------------------------
ADIOS2::~ADIOS2()
{
}

//-----------------------------------------------------------------------------
void
ADIOS2::declare_interface(Node &i)
{
    i["type_name"]   = "adios2";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
ADIOS2::verify_params(const conduit::Node &params,
                     conduit::Node &info)
{
  bool res = true;
  if (!params.has_child("filename") ||
      !params["filename"].dtype().is_string())
  {
    info["errors"].append() = "missing required entry 'filename'";
    res = false;
  }

  if (!params.has_child("engine") ||
      !params["engine"].dtype().is_string())
  {
    info["errors"].append() = "missing required entry 'engine'";
    res = false;
  }

  std::string engineType = params["engine"].as_string();
  if (engineType != "BPFile" && engineType != "SST")
  {
    info["errors"].append() = "unsupported engine type: " + engineType;
    res = false;
  }

  std::string fileName = params["filename"].as_string();
  if (engineType == "SST" && fileName.find("/") != std::string::npos )
  {
    info["errors"].append() = "filename with directory not supported for SST engine";
    res = false;
  }

  return res;
}

//-----------------------------------------------------------------------------
void
ADIOS2::execute()
{
  ASCENT_INFO("execute");

  std::string engineType = params()["engine"].as_string();
  std::string fileName   = params()["filename"].as_string();

  if (writer == NULL)
    writer = new fides::io::DataSetAppendWriter(fileName);

  if(!input(0).check_type<DataObject>())
  {
    ASCENT_ERROR("ADIOS2 input must be a data object");
  }

  //If fields set, set the WriteFields attribute.
  if (params().has_child("fields"))
  {
    std::string fields = params()["fields"].as_string();
    if (!fields.empty())
    {
      std::vector<std::string> fieldList;

      std::istringstream iss(fields);
      std::copy(std::istream_iterator<std::string>(iss),
                std::istream_iterator<std::string>(),
                std::back_inserter(fieldList));

      writer->SetWriteFields(fieldList);
    }
  }


  DataObject *data_object = input<DataObject>(0);
  std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

  std::string topo_name = detail::resolve_topology(params(),
                                                   this->name(),
                                                   collection);

  vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

  vtkm::cont::PartitionedDataSet pds;
  vtkm::Id numDS = data.GetNumberOfDomains();
  for (vtkm::Id i = 0; i < numDS; i++)
    pds.AppendPartition(data.GetDomain(i));

  writer->Write(pds, engineType);
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

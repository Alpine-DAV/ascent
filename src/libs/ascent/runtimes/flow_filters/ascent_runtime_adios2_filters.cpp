//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
#include <ascent_runtime_param_check.hpp>
#include <ascent_logging.hpp>
#include <ascent_data_object.hpp>

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <ascent_runtime_vtkh_utils.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes

#include <mpi.h>

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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";

    string_schema(param_schema["properties/filename"]);
    string_schema(param_schema["properties/engine"]);

    conduit::Node &bpfile_schema = param_schema["oneOf"].append();
    bpfile_schema["type"] = "object";
    string_schema(bpfile_schema["properties/engine"]);
    bpfile_schema["properties/engine/constraints/const"] = "BPFile";

    conduit::Node sst_schema = param_schema["oneOf"].append();
    sst_schema["type"] = "object";
    string_schema(sst_schema["properties/engine"]);
    sst_schema["properties/engine/constraints/const"] = "SST";

    conduit::Node fname = string_schema(sst_schema["properties/filename"]);
    fname["pattern"] = "^[^/]*$";
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

  viskores::cont::PartitionedDataSet pds;
  viskores::Id numDS = data.GetNumberOfDomains();
  for (viskores::Id i = 0; i < numDS; i++)
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

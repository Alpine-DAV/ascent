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

/******************************************************
TODO:

 */


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

static fides::io::DataSetWriter *writer = NULL;

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
// empty
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
#if 0
    if (!params.has_child("transport") ||
        !params["transport"].dtype().is_string())
    {
        info["errors"].append() = "missing required entry 'transport'";
        res = false;
    }


    if (!params.has_child("filename") ||
        !params["transport"].dtype().is_string() )
    {
        info["errors"].append() = "missing required entry 'filename'";
        res = false;
    }
#endif

    return res;
}

//-----------------------------------------------------------------------------
void
ADIOS2::execute()
{
  ASCENT_INFO("execute");

  if (writer == NULL)
    writer = new fides::io::DataSetWriter("out.bp");

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("ADIOS2 input must be a data object");
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

    writer->Write(pds, "BPFile");

    return;

#if 0
    if(!input("in").check_type<Node>())
    {
        // error
        ASCENT_ERROR("adios2 filter requires a conduit::Node input");
    }

    transportType = params()["transport"].as_string();
    fileName      = params()["filename"].as_string();

    // get params

    #ifdef ADIOS2_HAVE_MPI
        adios2::ADIOS adios(mpi_comm, adios2::DebugON);
    #else
        adios2::ADIOS adios;
    #endif
    adios2::IO adiosWriter = adios.DeclareIO("adiosWriter");
    // ************************************************************
    // <-----------
    //      adios_define_schema_version(adiosGroup, (char*)"1.1");
    // <-----------
    // ************************************************************

    //Fetch input data
    Node *blueprint_data = input<Node>("in");

    NodeConstIterator itr = (*blueprint_data)["coordsets"].children();
    while (itr.has_next())
    {
        const Node &coordSet = itr.next();
        std::string coordSetType = coordSet["type"].as_string();

        if (coordSetType == "uniform")
        {
            UniformMeshSchema(coordSet); // Returns false
            break;
        }
        else if (coordSetType == "rectilinear")
        {
            RectilinearMeshSchema(coordSet); // Holds a adios_write
            break;
        }
    }

    if (blueprint_data->has_child("fields"))
    {
        const Node &fields = (*blueprint_data)["fields"];
        NodeConstIterator fields_itr = fields.children();

        while(fields_itr.has_next())
        {
            const Node& field = fields_itr.next();
            std::string field_name = fields_itr.name();
            FieldVariable(field_name, field); // Has a adios_write
        }
    }

/*
    // I think this comes down but how are the variables coming through?
    if (transportType == "file")
    {
        adios2::Engine adiosWriter = bpIO.Open(fileName, adios2::Mode::Write);
        //adios_init_noxml(mpi_comm);
        ////adios_set_max_buffer_size(100);
        //adios_declare_group(&adiosGroup, "test_data", "iter", adios_stat_default);
        //adios_select_method(adiosGroup, "MPI", "", "");
        //adios_open(&adiosFile, "test_data", fileName.c_str(), "w", mpi_comm);
        adiosWriter.SetEngine("BPReader");
    }
    else if ( transportType == "sst" )
    {
        ASCENT_ERROR("SST is not enabled at this time");
        //adiosWriter.SetEngine("SST");
    }
    else
    {
        ASCENT_ERROR("Transport type: " <<transportType << " not supported!");
    }
    //adios_close(adiosFile);
    //I think I just need to open and close at write. Maybe close here
    //adiosWriter.Close();
    //adios2::Engine adiosWriter = bpIO.Open(fileName, adios2::Mode::Write);
*/
#endif
}


#if 0
//-----------------------------------------------------------------------------
bool
ADIOS2::FieldVariable(const string &fieldName, const Node &node)
{
    // TODO: we can assuem this is true if verify is true and this is a recti
    // mesh.
    if (!node.has_child("values") ||
        !node.has_child("association") ||
        !node.has_child("type"))
        return false;

    const string &fieldType = node["type"].as_string();
    const string &fieldAssoc = node["association"].as_string();

    if (fieldType != "scalar")
    {
        ASCENT_INFO("Field type "
                    << fieldType
                    << " not supported for ADIOS2 this time");
        return false;
    }
    if (fieldAssoc != "vertex" && fieldAssoc != "element")
    {
        ASCENT_INFO("Field association "
                    << fieldAssoc
                    <<" not supported for ADIOS2 this time");
        return false;
    }

    const Node &field_values = node["values"];
    const double *vals = field_values.as_double_ptr();

    /*
    DataType dt(fieldNode.dtype());
    cout<<"FIELD "<<fieldName<<" #= "<<dt.number_of_elements()<<endl;
    cout<<"localDims: "<<dimsToStr(localDims, (fieldAssoc=="vertex"))<<endl;
    cout<<"globalDims: "<<dimsToStr(globalDims, (fieldAssoc=="vertex"))<<endl;
    cout<<"offset: "<<dimsToStr(offset, (fieldAssoc=="vertex"))<<endl;
    */


    /*
    int64_t varId = adios_define_var(adiosGroup,
                                     (char*)fieldName.c_str(),
                                     "",
                                     adios_double,
                                     dimsToStr(localDims, (fieldAssoc=="vertex")).c_str(),
                                     dimsToStr(globalDims, (fieldAssoc=="vertex")).c_str(),
                                     dimsToStr(offset, (fieldAssoc=="vertex")).c_str());
    adios_define_var_mesh(adiosGroup,
                          (char*)fieldName.c_str(),
                          meshName.c_str());
    adios_define_var_centering(adiosGroup,
                               fieldName.c_str(),
                               (fieldAssoc == "vertex" ? "point" : "cell"));
    adios_write(adiosFile, fieldName.c_str(), (void*)vals);
    */
    return true;
}
#endif

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

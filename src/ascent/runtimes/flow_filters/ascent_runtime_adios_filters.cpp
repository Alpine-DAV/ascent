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

/******************************************************
TODO:

 */


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_adios_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_adios_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_file_system.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif


#include <adios.h>
#include <set>
#include <cstring>
#include <limits>

using namespace std;
using namespace conduit;
using namespace flow;

struct coordInfo
{
    coordInfo(int r, int n, double r0, double r1) : num(n), rank(r) {range[0]=r0; range[1]=r1;}
    coordInfo() {num=0; rank=-1; range[0]=range[1]=0;}
    coordInfo(const coordInfo &c) {num=c.num; rank=c.rank; range[0]=c.range[0]; range[1]=c.range[1];}
    
    int num, rank;
    double range[2];
};

inline bool operator<(const coordInfo &c1, const coordInfo &c2)
{
    return c1.range[0] < c2.range[0];
}

inline ostream& operator<<(ostream &os, const coordInfo &ci)
{
    os<<"(r= "<<ci.rank<<" : n= "<<ci.num<<" ["<<ci.range[0]<<","<<ci.range[1]<<"])";
    return os;
}

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
ADIOS::ADIOS()
    :Filter()
{
    mpi_comm = 0;
    rank = 0;
    numRanks = 1;
    meshName = "mesh";
    globalDims.resize(3);
    localDims.resize(3);
    offset.resize(3);
    for (int i = 0; i < 3; i++)
        globalDims[i] = localDims[i] = offset[i] = 0;
    
#ifdef ASCENT_MPI_ENABLED
    mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &numRanks);
#endif
}

//-----------------------------------------------------------------------------
ADIOS::~ADIOS()
{
// empty
}

//-----------------------------------------------------------------------------
void 
ADIOS::declare_interface(Node &i)
{
    i["type_name"]   = "adios";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
ADIOS::verify_params(const conduit::Node &params,
                     conduit::Node &info)
{    
    bool res = true;

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
    
    return res;
}

//-----------------------------------------------------------------------------
void 
ADIOS::execute()
{
    ASCENT_INFO("execute");

    if(!input("in").check_type<Node>())
    {
        // error
        ASCENT_ERROR("adios filter requires a conduit::Node input");
    }

    transportType = params()["transport"].as_string();
    fileName      = params()["filename"].as_string();

    // get params
    
    if (transportType == "file")
    {
        adios_init_noxml(mpi_comm);
        //adios_set_max_buffer_size(100);
        adios_declare_group(&adiosGroup, "test_data", "iter", adios_stat_default);
        adios_select_method(adiosGroup, "MPI", "", "");
        adios_open(&adiosFile, "test_data", fileName.c_str(), "w", mpi_comm);
    }
    else
    {
        //  if (transportType == "staging")
        ASCENT_ERROR("Transport type: " <<transportType << " not supported!");
    }
    adios_define_schema_version(adiosGroup, (char*)"1.1");

    //Fetch input data
    Node *blueprint_data = input<Node>("in");

    NodeConstIterator itr = (*blueprint_data)["coordsets"].children();
    while (itr.has_next())
    {
        const Node &coordSet = itr.next();
        std::string coordSetType = coordSet["type"].as_string();
        
        if (coordSetType == "uniform")
        {
            UniformMeshSchema(coordSet);
            break;
        }
        else if (coordSetType == "rectilinear")
        {
            RectilinearMeshSchema(coordSet);
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
            FieldVariable(field_name, field);
        }
    }
    adios_close(adiosFile);

}

//-----------------------------------------------------------------------------
bool
ADIOS::UniformMeshSchema(const Node &node)
{
    return false;
}

//-----------------------------------------------------------------------------
bool
ADIOS::CalcRectilinearMeshInfo(const conduit::Node &node,
                               vector<vector<double>> &XYZ)
{
    const Node &X = node["x"];
    const Node &Y = node["y"];
    const Node &Z = node["z"];
    
    const double *xyzPtr[3] = {X.as_float64_ptr(),
                               Y.as_float64_ptr(),
                               Z.as_float64_ptr()};

    localDims = {X.dtype().number_of_elements(),
                 Y.dtype().number_of_elements(),
                 Z.dtype().number_of_elements()};

    //Stuff the XYZ coordinates into the conveniently provided array.
    XYZ.resize(3);
    for (int i = 0; i < 3; i++)
    {
        XYZ[i].resize(localDims[i]);
        std::memcpy(&(XYZ[i][0]), xyzPtr[i], localDims[i]*sizeof(double));
    }

    //Participation trophy if you only bring 1 rank to the game.
    if (numRanks == 1)
    {
        offset = {0,0,0};
        globalDims = localDims;
        return true;
    }

#ifdef ASCENT_MPI_ENABLED
    
    // Have to figure out the indexing for each rank.
    vector<int> ldims(3*numRanks, 0), buff(3*numRanks,0);
    ldims[3*rank + 0] = localDims[0];
    ldims[3*rank + 1] = localDims[1];
    ldims[3*rank + 2] = localDims[2];

    int mpiStatus;
    mpiStatus = MPI_Allreduce(&ldims[0], &buff[0], ldims.size(), MPI_INT, MPI_SUM, mpi_comm);
    if (mpiStatus != MPI_SUCCESS)
        return false;

    //Calculate the global dims. This is just the sum of all the localDims.
    globalDims = {0,0,0};
    for (int i = 0; i < buff.size(); i+=3)
    {
        globalDims[0] += buff[i + 0];
        globalDims[1] += buff[i + 1];
        globalDims[2] += buff[i + 2];
    }
    
    //And now for the offsets. It is the sum of all the localDims before me.
    offset = {0,0,0};
    for (int i = 0; i < rank; i++)
    {
        offset[0] += buff[i*3 + 0];
        offset[1] += buff[i*3 + 1];
        offset[2] += buff[i*3 + 2];        
    }

#if 0
    if (rank == 0)
    {
        cout<<"***************************************"<<endl;        
        cout<<"*****globalDims: "<<globalDims<<endl;
    }
    MPI_Barrier(mpi_comm);
    for (int i = 0; i < numRanks; i++)
    {
        if (i == rank)
        {
            cout<<"  "<<rank<<": *****localDims:"<<localDims<<endl;
            cout<<"  "<<rank<<": *****offset:"<<offset<<endl;
            cout<<"  X: "<<rank<<XYZ[0]<<endl;
            cout<<"  Y: "<<rank<<XYZ[1]<<endl;
            cout<<"  Z: "<<rank<<XYZ[2]<<endl;
            cout<<"***************************************"<<endl<<endl;
        }
        MPI_Barrier(mpi_comm);        
    }
#endif
    
    return true;
    
#endif
}


//-----------------------------------------------------------------------------
bool
ADIOS::RectilinearMeshSchema(const Node &node)
{
    if (!node.has_child("values"))
        return false;
    
    const Node &coords = node["values"];
    if (!coords.has_child("x") || !coords.has_child("y") || !coords.has_child("z"))
        return false;

    vector<vector<double>> XYZ;
    if (!CalcRectilinearMeshInfo(coords, XYZ))
        return false;

    string coordNames[3] = {"coords_x", "coords_y", "coords_z"};

    //Write schema metadata for Rect Mesh.
    if (rank == 0)
    {
        /*
        cout<<"**************************************************"<<endl;
        cout<<rank<<": globalDims: "<<dimsToStr(globalDims)<<endl;
        cout<<rank<<": localDims: "<<dimsToStr(localDims)<<endl;
        cout<<rank<<": offset: "<<dimsToStr(offset)<<endl;
        */
        
        adios_define_mesh_timevarying("no", adiosGroup, meshName.c_str());
        adios_define_mesh_rectilinear((char*)dimsToStr(globalDims).c_str(),
                                      (char*)(coordNames[0]+","+coordNames[1]+","+coordNames[2]).c_str(),
                                      0,
                                      adiosGroup,
                                      meshName.c_str());
    }

    //Write out coordinates.
    int64_t ids[3];
    for (int i = 0; i < 3; i++)
    {
        ids[i] = adios_define_var(adiosGroup,
                                  coordNames[i].c_str(),
                                  "",
                                  adios_double,
                                  to_string(localDims[i]).c_str(),
                                  to_string(globalDims[i]).c_str(),
                                  to_string(offset[i]).c_str());
        adios_write_byid(adiosFile, ids[i], (void *)&(XYZ[i][0]));
    }
    
    return true;
}

//-----------------------------------------------------------------------------
bool
ADIOS::FieldVariable(const string &fieldName, const Node &node)
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
                    << " not supported for ADIOS this time");
        return false;
    }
    if (fieldAssoc != "vertex" && fieldAssoc != "element")
    {
        ASCENT_INFO("Field association "
                    << fieldAssoc 
                    <<" not supported for ADIOS this time");
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

    return true;
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






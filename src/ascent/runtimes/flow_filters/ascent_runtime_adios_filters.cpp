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

#ifdef PARALLEL
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
    
#ifdef PARALLEL
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
    cout<<__FILE__<<" "<<__LINE__<<" ****************************** ADIOS::verify_params"<<endl;
    cout<<"Params: "<<endl;
    //params.print();
    cout<<"info: "<<endl;
    info.print();
    
    bool res = true;
    
    if( !params.has_child("important_param") ) 
    {
        info["errors"].append() = "missing required entry 'important_param'";
        res = false;
    }

    if (!params.has_child("transport"))
    {
        info["errors"].append() = "missing required entry 'transport'";
        res = false;
    }
    else
        transportType = params["transport"].as_string();

    if (!params.has_child("filename"))
    {
        info["errors"].append() = "missing required entry 'filename'";
        res = false;
    }
    else
        fileName = params["filename"].as_string();
    
    return res;
}
//-----------------------------------------------------------------------------
void 
ADIOS::execute()
{
    cout<<__FILE__<<" "<<__LINE__<<" ****************************** ADIOS::execute"<<endl;

    if (transportType == "file")
    {
        adios_init_noxml(mpi_comm);
        //adios_set_max_buffer_size(100);
        adios_declare_group(&adiosGroup, "test_data", "iter", adios_stat_default);
        adios_select_method(adiosGroup, "MPI", "", "");
        adios_open(&adiosFile, "test_data", fileName.c_str(), "w", mpi_comm);
    }
    else if (transportType == "staging")
    {
        cout<<"Transport not supported!"<<endl;
        return;
    }

    adios_define_schema_version(adiosGroup, (char*)"1.1");


    //Output the mesh.
    Node *blueprint_data = input<Node>("in");

//    cout<<"DATA"<<endl;
//    blueprint_data->print();
//    cout<<"__DATA"<<endl;    

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
        vector<string> fieldNames = fields.child_names();
        for (int i = 0; i < fieldNames.size(); i++)
            FieldVariable(fieldNames[i], fields[fieldNames[i]]);
    }
    
    adios_close(adiosFile);




#if 0
    std::string important_param;
    important_param = params()["important_param"].as_string();
    if(par_rank == 0)
    {
      std::cout<<"The important param is "<<important_param<<"\n";
    }

    if(params().has_child("int"))
    {
        int the_int = params()["int"].as_int32();
    }

    if(params().has_child("float"))
    {
        float the_float = params()["float"].as_float32();
    }

    if(params().has_child("double"))
    {
        double the_double = params()["double"].as_float64();
    }

    if(params().has_child("float_values"))
    {
       float *vals = params()["float_values"].as_float32_ptr();
    }

    if(params().has_child("double_values"))
    {
       double *vals = params()["double_values"].as_float64_ptr();
    }

    if(params().has_child("actions"))
    {
      const conduit::Node actions = params()["actions"];
      if(par_rank == 0)
      {
        std::cout<<"Actions passed to adios filter:\n";
        actions.print();
      }
    }

    if(!input("in").check_type<Node>())
    {
        // error
        ASCENT_ERROR("adios requires a conduit::Node input");
    }
#endif
}

bool
ADIOS::UniformMeshSchema(const Node &node)
{
    return false;
}

bool
ADIOS::CalcRectilinearMeshInfo(const conduit::Node &node,
                               vector<vector<double>> &XYZ)
{
    const Node &X = node["x"];
    const Node &Y = node["y"];
    const Node &Z = node["z"];
    const double *xyzPtr[3] = {X.as_float64_ptr(), Y.as_float64_ptr(), Z.as_float64_ptr()};

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

#ifdef PARALLEL
    
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

bool
ADIOS::CalcRectilinearMeshInfoOLD(const conduit::Node &node,
                               vector<vector<double>> &globalCoords)
{
    const Node &X = node["x"];
    const Node &Y = node["y"];
    const Node &Z = node["z"];

    DataType dx(X.dtype()), dy(Y.dtype()), dz(Z.dtype());
    int nx = dx.number_of_elements();
    int ny = dy.number_of_elements();
    int nz = dz.number_of_elements();

    const double *xyzc[3] = {NULL,NULL,NULL};
    xyzc[0] = X.as_float64_ptr();
    xyzc[1] = Y.as_float64_ptr();
    xyzc[2] = Z.as_float64_ptr();    

    globalCoords.resize(3);

    //Easy case. 
    if (numRanks == 1)
    {
        offset = {0,0,0};
        localDims = {nx,ny,nz};
        globalDims = {nx,ny,nz};

        for (int d = 0; d < 3; d++)
        {
            globalCoords[d].resize(globalDims[d]);
            std::memcpy(&globalCoords[d][0], xyzc[d], globalDims[d]*sizeof(double));
        }
        
        return true;
    }
#ifdef PARALLEL

    // Have to figure out the indexing for each rank.
    vector<double> ldims(10*numRanks, 0.0), buff(10*numRanks,0.0);
    ldims[rank*10 + 0] = rank;
    ldims[rank*10 + 1] = nx;
    ldims[rank*10 + 2] = xyzc[0][0];
    ldims[rank*10 + 3] = xyzc[0][nx-1];
    ldims[rank*10 + 4] = ny;
    ldims[rank*10 + 5] = xyzc[1][0];
    ldims[rank*10 + 6] = xyzc[1][ny-1];
    ldims[rank*10 + 7] = nz;
    ldims[rank*10 + 8] = xyzc[2][0];
    ldims[rank*10 + 9] = xyzc[2][nz-1];
    MPI_Allreduce(&ldims[0], &buff[0], ldims.size(), MPI_DOUBLE, MPI_SUM, mpi_comm);
    //if (rank == 0) cout<<"LDIMS: "<<buff<<endl;

    //Put this data into a set to remove duplicates, AND keep them sorted (see operator< above).
    set<coordInfo> xyzInfo[3];
    for (int i = 0; i < numRanks; i++)
    {
        xyzInfo[0].insert(coordInfo((int)buff[i*10+0],
                                    (int)buff[i*10+1],
                                    buff[i*10+2],
                                    buff[i*10+3]));
        
        xyzInfo[1].insert(coordInfo((int)buff[i*10+0],
                                    (int)buff[i*10+4],
                                    buff[i*10+5],
                                    buff[i*10+6]));
        
        xyzInfo[2].insert(coordInfo((int)buff[i*10+0],
                                    (int)buff[i*10+7],
                                    buff[i*10+8],
                                    buff[i*10+9]));
    }
    
    //Now we can compute globalDims.
    for (int i = 0; i < 3; i++)
    {
        globalDims[i] = 0;
        for (auto it = xyzInfo[i].begin(); it != xyzInfo[i].end(); it++)
            globalDims[i] += (*it).num;
        globalDims[i] -= (xyzInfo[i].size()-1);
    }

    //Local dims is just {nx,ny,nz};
    localDims = {nx,ny,nz};

    //Compute offset.
    vector<coordInfo> localCoords = {coordInfo(rank, nx, xyzc[0][0], xyzc[0][nx-1]),
                                     coordInfo(rank, ny, xyzc[1][0], xyzc[1][ny-1]),
                                     coordInfo(rank, nz, xyzc[2][0], xyzc[2][nz-1])};
    for (int i = 0; i < 3; i++)
    {
        for (auto it = xyzInfo[i].begin(); it != xyzInfo[i].end(); it++)
        {
            if (localCoords[i].range[0] > (*it).range[0])
                offset[i] += (*it).num;
        }
    }

    //Get the coordinates onto rank 0.
    for (int i = 0; i < 3; i++)
    {
        globalCoords[i].resize(globalDims[i]);
        
        //One rank has the whole coordinate array.
        if (xyzInfo[i].size() == 1)
        {
            auto it = xyzInfo[i].begin();
            if (it->rank == 0)
                std::memcpy(&globalCoords[i][0], xyzc[i], globalDims[i]*sizeof(double));
            else
            {
                int tag = 9382;
                MPI_Status stat;
                int sendRes = MPI_SUCCESS, recvRes = MPI_SUCCESS;
                if (it->rank == rank)
                    sendRes = MPI_Send(xyzc[i], globalDims[i], MPI_DOUBLE, 0, tag, mpi_comm);
                else
                    recvRes = MPI_Recv(&globalCoords[i], globalDims[i], MPI_DOUBLE, it->rank, tag, mpi_comm, &stat);
                if (sendRes != MPI_SUCCESS || recvRes != MPI_SUCCESS)
                {
                    cerr<<"Error: rectilinear send/recv for single rank"<<endl;
                    return false;
                }
            }
        }
        // Multiple ranks have the coordinate array.
        // Not the most efficient way to do this, but the easiest...
        else
        {
            vector<double> tmp(globalDims[i], -numeric_limits<double>::max());
            int off = 0;
            for (auto it = xyzInfo[i].begin(); it != xyzInfo[i].end(); it++)
            {
                if (it->rank == rank)
                {
                    for (int j = 0; j < localDims[i]; j++)
                        tmp[off+j] = xyzc[i][j];
                }
                off += it->num-1;
            }
            //cout<<rank<<" tmp= "<<tmp<<" "<<count[i]<<endl;
            int res = MPI_Reduce(&tmp[0], &globalCoords[i][0], tmp.size(), MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
            if (res != MPI_SUCCESS)
            {
                cerr<<"Error: rectilinear coordinate communication"<<endl;
                return false;
            }
        }
    }
    if (rank == 1)
        localDims[0]--;
    
    if (rank == 0)
    {
        cout<<"*****X: "<<globalCoords[0].size()<<": "<<globalCoords[0]<<endl;
        cout<<"*****Y: "<<globalCoords[1].size()<<": "<<globalCoords[1]<<endl;
        cout<<"*****Z: "<<globalCoords[2].size()<<": "<<globalCoords[2]<<endl;
        cout<<"*****xinfo: "<<xyzInfo[0]<<endl;
        cout<<"*****yinfo: "<<xyzInfo[1]<<endl;
        cout<<"*****zinfo: "<<xyzInfo[2]<<endl;
        cout<<"*****globalDims: "<<globalDims<<endl;
        cout<<"***************************************"<<endl;
    }
    MPI_Barrier(mpi_comm);
    for (int i = 0; i < numRanks; i++)
    {
        if (i == rank)
        {
            cout<<"  "<<rank<<": *****localDims:"<<localDims<<endl;
            cout<<"  "<<rank<<": *****offset:"<<offset<<endl;
            cout<<"***************************************"<<endl<<endl;
        }
        MPI_Barrier(mpi_comm);        
    }
#endif

    return true;
}
    
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

bool
ADIOS::FieldVariable(const string &fieldName, const Node &node)
{
    if (!node.has_child("values") || !node.has_child("association") || !node.has_child("type"))
        return false;

    const string &fieldType = node["type"].as_string();
    const string &fieldAssoc = node["association"].as_string();

    if (fieldType != "scalar")
    {
        cerr<<"Field type "<<fieldType<<" not supported at this time"<<endl;
        return false;
    }
    if (fieldAssoc != "vertex" && fieldAssoc != "element")
    {
        cerr<<"Field association "<<fieldAssoc<<" not supported at this time"<<endl;
        return false;        
    }

    const Node &fieldNode = node["values"];
    DataType dt(fieldNode.dtype());
    const double *vals = fieldNode.as_float64_ptr();

    /*
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






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
/// file: ascent_runtime_adios_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_FLOW_PIPELINE_ADIOS_FILTERS_HPP
#define ASCENT_FLOW_PIPELINE_ADIOS_FILTERS_HPP

#include <flow_filter.hpp>
#ifdef PARALLEL
#include <mpi.h>
#endif

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
///
/// Filters Related to Conduit Relay IO
///
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class ADIOS : public ::flow::Filter
{
public:
    ADIOS();
   ~ADIOS();
    
    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();

private:
    bool UniformMeshSchema(const conduit::Node &node);
    
    bool RectilinearMeshSchema(const conduit::Node &node);
    
    bool FieldVariable(const std::string &fieldName, const conduit::Node &fieldNode);

    bool CalcRectilinearMeshInfo(const conduit::Node &node,
                                 std::vector<std::vector<double>> &globalCoords);

    int rank, numRanks;
#ifdef PARALLEL
    MPI_Comm mpi_comm;
#else
    int mpi_comm;
#endif
    
    int64_t adiosGroup, adiosFile;
    std::string transportType;
    std::string fileName;
    std::string meshName;

    //var dimensions for this rank:
    std::vector<int64_t> globalDims, localDims, offset;

    template <typename T>
    std::string dimsToStr(const std::vector<T> &d, bool pointCentered=true)
    {
        std::string str("");
        if (d.size() > 0)
        {
            for (int i = 0; i < d.size()-1; i++)
            {
                T v = d[i];
                if (!pointCentered && v > 0)
                    v--;
                str = str + std::to_string(v) + ",";
            }

            T v = d[d.size()-1];
            if (!pointCentered && v > 0)
                v--;
            str = str + std::to_string(v);
        }
        return str;
    }
};

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


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------



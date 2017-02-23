//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: alpine_block_timer.hpp
///
//-----------------------------------------------------------------------------

#ifndef ALPINE_BLOCK_TIMER_HPP
#define ALPINE_BLOCK_TIMER_HPP

#define ALPINE_BLOCK_TIMER(NAME) alpine::BlockTimer ALPINE_BLOCK_TIMER_##NAME(#NAME);
#define MAX_DEPTH 5

#include<sys/time.h>
#include <string>
#include <map>
#include <set>
#include <cstdlib>
    
#include <conduit.hpp>
#include <alpine_config.h>

//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

//-----------------------------------------------------------------------------
class BlockTimer
{
public:
    // methods
    BlockTimer(const std::string &name);
    ~BlockTimer();
    static void StartTimer(const char *name);
    static void StopTimer(const char *name);
    static conduit::Node &Finalize();
    static void           WriteLogFile();

private:
    
    static void Start(const std::string &name);
    static void Stop(const std::string &name);
    static inline conduit::Node &GlobalRoot() 
        {return s_global_root;}

    static void ReduceGlobalRoot();
    
    // non-static methods
    
    // Initializes values if the current location hasn't been visited yet,
    // and updates the set of visited locations.
    static void Precheck();

    // Goes up one function in the current location path.
    static void GoUp();

    // non-static data members
    std::string m_name;

    // private static methods
    static void ReduceAll(conduit::Node &);
    static conduit::Node &CurrentNode();
    
    static void Reduce(conduit::Node &,
                       conduit::Node &);

    static bool CheckForKnownPath(std::string &);

    static void AverageByCount(conduit::Node &,
                               int);
    static void FillDataArray(conduit::Node &,
                              const int &,
                              double &,
                              double [][3],
                              std::string);
    // static data members 
    static conduit::Node                  s_global_root;
    static int                            s_rank; // MPI rank
    static int                            s_global_depth;
    static std::string                    s_current_path;
    static std::map<std::string, timeval> s_timers;
    static std::set<std::string>          s_visited;
    
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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
/// file: block_timer.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_block_timer.hpp"
#include <climits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <set>
#include <map>
#include <fstream>
#ifdef STRAWMAN_PLATFORM_UNIX
#include <sys/sysinfo.h>
#endif



using namespace conduit;

#ifdef PARALLEL
#include "conduit_relay_mpi.hpp"
#include <mpi.h>
using namespace conduit::relay::mpi;
#endif



//-----------------------------------------------------------------------------
// -- begin strawman:: --
//-----------------------------------------------------------------------------
namespace strawman
{

// Initialize BlockTimer static data members.
int                             BlockTimer::s_global_depth = 0;
conduit::Node                   BlockTimer::s_global_root;
std::string                     BlockTimer::s_current_path = "";
std::map<std::string, timeval>  BlockTimer::s_timers;
std::set<std::string>           BlockTimer::s_visited;
int                             BlockTimer::s_rank = 0;

//-----------------------------------------------------------------------------
BlockTimer::BlockTimer(std::string const &name)
: m_name(name)
{
  Start(name);
}

//-----------------------------------------------------------------------------
void
BlockTimer::StartTimer(const char *name)
{
  std::string s_name(name);
  Start(s_name);
}
//-----------------------------------------------------------------------------
void
BlockTimer::StopTimer(const char *name)
{
  std::string s_name(name);
  Stop(s_name);
}
//-----------------------------------------------------------------------------
int
parseLine(char *line)
{
    int i = strlen(line);
    while (*line < '0' || *line > '9')
    {
        line++;
    }
    
    line[i-3] = '\0';
    i = atoi(line);
    
    return i;
}

//-----------------------------------------------------------------------------
void
BlockTimer::Start(const std::string &name)
{
#ifdef PARALLEL
    MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    s_rank = 0;
#endif
    
    ++s_global_depth;

    if (s_global_depth <= MAX_DEPTH)
    {
        s_current_path += "children/" + name + "/";
        Precheck();

        // Start timing.
        if (s_timers.count(name) == 0)
        {
            timeval timer;
            s_timers.insert(std::pair<std::string, timeval>(name, timer));
        }
        gettimeofday(&s_timers[name], NULL);
    }

}
//-----------------------------------------------------------------------------
void
BlockTimer::Stop(const std::string &name)
{
#ifdef PARALLEL
    //MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (s_global_depth <= MAX_DEPTH)
    {
        // Record timer.
        timeval start, end;
        gettimeofday(&end, NULL);
        start = s_timers[name];

        // Calculate elapsed time.
        double elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000;

        Node &curr = CurrentNode();

        // Update time spent at current location. added after max (changed)
        double newval = curr["value"].as_float64() + elapsed_time;
        
        curr["value"] = newval;
        curr["min"]   = newval;
        curr["avg"]   = newval;

        //increment the counter
        unsigned int count = curr["count"].as_uint32() + 1;
        curr["count"] = count;
       
        // 
        // Get system memory info and average
        //

        // Get memory info
#ifdef STRAWMAN_PLATFORM_UNIX
        struct sysinfo system_info;
        sysinfo(&system_info);
        long long memUsed = (system_info.totalram -system_info.freeram);
        memUsed *= system_info.mem_unit;
        memUsed = memUsed / 1024 / 1024;
        unsigned int cSysMem = curr["sysMemUsed"].as_uint64();
        
        cSysMem = ((cSysMem * (count - 1) + memUsed) )/ count;
        curr["sysMemUsed"] = uint64(cSysMem);
        
        // 
        // Get process memory usage and average
        //
        FILE* file = fopen("/proc/self/status", "r");
        int kb = -1;
        char line[128];
        while (fgets(line, 128, file) != NULL)
        {
            if (strncmp(line, "VmRSS:", 6) == 0)
            {
                kb = parseLine(line);
                break;
            }
        }
        fclose(file);
        
        kb = kb / 1024;
        
        int cProcUsage = curr["procMemMB"].as_int32();
        cProcUsage = ((cProcUsage * (count - 1) + kb ))  / count;
        curr["procMemMB"] = cProcUsage;

#else
        curr["sysMemUsed"] = 0;
        curr["procMemMB"]  = 0;
#endif
        GoUp();
    }
    
    // Update current location.
    --s_global_depth;

}
//-----------------------------------------------------------------------------
BlockTimer::~BlockTimer()
{
  Stop(m_name);
}

//-----------------------------------------------------------------------------
Node &
BlockTimer::Finalize()
{
    BlockTimer::ReduceGlobalRoot();
    return GlobalRoot();
}


//-----------------------------------------------------------------------------
Node &
BlockTimer::CurrentNode()
{
    return s_global_root[s_current_path];
}

//-----------------------------------------------------------------------------
// Initializes values if the current location hasn't been s_visited yet,
// and updates the set of s_visited locations.
//-----------------------------------------------------------------------------
void BlockTimer::Precheck()
{
    if (s_visited.count(s_current_path + "value") == 0)
    { // != "" is to prevent a root of ""
        Node &curr= CurrentNode();
        curr["value"]      = 0.0;
        curr["id"]         = s_rank;
        curr["count"]      = 0u;
        // added after max (the following 3)
        curr["min"]        = 0.0;
        curr["minid"]      = s_rank;
        curr["avg"]        = 0.0;
        curr["sysMemUsed"] = 0ul;
        curr["procMemMB"]  = 0;

        s_visited.insert(s_current_path + "value");
    }
}

//-----------------------------------------------------------------------------
bool
BlockTimer::CheckForKnownPath(std::string &path)
{
  if(path == "value")       return true;
  if(path == "id")          return true;
  if(path == "count")       return true;
  if(path == "avg")         return true;
  if(path == "minimum")     return true;
  if(path == "minid")       return true;
  if(path == "sysMemUsed")  return true;
  if(path == "procMemMB")   return true;
  return false;
}
//-----------------------------------------------------------------------------
// Goes up one function in the current location path.
//-----------------------------------------------------------------------------
void
BlockTimer::GoUp()
{
    const unsigned int len = s_current_path.length();
    if (len == 0)
    {
        s_current_path = "";
        return;
    }
    
    unsigned int ctr = 1;
    unsigned int numslashes = 0;
  
    std::string::iterator striter = s_current_path.end();
    --striter;

    while (ctr < len)
    {
        if (*striter == '/')
        {
            numslashes += 1;
            if (numslashes >= 3)
            {
                s_current_path = s_current_path.substr(0, len - ctr + 1);
                return;
            }
        }
        
        --striter;
        ++ctr;
    }
    
    s_current_path = "";
    return;
}

//-----------------------------------------------------------------------------
void 
//-----------------------------------------------------------------------------
BlockTimer::Reduce(Node &a, Node &b)
{
    // If a has it
    if (a.dtype().is_object() && a.has_path("value")) 
    {
      // Update (reduce) data
      a["count"] = a["count"].as_uint32() + b["count"].as_uint32();
      
      if (b["value"].as_float64() > a["value"].as_float64())
      {
          a["value"] = b["value"];
          a["id"] = b["id"];
      }
      
      // added after max
      if (a["min"].as_float64() > b["min"].as_float64())
      {
          a["min"] = b["min"];
          a["minid"] = b["minid"];
      }
      
      unsigned int count_a = a["count"].as_uint32();
      unsigned int count_b = b["count"].as_uint32();
      a["avg"] = (a["avg"].as_float64() * count_a + b["avg"].as_float64() * count_b) / (count_a + count_b);
    }
  
    NodeIterator itr_b(&b);

    while(itr_b.has_next())
    {
        itr_b.next();
        std::string bpath = itr_b.name();
        //
        // If we don't know the path then
        // it is a timer that needs processing 
        //
        if(CheckForKnownPath(bpath))
        {
            continue;
        }
        
        if (a.dtype().is_object() &&  a.has_path(bpath))
        {
            Reduce(a[bpath], b[bpath]);
        }
    }

    return;
}
//-----------------------------------------------------------------------------
void 
BlockTimer::AverageByCount(Node &node, int numRanks)
{
    if(node.dtype().is_object() && node.has_path("value"))
    {
        double count  = node["count"].as_uint32()  / numRanks;
        node["value"] = node["value"].as_float64() / count;
        node["min"]   = node["min"].as_float64()   / count;
        node["avg"]   = node["avg"].as_float64()   / count;
        node["count"] = uint32(count);
    }
  
  
    NodeIterator itr = node.children();

    while(itr.has_next()) 
    {
        Node &curr_node = itr.next();
        std::string curr_path = itr.name();

        if(CheckForKnownPath(curr_path))
        {
            continue;
        }
        
        AverageByCount(curr_node, numRanks);
    }

    return;
}

//-----------------------------------------------------------------------------
void 
BlockTimer::ReduceAll(Node &thisRanksNode)
{
#ifdef PARALLEL
    int temp;
    MPI_Comm_size(MPI_COMM_WORLD, &temp);
    const unsigned int numProcesses = temp;
    MPI_Comm_rank(MPI_COMM_WORLD, &temp);
    const unsigned int rank = temp;

    std::vector<Node> recvNodes(numProcesses+1);

    if (rank == 0)
    {
        for (unsigned int i = 1; i < numProcesses; ++i)
        {
          recv(recvNodes[i], // node
               i, // source
               42, // tag
               MPI_COMM_WORLD // comm
               );
           }
     }
     else 
     { 
         send(thisRanksNode, // node
              0, // dest
              42, // tag
              MPI_COMM_WORLD // comm
              );
    }

    if (rank == 0)
    {
        for (unsigned int i = 1; i < numProcesses; ++i)
        {
            Reduce(thisRanksNode.fetch("children"),
                   recvNodes[i].fetch("children"));
        }
    }
  
    // Get the average time per iteration
    if(rank == 0)
    {
        AverageByCount(s_global_root, numProcesses);
    }

#else
    AverageByCount(BlockTimer::GlobalRoot(),1);
#endif
}

//-----------------------------------------------------------------------------
void BlockTimer::ReduceGlobalRoot()
{
    ReduceAll(GlobalRoot());
}

//-----------------------------------------------------------------------------
void BlockTimer::WriteLogFile()
{
    BlockTimer::ReduceGlobalRoot();

    std::string logfile = "strawman.log";
    
    if(s_rank == 0 )
    {   
        GlobalRoot().print();
        GlobalRoot().to_json_stream(logfile.c_str(), "json", 2, 5);
    }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end strawman:: --
//-----------------------------------------------------------------------------


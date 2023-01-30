//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: block_timer.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_block_timer.hpp"
#include <climits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <set>
#include <map>
#include <fstream>
#ifdef ASCENT_PLATFORM_UNIX
#include <sys/sysinfo.h>
#endif


using namespace conduit;

#ifdef ASCENT_MPI_ENABLED
#include "conduit_relay_mpi.hpp"
#include <mpi.h>
using namespace conduit::relay::mpi;
#endif



//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

// Initialize BlockTimer static data members.
int                                            BlockTimer::s_global_depth = 0;
conduit::Node                                  BlockTimer::s_global_root;
std::string                                    BlockTimer::s_current_path = "";
std::map<std::string, BlockTimer::time_point>  BlockTimer::s_timers;
std::set<std::string>                          BlockTimer::s_visited;
int                                            BlockTimer::s_rank = 0;

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
#ifdef ASCENT_MPI_ENABLED
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
            s_timers.insert(std::pair<std::string, time_point>(name,
                                                               high_resolution_clock::now()));
        }
    }

}
//-----------------------------------------------------------------------------
void
BlockTimer::Stop(const std::string &name)
{
#ifdef ASCENT_MPI_ENABLED
    //MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (s_global_depth <= MAX_DEPTH)
    {
        // Record timer.
        auto t_start = s_timers[name];
        auto ftime = std::chrono::duration_cast<fsec>(high_resolution_clock::now() - t_start);
        double elapsed_time = ftime.count();

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
#ifdef ASCENT_PLATFORM_UNIX
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
#ifdef ASCENT_MPI_ENABLED
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

    std::string logfile = "ascent.log";

    if(s_rank == 0 )
    {
        GlobalRoot().print();
        GlobalRoot().to_json_stream(logfile.c_str(), "json", 2, 5);
    }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


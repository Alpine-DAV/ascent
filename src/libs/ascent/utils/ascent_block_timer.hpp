//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_block_timer.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_BLOCK_TIMER_HPP
#define ASCENT_BLOCK_TIMER_HPP

#define ASCENT_BLOCK_TIMER(NAME) ascent::BlockTimer ASCENT_BLOCK_TIMER_##NAME(#NAME);
#define MAX_DEPTH 5

#include <string>
#include <map>
#include <set>
#include <cstdlib>
#include <chrono>

#include <conduit.hpp>
#include <ascent_config.h>
#include <ascent_exports.h>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
class ASCENT_API BlockTimer
{
    using time_point = std::chrono::high_resolution_clock::time_point;
    using high_resolution_clock = std::chrono::high_resolution_clock;
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
    static conduit::Node                     s_global_root;
    static int                               s_rank; // MPI rank
    static int                               s_global_depth;
    static std::string                       s_current_path;
    static std::map<std::string, time_point> s_timers;
    static std::set<std::string>             s_visited;

};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------




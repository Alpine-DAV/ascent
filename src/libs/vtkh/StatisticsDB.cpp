#include <vtkh/vtkh.hpp>
#include <vtkh/StatisticsDB.hpp>

namespace vtkh
{
StatisticsDB stats;

void
StatisticsDB::DumpStats(const std::string &fname, const std::string &preamble, bool append)
{
    stats.calcStats();

#ifdef VTKH_PARALLEL
    int rank = vtkh::GetMPIRank();
    if (rank != 0)
        return;
#endif

    if (!append || !outputStream.is_open())
        outputStream.open(fname, std::ofstream::out);

    if (!preamble.empty())
        outputStream<<preamble;

    if (!stats.timers.empty())
    {
        outputStream<<"TIMERS:"<<std::endl;
        for (auto &ti : stats.timers)
            outputStream<<ti.first<<": "<<ti.second.GetTime()<<std::endl;
        outputStream<<std::endl;
        outputStream<<"TIMER_STATS"<<std::endl;
        for (auto &ti : stats.timers)
            outputStream<<ti.first<<" "<<stats.timerStat(ti.first)<<std::endl;
    }
    if (!stats.counters.empty())
    {
        outputStream<<std::endl;
        outputStream<<"COUNTERS:"<<std::endl;
        for (auto &ci : stats.counters)
            outputStream<<ci.first<<" "<<stats.totalVal(ci.first)<<std::endl;
        outputStream<<std::endl;
        outputStream<<"COUNTER_STATS"<<std::endl;
        for (auto &ci : stats.counters)
            outputStream<<ci.first<<" "<<stats.counterStat(ci.first)<<std::endl;
    }
}

};

#ifndef __STATISTICS_DB_H
#define __STATISTICS_DB_H

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

#include <math.h>
#include <time.h>
#include <sys/timeb.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ratio>

#include <vtkh/vtkh_exports.h>
#include <vtkh/utils/StreamUtil.hpp>

namespace vtkh
{

class VTKH_API StopWatch
{
public:
    StopWatch(bool _keepHistory=false) : t(0.0f), isRunning(false), keepHistory(_keepHistory) {}
    ~StopWatch() {}

    void Start()
    {
        startTime = std::chrono::high_resolution_clock::now();
        isRunning = true;
    }

    double Stop()
    {
        endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timeDiff = endTime-startTime;
        double dt = timeDiff.count();
        t += dt;

        if (keepHistory)
            history.push_back(dt);
        isRunning = false;
        return t;
    }
    void Reset() {t=0.0; history.resize(0);}
    bool IsRunning() const {return isRunning;}
    double GetTime() const {return t;}

private:
    bool isRunning, keepHistory;
    double t;
    std::chrono::high_resolution_clock::time_point startTime, endTime;
    std::vector<double> history;
};


class VTKH_API EventHistory
{
public:
    EventHistory() : t0(-1000) {}
    void SetT0(double t) {t0=t;}

    double getTime()
    {
#ifdef VTKH_PARALLEL
        return MPI_Wtime();
#else
    return std::chrono::system_clock::now().time_since_epoch() / 
                std::chrono::seconds(1);
#endif
    }

    void begin() {ts = getTime()-t0;}
    void end() {history.push_back(std::make_pair(ts, getTime()-t0));}

    void Normalize(float tmin, float tmax)
    {
        float dt = tmax-tmin;
        for (int i = 0; i < history.size(); i++)
        {
            float t0 = history[i].first;
            float t1 = history[i].second;
            t0 = (t0-tmin)/dt;
            t1 = (t1-tmin)/dt;
            history[i].first = t0;
            history[i].second = t1;
        }
    }

    double ts, t0;
    std::vector<std::pair<double,double>> history;
};

class VTKH_API StatisticsDB
{
public:

    template <typename T>
    struct statValue
    {
        statValue() {}
        statValue(std::vector<T> vals)
        {
            if (vals.empty())
            {
                minI = maxI = -1;
                min = max = med = sum = -1;
                mean = std_dev = -1;
            }
            else
            {
                sum = accumulate(vals.begin(), vals.end(), 0);

                //Get min/max info.
                auto res = minmax_element(vals.begin(), vals.end());
                minI = res.first - vals.begin();
                maxI = res.second - vals.end();
                min = (*res.first);
                max = (*res.second);

                //compute mean/median
                int n = vals.size();
                sort(vals.begin(), vals.end());
                mean = (float)sum / (float)n;
                med = vals[vals.size()/2];

                //compute standard deviation
                float x = 0;
                for (int i = 0; i < n; i++)
                    x += (vals[i]-mean)*(vals[i]-mean);

                std_dev = sqrt(x/(float)n);
                values = vals;
            }
        }

        int minI, maxI;
        T min,max, med, sum;
        float mean, std_dev;
        std::vector<T> values;

        friend std::ostream &
        operator<<(std::ostream &os, const statValue<T> &s)
        {
            return os<<"AVG: "<<s.mean<<" MED: "<<s.med<<" ("<<s.min<<","<<s.max<<":"<<s.std_dev<<")";
        }
    };

    StatisticsDB() : statsComputed(false)
    {
    }

    StatisticsDB(const StatisticsDB &s)
    {
        statsComputed = s.statsComputed;
        timers = s.timers;
        counters = s.counters;
        timerStats = s.timerStats;
        counterStats = s.counterStats;
    }

    void insert(const std::vector<StatisticsDB> &v)
    {
        statsComputed = false;
        for (int i = 0; i < v.size(); i++)
        {
            const StatisticsDB &s = v[i];
            for (auto ti = s.timers.begin(); ti != s.timers.end(); ti++)
            {
                if (timers.find(ti->first) != timers.end())
                    throw std::runtime_error("Duplicate timer: "+ti->first);
                timers[ti->first] = ti->second;
            }

            for (auto ci = s.counters.begin(); ci != s.counters.end(); ci++)
            {
                if (counters.find(ci->first) != counters.end())
                    throw std::runtime_error("Duplicate counter: "+ci->first);
                counters[ci->first] = ci->second;
            }

            for (auto ei = s.events.begin(); ei != s.events.end(); ei++)
            {
                if (events.find(ei->first) != events.end())
                    throw std::runtime_error("Duplicate event: "+ei->first);
                events[ei->first] = ei->second;
            }
        }
    }

    ~StatisticsDB() {}

    //Timers.
    void AddTimer(const std::string &nm, bool keepHistory=false)
    {
        if (timers.find(nm) != timers.end())
            throw nm + " timer already exists!";
        timers[nm] = StopWatch(keepHistory);
        timers[nm].Reset();
    }
    void Start(const std::string &nm) {vt(nm); timers[nm].Start();}
    float Stop(const std::string &nm) {vt(nm); return timers[nm].Stop();}
    float Time(const std::string &nm) {vt(nm); return timers[nm].GetTime();}
    void Reset(const std::string &nm) {vt(nm); timers[nm].Reset();}

    //Counters.
    void AddCounter(const std::string &nm)
    {
        if (counters.find(nm) != counters.end())
            throw nm + " counter already exists!";
        counters[nm] = 0;
    }
    void Increment(const std::string &nm) {vc(nm); counters[nm]++;}
    void Increment(const std::string &nm, unsigned long val) {vc(nm); counters[nm]+=val;}
    unsigned long val(const std::string &nm) {vc(nm); return counters[nm];}

    //Events.
    void AddEvent(const std::string &nm)
    {
        if (events.find(nm) != events.end())
            throw nm + " event already exists!";
        events[nm] = EventHistory();
    }
    void Begin(const std::string &nm) {ve(nm); events[nm].begin();}
    void End(const std::string &nm) {ve(nm); events[nm].end();}
    void SetEventT0(double t0)
    {
        for (auto it = events.begin(); it != events.end(); it++)
            it->second.SetT0(t0);
    }

    //Output to file
    void DumpStats(const std::string &fname, const std::string &preamble="", bool append=false);

    statValue<float> timerStat(const std::string &nm) {cs(); return timerStats[nm];}
    statValue<unsigned long> counterStat(const std::string &nm) {cs(); return counterStats[nm];}
    unsigned long totalVal(const std::string &nm) {cs(); return counterStats[nm].sum;}

    void cs() {calcStats();}
    void calcStats()
    {
        if (statsComputed)
            return;

#ifdef VTKH_PARALLEL
        int rank, nProcs;
        MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int sz = 0;

        sz = timers.size();
        if (sz > 0)
        {
            std::vector<float> vals(sz);
            std::map<std::string,StopWatch>::iterator it = timers.begin();
            for (int i = 0; it != timers.end(); it++, i++)
                vals[i] = it->second.GetTime();

            it = timers.begin();
            for (int i = 0; i < sz; i++, it++)
            {
                std::vector<float> res(nProcs, 0.0);
                if (nProcs == 1)
                    res[0] = vals[i];
                else
                {
                    std::vector<float>tmp(nProcs,0.0);
                    tmp[rank] = vals[i];
                    MPI_Reduce(&tmp[0], &res[0], nProcs, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                }
                if (rank == 0)
                    timerStats[it->first] = statValue<float>(res);
            }
        }

        sz = counters.size();
        if (sz > 0)
        {
            std::vector<unsigned long> vals(sz);
            std::map<std::string,unsigned long>::iterator it = counters.begin();
            for (int i = 0; it != counters.end(); it++, i++)
                vals[i] = it->second;

            it = counters.begin();
            for (int i = 0; i < sz; i++, it++)
            {
                std::vector<unsigned long> res(nProcs,0);
                if (nProcs == 1)
                    res[0] = vals[i];
                else
                {
                    std::vector<unsigned long> tmp(nProcs,0);
                    tmp[rank] = vals[i];
                    MPI_Reduce(&tmp[0], &res[0], nProcs, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                }
                if (rank == 0)
                    counterStats[it->first] = statValue<unsigned long>(res);
            }
        }

        sz = events.size();
        if (sz > 0)
        {
            //Normalize all the values.
            std::vector<float> vals0(nProcs,0.0f), valsN(nProcs, 0.0f);

            //find min/max per rank.
            float myMin = std::numeric_limits<float>::max();
            float myMax = std::numeric_limits<float>::min();
            int myMaxSz = -1;

            auto it = events.begin();
            for (int i = 0; i < sz; i++, it++)
            {
                int n = it->second.history.size();
                if (n == 0)
                    continue;
                float v0 = it->second.history[0].first;
                float vn = it->second.history[n-1].second;

                if (v0 < myMin) myMin = v0;
                if (vn > myMax) myMax = vn;
                if (n > myMaxSz) myMaxSz = n;
            }

            float allMin, allMax;
            int allMaxSz;
            MPI_Allreduce(&myMin, &allMin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&myMax, &allMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&myMaxSz, &allMaxSz, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            int buffSz = allMaxSz*2 + 1, tag = 0;
            for (it = events.begin(); it != events.end(); it++)
            {
                //Normalize timings.
                it->second.Normalize(allMin, allMax);

                //Rank 0 will recv everything.
                if (rank == 0)
                {
                    eventStats.resize(nProcs);
                    for (int i = 0; i < nProcs; i++) eventStats[i][it->first] = EventHistory();

                    std::vector<float> buff(buffSz);
                    for (int i = 1; i < nProcs; i++)
                    {
                        MPI_Status status;
                        MPI_Recv(&buff[0], buffSz, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
                        int n = int(buff[0]);
                        for (int j = 0; j < n; j+=2)
                            eventStats[i][it->first].history.push_back(std::make_pair(buff[1+j], buff[1+j+1]));
                    }
                    //Put rank 0 data into global stats.
                    eventStats[0][it->first] = it->second;
                }
                else
                {
                    std::vector<float> evData(buffSz, 0.0f);
                    int sz = it->second.history.size();

                    evData[0] = sz*2;
                    for (int j = 0; j < sz; j++)
                    {
                        evData[1+j*2+0] = it->second.history[j].first;
                        evData[1+j*2+1] = it->second.history[j].second;
                    }
                    MPI_Send(&evData[0], buffSz, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
                }
            }
        }
#endif

        statsComputed = true;
    }

private:
    void vt(const std::string &nm)
    {
        if (timers.find(nm) == timers.end())
        {
            std::string msg = nm + " timer not found.";
            std::cerr<<"Error: "<<msg<<std::endl;
            throw msg;
        }
    }

    void vc(const std::string &nm)
    {
        if (counters.find(nm) == counters.end())
        {
            std::string msg = nm + " counter not found.";
            std::cerr<<"Error: "<<msg<<std::endl;
            throw msg;
        }
    }

    void ve(const std::string &nm)
    {
        if (events.find(nm) == events.end())
        {
            std::string msg = nm + " event not found.";
            std::cerr<<"Error: "<<msg<<std::endl;
            throw msg;
        }
    }
    std::map<std::string, StopWatch> timers;
    std::map<std::string, EventHistory> events;
    std::map<std::string, unsigned long> counters;

    bool statsComputed;
    std::map<std::string, statValue<float> > timerStats;
    std::map<std::string, statValue<unsigned long> > counterStats;
    std::vector<std::map<std::string,EventHistory>> eventStats;

    std::ofstream outputStream;
};

extern vtkh::StatisticsDB stats;
#ifdef ENABLE_STATISTICS
#define ADD_COUNTER(nm) stats.AddCounter(nm)
#define COUNTER_INC(nm, val) stats.Increment(nm, val)

#define ADD_TIMER(nm) stats.AddTimer(nm)
#define TIMER_START(nm) stats.Start(nm)
#define TIMER_STOP(nm) stats.Stop(nm)
#define DUMP_STATS(fname) stats.DumpStats(fname)
#else
#define ADD_COUNTER(nm)
#define COUNTER_INC(nm, val)

#define ADD_TIMER(nm)
#define TIMER_START(nm)
#define TIMER_STOP(nm)
#define DUMP_STATS(fname)
#endif
}

#endif //__STATISTICS_DB_H

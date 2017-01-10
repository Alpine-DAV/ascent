/******************************************************************************
 *
 * Header file for doing timing
 *
 *****************************************************************************/

#ifndef KRIPKE_TIMING_H__
#define KRIPKE_TIMING_H__

#include<Kripke.h>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#ifdef KRIPKE_USE_PAPI
#include<papi.h>
#endif

inline double getTime(void){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}


struct Timer {
  Timer() :
    started(false),
    start_time(0.0),
    total_time(0.0),
    count(0)
  {}

  bool started;
  double start_time;
  double total_time;
  size_t count;
#ifdef KRIPKE_USE_PAPI
  std::vector<long long> papi_start_values;
  std::vector<size_t> papi_total;
#endif
};

class Timing {
  public:
    ~Timing();

    void start(std::string const &name);
    void stop(std::string const &name);

    void stopAll(void);
    void clear(void);

    void print(void) const;
    void printTabular(bool print_header,
        std::vector<std::string> const &headers,
        std::vector<std::string> const &values,
        FILE *fp = stdout) const;

    double getTotal(std::string const &name) const;

    void setPapiEvents(std::vector<std::string> names);

  private:
    typedef std::map<std::string, Timer> TimerMap;
    TimerMap timers;
#ifdef KRIPKE_USE_PAPI
  std::vector<std::string> papi_names;
  std::vector<int> papi_event;
  int papi_set;
#endif
};


#include<stdio.h>

// Aides timing a block of code, with automatic timer stopping
class BlockTimer {
  public:
  inline BlockTimer(Timing &timer_obj, std::string const &timer_name) :
      timer(timer_obj),
      name(timer_name)
  {
      timer.start(name);
  }
  inline ~BlockTimer(){
    timer.stop(name);
  }

  private:
      Timing &timer;
      std::string name;
};

#define BLOCK_TIMER(TIMER, NAME) BlockTimer BLK_TIMER_##NAME(TIMER, #NAME);


#endif

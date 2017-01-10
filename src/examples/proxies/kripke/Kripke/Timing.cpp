/******************************************************************************
 *
 * Routines for doing timing.
 *
 *****************************************************************************/

#include <Kripke/Timing.h>

#include<Kripke.h>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <mpi.h>

#ifdef KRIPKE_USE_BGPM
extern "C" void HPM_Start(char const *);
extern "C" void HPM_Stop(char const *);
#endif


#ifdef KRIPKE_USE_PAPI
#include <papi.h>
#endif


Timing::~Timing(){
#ifdef KRIPKE_USE_PAPI
long long tmp[16];
PAPI_stop(papi_set, tmp);
#endif

}

void Timing::start(std::string const &name){
  // get or create timer
  Timer &timer = timers[name];

  if(!timer.started){
    timer.started = true;
    timer.start_time = MPI_Wtime();

#ifdef KRIPKE_USE_PAPI
    int num_papi = papi_event.size();
    if(num_papi > 0){
      if(timer.papi_total.size() == 0){
        timer.papi_start_values.resize(num_papi, 0);
        timer.papi_total.resize(num_papi, 0);
      }

      /*
      // start timers
      PAPI_start_counters(&papi_event[0], num_papi);

      // clear timers
      long long tmp[16];
      PAPI_read_counters(tmp, num_papi);
      */

      // read initial values
      PAPI_read(papi_set, &timer.papi_start_values[0]);

    }
#endif

#ifdef KRIPKE_USE_BGPM
    HPM_Start(name.c_str());
#endif
  }
}

void Timing::stop(std::string const &name){
  // get or create timer
  Timer &timer = timers[name];

#ifdef KRIPKE_USE_BGPM
    HPM_Stop(name.c_str());
#endif

  if(timer.started){
#ifdef KRIPKE_USE_PAPI
    int num_papi = papi_event.size();
    if(num_papi > 0){
      // read timers
      long long tmp[16];
      //PAPI_stop_counters(tmp, num_papi);
      PAPI_read(papi_set, tmp);

      // accumulate to all started timers (since this clears the PAPI values)
      for(int i = 0;i < num_papi;++ i){
        timer.papi_total[i] += tmp[i] - timer.papi_start_values[i];
      }

    }
#endif

    // Stop the timer
    timer.started = false;
    timer.total_time += MPI_Wtime() - timer.start_time;
    timer.count ++;

  }
}

void Timing::stopAll(void){
  for(TimerMap::iterator i = timers.begin();i != timers.end();++ i){
    stop((*i).first);
  }
}

void Timing::clear(void){
  timers.clear();
}

void Timing::print(void) const {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank != 0){
    return;
  }

  // build a sorted list of names
  std::vector<std::string> names;
  for(TimerMap::const_iterator i = timers.begin();i != timers.end();++ i){
    names.push_back((*i).first);

  }
  std::sort(names.begin(), names.end());

  std::vector<Timer const *> ord_timers;
  for(int i = 0;i < names.size();++ i){
    std::string &name = names[i];
    TimerMap::const_iterator iter = timers.find(name);
    ord_timers.push_back(&(*iter).second);
  }

  // Display column names
  printf("Timers:\n");
  printf("  %-16s  %12s  %12s", "Timer", "Count", "Seconds");
#ifdef KRIPKE_USE_PAPI
  int num_papi = papi_names.size();
  for(int i = 0;i < num_papi;++i){
    printf("  %16s", papi_names[i].c_str());
  }
#endif
  printf("\n");

  // Dislpay timer results
  for(int i = 0;i < names.size();++ i){
    printf("  %-16s  %12d  %12.5lf", names[i].c_str(), (int)ord_timers[i]->count, ord_timers[i]->total_time);
#ifdef KRIPKE_USE_PAPI
    for(int p = 0;p < num_papi;++ p){
      printf("  %16ld", (long)ord_timers[i]->papi_total[p]);
    }
#endif
    printf("\n");
  }
}


namespace {
  void printTabVector(FILE *fp, std::vector<std::string> const &values){
    int len = values.size();
    for(int i = 0;i < len;++ i){
      if(i > 0){
        fprintf(fp, "\t");
      }
      fprintf(fp, "%s", values[i].c_str());
    }
    fprintf(fp, "\n");
  }

  template<typename T>
  std::string toString(T const &val){
    std::stringstream ss;
    ss << val;
    return ss.str();
  }
}

void Timing::printTabular(bool print_header,
    std::vector<std::string> const &headers0,
    std::vector<std::string> const &values0,
    FILE *fp) const {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank != 0){
    return;
  }

  // build a sorted list of names
  std::vector<std::string> names;
  for(TimerMap::const_iterator i = timers.begin();i != timers.end();++ i){
    names.push_back((*i).first);

  }
  std::sort(names.begin(), names.end());

  std::vector<Timer const *> ord_timers;
  for(int i = 0;i < names.size();++ i){
    std::string &name = names[i];
    TimerMap::const_iterator iter = timers.find(name);
    ord_timers.push_back(&(*iter).second);
  }

  // Output a header
  std::vector<std::string> values = headers0;
  values.push_back("timer");
  values.push_back("count");
  values.push_back("seconds");
#ifdef KRIPKE_USE_PAPI
  int num_papi = papi_names.size();
  for(int i = 0;i < papi_names.size();++i){
    values.push_back(papi_names[i]);
  }
#endif
  if(print_header){
    printTabVector(fp, values);
  }

  // For each timer, print the values
  for(int i = 0;i < names.size();++ i){
    values = values0;
    values.push_back(names[i]);
    values.push_back(toString(ord_timers[i]->count));
    values.push_back(toString(ord_timers[i]->total_time));
#ifdef KRIPKE_USE_PAPI
    for(int p = 0;p < num_papi;++ p){
      values.push_back(toString(ord_timers[i]->papi_total[p]));
     }
#endif

    printTabVector(fp, values);
  }
}

double Timing::getTotal(std::string const &name) const{
  TimerMap::const_iterator i = timers.find(name);
  if(i == timers.end()){
    return 0.0;
  }
  return (*i).second.total_time;
}



void Timing::setPapiEvents(std::vector<std::string> names){
#ifdef KRIPKE_USE_PAPI


  static bool papi_initialized = false;
  if(!papi_initialized){
    //printf("PAPI INIT\n");
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    papi_initialized = true;

    if(retval != PAPI_VER_CURRENT){
      fprintf(stderr, "ERROR INITIALIZING PAPI\n");
      exit(1);
    }
  }

  //printf("PAPI VERSION=%x\n",
  //    PAPI_VERSION);

  papi_set = PAPI_NULL;
  PAPI_create_eventset(&papi_set);


  for(int i = 0;i < names.size();++ i){
    // Convert text string to PAPI id
    int event_code;
    PAPI_event_name_to_code(
        const_cast<char*>(names[i].c_str()),
        &event_code);

    // TODO: error checking?

    // Add to our list of PAPI events
    papi_names.push_back(names[i]);
    papi_event.push_back(event_code);

    int retval = PAPI_add_event(papi_set, event_code);
    if(retval != PAPI_OK){
      fprintf(stderr, "ERROR ADDING %s, retval=%d, ID=0x%-10x\n", names[i].c_str(), retval, event_code);
    }

    //printf("EVT=%s, ID=0x%-10x\n", names[i].c_str(), event_code);
  }
  PAPI_start(papi_set);
#else
  if(names.size() > 0){
    fprintf(stderr, "WARNING: PAPI NOT ENABLED, IGNORING PAPI EVENTS\n");
  }
#endif
}

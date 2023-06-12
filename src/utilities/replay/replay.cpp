//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: relay.cpp
///
//-----------------------------------------------------------------------------
#include <ascent.hpp>
#include <flow_timer.hpp>
#include <ascent_hola.hpp>

#include <fstream>
#include <vector>
#include <algorithm>
#ifdef REPLAY_MPI
#include <mpi.h>
#endif

void usage()
{
  std::cout<<"replay usage:\n";
  std::cout<<"replay is a utility for 'replaying' the data files saved during the course ";
  std::cout<<"of a simulation. Domain overloading is supported, so you can load x domains ";
  std::cout<<"with y mpi ranks where x >= y\n\n.";
  std::cout<<"======================== Options  =========================\n";
  std::cout<<"  --root    : the root file for a blueprint hdf5 set of files.\n";
  std::cout<<"  --cycles  : a text file containing a list of root files, one per line.\n";
  std::cout<<"              Each file will be loaded and sent to Ascent in order.\n";
  std::cout<<"  --actions : a json file containing ascent actions. Default value\n";
  std::cout<<"              is 'ascent_actions.json'.\n\n";
  std::cout<<"======================== Examples =========================\n";
  std::cout<<"./relay_ser --root=clover.cycle_000060.root\n";
  std::cout<<"./relay_ser --root=clover.cycle_000060.root --actions=my_actions.json\n";
  std::cout<<"srun -n 4 relay_mpi --cycles=cycles_file\n";
  std::cout<<"\n\n";
}

struct Options
{
  std::string m_actions_file = "ascent_actions.json";
  std::string m_root_file;
  std::string m_cycles_file;

  void parse(int argc, char** argv)
  {
    for(int i = 1; i < argc; ++i)
    {
      if(contains(argv[i], "--root="))
      {
        m_root_file = get_arg(argv[i]);
      }
      else if(contains(argv[i], "--cycles="))
      {
        m_cycles_file = get_arg(argv[i]);
      }
      else if(contains(argv[i], "--actions="))
      {
        m_actions_file = get_arg(argv[i]);
      }
      else
      {
        bad_arg(argv[i]);
      }
    }
    if(m_root_file == "" && m_cycles_file == "")
    {
      std::cerr<<"You must specify a '--root' or '--cycles' files. Bailing...\n";
      usage();
      exit(1);
    }
    if(m_root_file != "" && m_cycles_file != "")
    {
      std::cerr<<"You must specify either '--root' or '--cycles' files but not both. Bailing...\n";
      usage();
      exit(1);
    }
  }

std::vector<std::string> &split(const std::string &s,
                                char delim,
                                std::vector<std::string> &elems)
{
  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delim))
  {
   elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}
  std::string get_arg(const char *arg)
  {
    std::vector<std::string> parse;
    std::string s_arg(arg);
    std::string res;

    parse = split(s_arg, '=');

    if(parse.size() != 2)
    {
      bad_arg(arg);
    }
    else
    {
      res = parse[1];
    }
    return res;
  }

bool contains(const std::string haystack, std::string needle)
{
  std::size_t found = haystack.find(needle);
  return (found != std::string::npos);
}

  void bad_arg(std::string bad_arg)
  {
    std::cerr<<"Invalid argument \""<<bad_arg<<"\"\n";
    usage();
    exit(0);
  }
};

void trim(std::string &s)
{
     s.erase(s.begin(),
             std::find_if_not(s.begin(),
                              s.end(),
                              [](char c){ return std::isspace(c); }));
     s.erase(std::find_if_not(s.rbegin(), 
                              s.rend(),
                              [](char c){ return std::isspace(c); }).base(),
             s.end());
}

int main (int argc, char *argv[])
{
  Options options;
  options.parse(argc, argv);

  std::vector<std::string> time_steps;

  if(options.m_root_file != "")
  {
    time_steps.push_back(options.m_root_file);
  }
  else
  {

    std::ifstream in_file(options.m_cycles_file);
    std::string line;
    while(!getline(in_file, line).eof())
    {
      // only add non-empty entries
      trim(line);
      if(!line.empty())
      {
          time_steps.push_back(line);
      }
    }
  }

  int comm_size = 1;
  int rank = 0;

#ifdef REPLAY_MPI
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  conduit::Node replay_data, replay_opts;
#ifdef REPLAY_MPI
  replay_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
  //replay_data.print();
  conduit::Node ascent_opts;
  ascent_opts["actions_file"] = options.m_actions_file;
  ascent_opts["ascent_info"] = "verbose";
#ifdef REPLAY_MPI
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif

  //
  // Blank actions to be populated from actions file
  //
  conduit::Node actions;

  ascent::Ascent ascent;
  ascent.open(ascent_opts);

  for(int i = 0; i < time_steps.size(); ++i)
  {
    replay_opts["root_file"] = time_steps[i];
    if(rank == 0)
    {
      std::cout<<"Root file "<<time_steps[i]<<"\n";
    }
    flow::Timer load;
    ascent::hola("relay/blueprint/mesh", replay_opts, replay_data);
#ifdef REPLAY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    float load_time = load.elapsed();
      
    flow::Timer publish;
    ascent.publish(replay_data);
#ifdef REPLAY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    float publish_time = publish.elapsed();

    flow::Timer execute;
    ascent.execute(actions);
#ifdef REPLAY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    float execute_time = execute.elapsed();
    if(rank == 0)
    {
      std::cout<<" Load -----: "<<load_time<<"\n";
      std::cout<<" Publish --: "<<publish_time<<"\n";
      std::cout<<" Execute --: "<<execute_time<<"\n";
    }
  }

  ascent.close();

#ifdef REPLAY_MPI
  MPI_Finalize();
#endif
  return 0;
}

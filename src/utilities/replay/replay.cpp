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
#include <conduit_relay_io_blueprint.hpp>

#include <fstream>
#include <vector>
#include <algorithm>

#if defined(ASCENT_REPLAY_MPI)
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#endif

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>

#include <fstream>

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
  std::cout<<"  --actions : a yaml file containing ascent actions. Default value\n";
  std::cout<<"              is 'ascent_actions.yaml'.\n\n";
  std::cout<<"======================== Examples =========================\n";
  std::cout<<"./ascent_replay --root=clover.cycle_000060.root\n";
  std::cout<<"./ascent_replay --root=clover.cycle_000060.root --actions=my_actions.yaml\n";
  std::cout<<"srun -n 4 ascent_replay_mpi --cycles=cycles_file\n";
  std::cout<<"\n\n";
}

struct Options
{
  std::string m_actions_file = "ascent_actions.yaml";
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

void load_actions(const std::string &file_name, int mpi_comm_id, conduit::Node &actions)
{
    int comm_size = 1;
    int rank = 0;
#ifdef ASCENT_REPLAY_MPI
    if(mpi_comm_id == -1)
    {
      // do nothing, an error will be thrown later
      // so we can respect the exception handling
      return;
    }
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    int has_file = 0;
    if(rank == 0 && conduit::utils::is_file(file_name))
    {
      has_file = 1;
    }
#ifdef ASCENT_REPLAY_MPI
    MPI_Bcast(&has_file, 1, MPI_INT, 0, mpi_comm);
#endif
    if(has_file == 0)
    {
      return;
    }

    int actions_file_valid = 0;
    std::string emsg = "";

    if(rank == 0)
    {
      std::string curr,next;

      std::string protocol = "json";
      // if file ends with yaml, use yaml as proto
      conduit::utils::rsplit_string(file_name,
                                    ".",
                                    curr,
                                    next);

      if(curr == "yaml")
      {
        protocol = "yaml";
      }

      try
      {
        conduit::Node file_node;
        file_node.load(file_name, protocol);

        actions = file_node;
 
        actions_file_valid = 1;
      }
      catch(conduit::Error &e)
      {
        // failed to open or parse the actions file
        actions_file_valid = 0;
        emsg = e.message();
      }
    }

#ifdef ASCENT_REPLAY_MPI
    // make sure all ranks error if the parsing on rank 0 failed.
    MPI_Bcast(&actions_file_valid, 1, MPI_INT, 0, mpi_comm);
#endif

    if(actions_file_valid == 0)
    {
        // show on rank 0 and exit
        if(rank == 0)
        {
          std::cout << "Failed to load actions file: "
                    << file_name
                    << "\n" << emsg
                    << std::endl;
        }
        exit(-1);
    }
#ifdef ASCENT_REPLAY_MPI
    conduit::relay::mpi::broadcast_using_schema(actions, 0, mpi_comm);
#endif
}

//---------------------------------------------------------------------------//
int
main(int argc, char *argv[])
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

#if defined(ASCENT_REPLAY_MPI)
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  conduit::Node replay_data;
  //replay_data.print();
  conduit::Node ascent_opts;
  ascent_opts["ascent_info"] = "verbose";
#if defined(ASCENT_REPLAY_MPI)
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif

  //
  // Populate actions with actions file
  //
  int mpi_comm = ascent_opts.has_child("mpi_comm") ? ascent_opts["mpi_comm"].to_int() : -1;
  conduit::Node actions;
  load_actions(options.m_actions_file, mpi_comm, actions);

  ascent::Ascent ascent;
  ascent.open(ascent_opts);

  for(int i = 0; i < time_steps.size(); ++i)
  {
    if(rank == 0)
    {
      std::cout<< "[" << i << "]: Root file "<<time_steps[i]<<"\n";
    }
    flow::Timer load;

#if defined(ASCENT_REPLAY_MPI)
    MPI_Comm comm  = MPI_Comm_f2c(ascent_opts["mpi_comm"].to_int());
    conduit::relay::mpi::io::blueprint::load_mesh(time_steps[i],replay_data,comm);
#else
    conduit::relay::io::blueprint::load_mesh(time_steps[i],replay_data);
#endif

#if defined(ASCENT_REPLAY_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    float load_time = load.elapsed();

    flow::Timer publish;
    ascent.publish(replay_data);
#if defined(ASCENT_REPLAY_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    float publish_time = publish.elapsed();

    flow::Timer execute;
    ascent.execute(actions);
#if defined(ASCENT_REPLAY_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    float execute_time = execute.elapsed();
    if(rank == 0)
    {
      std::cout<< "[" << i << "]: Load -----: "<<load_time<<"\n";
      std::cout<< "[" << i << "]: Publish --: "<<publish_time<<"\n";
      std::cout<< "[" << i << "]: Execute --: "<<execute_time<<"\n";
    }
  }

  ascent.close();

#if defined(ASCENT_REPLAY_MPI)
  MPI_Finalize();
#endif
  return 0;
}

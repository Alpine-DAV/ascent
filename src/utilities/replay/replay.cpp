//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://ascent.readthedocs.io/.
// 
// Please also read ascent/LICENSE
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
/// file: relay.cpp
///
//-----------------------------------------------------------------------------
#include <ascent.hpp>
#include <ascent_hola.hpp>

#include <vector>
#ifdef REPLAY_MPI
#include <mpi.h>
#endif

struct Options
{
  std::string m_actions_file = "ascent_actions.json";
  std::string m_root_file;

  void parse(int argc, char** argv)
  {
    for(int i = 1; i < argc; ++i)
    {
      if(contains(argv[i], "--root="))
      {
        m_root_file = get_arg(argv[i]); 
      }
      else if(contains(argv[i], "--actions="))
      {
        m_actions_file = get_arg(argv[i]); 
      }
      else
      {
        usage(argv[i]);
      }
    }
    if(m_root_file == "")
    {
      std::cerr<<"Required option '--root' not specified. Bailing...\n";
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
      usage(arg);
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

  void usage(std::string bad_arg)
  {
    std::cerr<<"Invalid argument \""<<bad_arg<<"\"\n";
    std::cout<<"Replay usage: "
             <<"       --root    : blueprint rootfile\n"
             <<"       --actions : ascent_actions.json file\n";
    exit(0);
  }
};

int main (int argc, char *argv[])
{
  Options options;
  options.parse(argc, argv);

  int comm_size = 1;
  int rank = 0;

#ifdef REPLAY_MPI
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  conduit::Node replay_data, replay_opts;
  replay_opts["root_file"] = options.m_root_file;
  ascent::hola("relay/blueprint/mesh", replay_opts, replay_data);
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
  ascent.publish(replay_data);
  ascent.execute(actions);
  ascent.close();

  return 0;
}

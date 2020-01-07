//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
/// file: json2yaml.cpp
///
//-----------------------------------------------------------------------------
#include <conduit.hpp>

void usage()
{
  std::cout<<"usage   : json2yaml --input=input_file [--output=output.yaml]\n";
  std::cout<<"Examples:\n";
  std::cout<<"  ./json2yaml --input=ascent_actions.yaml\n";
  std::cout<<"  ./json2yaml --input=ascent_actions.json --output=free_bananas.yaml\n";

  std::cout<<"\n\n";
}

struct Options
{
  std::string m_output_name = "ascent_actions.yaml";
  std::string m_input_name = "";

  void parse(int argc, char** argv)
  {
    for(int i = 1; i < argc; ++i)
    {
      if(contains(argv[i], "--input="))
      {
        m_input_name = get_arg(argv[i]);
      }
      else if(contains(argv[i], "--output="))
      {
        m_output_name = get_arg(argv[i]);
      }
      else
      {
        bad_arg(argv[i]);
      }
    }
    if(m_input_name == "")
    {
      std::cerr<<"You must specify '--input'. Bailing...\n";
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

int main (int argc, char *argv[])

{
  Options options;
  options.parse(argc, argv);

  conduit::Node input;
  input.load(options.m_input_name,"json");
  input.save(options.m_output_name, "yaml");

  return 0;
}

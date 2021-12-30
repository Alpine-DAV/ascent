//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <utils/vtk_dataset_reader.hpp>
#include <vtkm/io/VTKDataSetReader.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

namespace rover {

static
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
      std::string item;
        while (std::getline(ss, item, delim))
            {
                   elems.push_back(item);
                     }
          return elems;
           }
 static
std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
      split(s, delim, elems);
        return elems;
}
static
bool contains(const std::string haystack, std::string needle)
{
    std::size_t found = haystack.find(needle);
      return (found != std::string::npos);
}

VTKReader::VTKReader()
{
}

void
VTKReader::read_file(const std::string &file_name)
{
  vtkm::io::VTKDataSetReader reader(file_name.c_str());
  m_dataset= reader.ReadDataSet();

}

vtkmDataSet
VTKReader::get_data_set()
{
  return m_dataset;
}

MultiDomainVTKReader::MultiDomainVTKReader()
{
}

void
MultiDomainVTKReader::read_file(const std::string &directory, const std::string &file_name)
{
  //
  //  Get the list of file names from the main file
  //
  std::string full_name = directory + file_name;
  std::ifstream header_file(full_name.c_str());
  std::string line;
  if(header_file.is_open())
  {
     getline(header_file, line);
     //std::cout<<"Line: "<<line<<"\n";
     std::vector<std::string> tokens = split(line,' ');
     if(tokens.size() != 2)
     {
       std::cout<<"Error reading number of files\n";
       return;
     }
     const int number_of_domains = atoi(tokens.at(1).c_str());
     std::vector<std::string> file_names;
     for(int i = 0; i < number_of_domains; ++i)
     {
        getline(header_file, line);
        full_name = directory + line;
        file_names.push_back(full_name);
        //std::cout<<"Reading "<<full_name<<"\n";
     }

     int begining_domain = 0;
     int end_domain = number_of_domains - 1;
#ifdef ROVER_PARALLEL
     //
     // figure out which data sets to read
     //
     int rank;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     int num_ranks;
     MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
     if(rank == 0)
     {
        std::cout<<"Num ranks "<<num_ranks<<"\n";
     }
     if(num_ranks < number_of_domains)
     {
       int domains_per = number_of_domains / num_ranks;
       begining_domain = rank * domains_per;
       end_domain = (rank + 1) * domains_per - 1;
       if(rank == num_ranks - 1)
       {
         end_domain = number_of_domains - 1;
       }
     }
     else if(num_ranks == number_of_domains)
     {
        begining_domain = rank;
        end_domain = rank;
     }
     else
     {
       std::cout<<"CANNOT CURRENTLY handle empty data sets\n";
     }

#endif
     for(int i = begining_domain; i <= end_domain; ++i)
     {
        //std::cout<<"Reading "<<number_of_domains<<" files\n";
        vtkm::io::VTKDataSetReader reader(file_names[i].c_str());
        m_datasets.push_back(reader.ReadDataSet());
        //m_datasets[i].PrintSummary(std::cout);
     }

  }
  else
  {
    std::cout<<"Failed to open multi domain header file "<<full_name<<"\n";
    return;
  }

  header_file.close();
}

std::vector<vtkmDataSet>
MultiDomainVTKReader::get_data_sets()
{
  return m_datasets;
}

} // namespace rover

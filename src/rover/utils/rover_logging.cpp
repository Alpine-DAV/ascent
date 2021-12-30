//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <utils/rover_logging.hpp>
#include <rover_exceptions.hpp>
#include <iostream>
#include <sstream>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

namespace rover {

Logger* Logger::m_instance  = NULL;

Logger::Logger()
{
  std::stringstream log_name;
  log_name<<"rover";
#ifdef ROVER_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  log_name<<"_"<<rank;
#endif
  log_name<<".log";
  m_stream.open(log_name.str().c_str(), std::ofstream::out);
  if(!m_stream.is_open())
    std::cout<<"Warning: could not open the rover log file\n";
}

Logger::~Logger()
{
  if(m_stream.is_open())
    m_stream.close();
}

Logger* Logger::get_instance()
{
  if(m_instance == NULL)
    m_instance =  new Logger();
  return m_instance;
}

std::ofstream& Logger::get_stream()
{
  return m_stream;
}

void
Logger::write(const int level, const std::string &message, const char *file, int line)
{
  if(level == 0)
    m_stream<<"<Info> \n";
  else if (level == 1)
    m_stream<<"<Warning> \n";
  else if (level == 2)
    m_stream<<"<Error> \n";
  m_stream<<"  message: "<<message<<" \n  file: "<<file<<" \n  line: "<<line<<"\n";
}

// ---------------------------------------------------------------------------------------

DataLogger* DataLogger::Instance  = NULL;

DataLogger::DataLogger()
{
}

DataLogger::~DataLogger()
{
  Stream.str("");
}

DataLogger*
DataLogger::GetInstance()
{
  if(DataLogger::Instance == NULL)
  {
    DataLogger::Instance =  new DataLogger();
  }
  return DataLogger::Instance;
}

std::stringstream&
DataLogger::GetStream()
{
  return Stream;
}

void
DataLogger::WriteLog()
{
  std::stringstream log_name;
  std::ofstream stream;
  log_name<<"rover_data";
#ifdef ROVER_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  log_name<<"_"<<rank;
#endif
  log_name<<".log";
  stream.open(log_name.str().c_str(), std::ofstream::out);
  if(!stream.is_open())
  {
    std::cerr<<"Warning: could not open the rover data log file\n";
    return;
  }
  stream<<Stream.str();
  stream.close();
}

void
DataLogger::OpenLogEntry(const std::string &entryName)
{
    Stream<<entryName<<" "<<"<\n";
    Entries.push(entryName);
}
void
DataLogger::CloseLogEntry(const double &entryTime)
{
  this->Stream<<"total_time "<<entryTime<<"\n";
  this->Stream<<this->Entries.top()<<" >\n";
  Entries.pop();
}

} // namespace rover


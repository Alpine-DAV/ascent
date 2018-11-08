//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-749865
// 
// All rights reserved.
// 
// This file is part of Rover. 
// 
// Please also read rover/LICENSE
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


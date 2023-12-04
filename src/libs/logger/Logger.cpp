#include "Logger.hpp"


#include <iomanip>
#include <cstdlib>

namespace logging
{
std::map<std::string, Logger*> Logger::Loggers;

Logger::Logger(const std::string& name)
{
  std::stringstream logName;
  logName<<name;
  //TODO: Handle mpi calls. Probably just do it here. 
#ifdef MPI_ENABLED
//  logName<<"."<<vtkh::GetMPIRank();
#endif
  logName<<".log";

  m_stream.open(logName.str().c_str(), std::ofstream::out);
  if(!m_stream.is_open())
    std::cout<<"Warning: could not open the log file\n";
}

Logger::~Logger()
{
  if (m_stream.is_open())
    m_stream.close();
}

Logger* Logger::GetInstance(const std::string& name)
{
  if (Loggers.find(name) == Loggers.end())
    Loggers[name] = new Logger(name);

  return Loggers[name];
}

void
Logger::Write(const int level, const std::string &message, const char *file, int line)
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

DataLogger DataLogger::m_instance;

DataLogger::DataLogger()
  : m_at_block_start(true),
    m_rank(0)
{
  m_blocks.push(Block(0));
  m_key_counters.push(std::map<std::string,int>());
}

DataLogger::~DataLogger()
{
#ifdef ENABLE_LOGGING
  WriteLog();
#endif
  m_stream.str("");
}

DataLogger*
DataLogger::GetInstance()
{
  return &DataLogger::m_instance;
}

DataLogger::Block&
DataLogger::CurrentBlock()
{
  return m_blocks.top();
}

void
DataLogger::SetRank(int rank)
{
  m_rank = rank;
}

void
DataLogger::WriteIndent()
{
  int indent = this->CurrentBlock().m_indent;
  bool listStart = this->CurrentBlock().m_at_list_item_start;

  if (listStart)
  {
    --indent;
  }

  for (int i = 0; i < indent; ++i)
  {
    m_stream << "  ";
  }

  if (listStart)
  {
    m_stream << "- ";
    CurrentBlock().m_at_list_item_start = false;
  }
}

void
DataLogger::WriteLog()
{
  std::stringstream log_name;

  std::string log_prefix = "logger_data";
  if(const char* log_p = std::getenv("LOG_FILE_PREFIX"))
  {
    log_prefix = std::string(log_p);
  }

  log_name<<log_prefix<<"_";
  log_name<<std::setfill('0')<<std::setw(6)<<m_rank;
  log_name<<".yaml";

  std::ofstream stream;
  stream.open(log_name.str().c_str(), std::ofstream::app);

  if(!stream.is_open())
  {
    std::cerr<<"Warning: could not open the logger data file\n";
    return;
  }
  stream<<m_stream.str();
  stream.close();
}

void
DataLogger::OpenLogEntry(const std::string &entryName)
{
    WriteIndent();
    // ensure that we have unique keys for valid yaml
    int key_count = m_key_counters.top()[entryName]++;

    if(key_count != 0)
    {
      m_stream<<entryName<<"_"<<key_count<<":"<<"\n";
    }
    else
    {
      m_stream<<entryName<<":"<<"\n";
    }

    int indent = this->CurrentBlock().m_indent;
    m_blocks.push(Block(indent+1));
    m_key_counters.push(std::map<std::string,int>());

    Timer timer;
    m_timers.push(timer);
    m_at_block_start = true;

}
void
DataLogger::CloseLogEntry()
{
  WriteIndent();
  this->m_stream<<"time : "<<m_timers.top().elapsed()<<"\n";
  m_timers.pop();
  m_blocks.pop();
  m_key_counters.pop();
  m_at_block_start = false;
}

};

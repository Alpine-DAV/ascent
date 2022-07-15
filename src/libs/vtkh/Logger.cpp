#include <vtkh/vtkh.hpp>
#include <vtkh/Logger.hpp>

#include <iomanip>
#include <cstdlib>

namespace vtkh
{
std::map<std::string, Logger*> Logger::Loggers;

Logger::Logger(const std::string& name)
{
  std::stringstream logName;
  logName<<name;
#ifdef VTKH_PARALLEL
  logName<<"."<<vtkh::GetMPIRank();
#endif
  logName<<".log";

  Stream.open(logName.str().c_str(), std::ofstream::out);
  if(!Stream.is_open())
    std::cout<<"Warning: could not open the vtkh log file\n";
}

Logger::~Logger()
{
  if (Stream.is_open())
    Stream.close();
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
    Stream<<"<Info> \n";
  else if (level == 1)
    Stream<<"<Warning> \n";
  else if (level == 2)
    Stream<<"<Error> \n";
  Stream<<"  message: "<<message<<" \n  file: "<<file<<" \n  line: "<<line<<"\n";
}

// ---------------------------------------------------------------------------------------

DataLogger DataLogger::Instance;

DataLogger::DataLogger()
  : AtBlockStart(true),
    Rank(0)
{
  Blocks.push(Block(0));
  KeyCounters.push(std::map<std::string,int>());
}

DataLogger::~DataLogger()
{
#ifdef VTKH_ENABLE_LOGGING
  WriteLog();
#endif
  Stream.str("");
}

DataLogger*
DataLogger::GetInstance()
{
  return &DataLogger::Instance;
}

DataLogger::Block&
DataLogger::CurrentBlock()
{
  return Blocks.top();
}

void
DataLogger::SetRank(int rank)
{
  Rank = rank;
}

void
DataLogger::WriteIndent()
{
  int indent = this->CurrentBlock().Indent;
  bool listStart = this->CurrentBlock().AtListItemStart;

  if (listStart)
  {
    --indent;
  }

  for (int i = 0; i < indent; ++i)
  {
    Stream << "  ";
  }

  if (listStart)
  {
    Stream << "- ";
    CurrentBlock().AtListItemStart = false;
  }
}

void
DataLogger::WriteLog()
{
  std::stringstream log_name;

  std::string log_prefix = "vtkh_data";
  if(const char* log_p = std::getenv("VTKH_LOG_PREFIX"))
  {
    log_prefix = std::string(log_p);
  }

  log_name<<log_prefix<<"_";
  log_name<<std::setfill('0')<<std::setw(6)<<Rank;
  log_name<<".yaml";

  std::ofstream stream;
  stream.open(log_name.str().c_str(), std::ofstream::app);

  if(!stream.is_open())
  {
    std::cerr<<"Warning: could not open the vtkh data log file\n";
    return;
  }
  stream<<Stream.str();
  stream.close();
}

void
DataLogger::OpenLogEntry(const std::string &entryName)
{
    WriteIndent();
    // ensure that we have unique keys for valid yaml
    int key_count = KeyCounters.top()[entryName]++;

    if(key_count != 0)
    {
      Stream<<entryName<<"_"<<key_count<<":"<<"\n";
    }
    else
    {
      Stream<<entryName<<":"<<"\n";
    }

    int indent = this->CurrentBlock().Indent;
    Blocks.push(Block(indent+1));
    KeyCounters.push(std::map<std::string,int>());

    Timer timer;
    Timers.push(timer);
    AtBlockStart = true;

}
void
DataLogger::CloseLogEntry()
{
  WriteIndent();
  this->Stream<<"time : "<<Timers.top().elapsed()<<"\n";
  Timers.pop();
  Blocks.pop();
  KeyCounters.pop();
  AtBlockStart = false;
}
};

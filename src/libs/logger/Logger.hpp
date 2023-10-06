#ifndef LOGGER_HPP
#define LOGGER_HPP


#include <stack>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

#include "Timer.hpp"
//from vtkh logging
namespace logging
{

class Logger
{
public:
  static Logger *GetInstance(const std::string& name);

  ~Logger();
  void Write(const int level, const std::string &message, const char *file, int line);
  std::ofstream & GetStream() { return Stream; }

protected:
  Logger(const std::string& name);
  Logger(Logger const &);
  std::ofstream Stream;

  static std::map<std::string, Logger*> Loggers;
};

class DataLogger
{
public:
  struct Block
  {
    int Indent;
    bool IsList;
    bool AtListItemStart;

    Block(int indent)
      : Indent(indent), IsList(false), AtListItemStart(false)
    {  }
  };

  ~DataLogger();
  static DataLogger *GetInstance();
  void OpenLogEntry(const std::string &entryName);
  void CloseLogEntry();
  void SetRank(int rank);

  template<typename T>
  void AddLogData(const std::string key, const T &value)
  {
    WriteIndent();
    this->Stream << key << ": " << value <<"\n";
    AtBlockStart = false;
  }

  std::stringstream& GetStream() { return Stream; }
protected:
  DataLogger();
  DataLogger(DataLogger const &);

  void WriteLog();
  void WriteIndent();
  DataLogger::Block& CurrentBlock();
  std::stringstream Stream;
  static class DataLogger Instance;
  std::stack<Block> Blocks;
  std::stack<Timer> Timers;
  std::stack<std::map<std::string,int>> KeyCounters;
  bool AtBlockStart;
  int Rank;
};

#ifdef ENABLE_LOGGING
#define LOGGING_INFO(msg) logging::Logger::GetInstance("info")->GetStream()<<msg<<std::endl;
#define LOGGING_WARN(msg) logging::Logger::GetInstance("warning")->GetStream()<<msg<<std::endl;
#define LOGGING_ERROR(msg) logging::Logger::GetInstance("error")->GetStream()<<msg<<std::endl;
#define LOGGING_DATA_OPEN(key) logging::DataLogger::GetInstance()->OpenLogEntry(key);
#define LOGGING_DATA_CLOSE() logging::DataLogger::GetInstance()->CloseLogEntry();
#define LOGGING_DATA_ADD(key,value) logging::DataLogger::GetInstance()->AddLogData(key, value);

#else
#define LOGGING_INFO(msg)
#define LOGGING_WARN(msg)
#define LOGGING_ERROR(msg)
#define LOGGING_DATA_ADD(key,value)
#define LOGGING_DATA_OPEN(key)
#define LOGGING_DATA_CLOSE()
#endif


}; // namespace logging

#endif //LOGGER_HPP

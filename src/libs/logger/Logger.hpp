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
  std::ofstream & GetStream() { return m_stream; }

protected:
  Logger(const std::string& name);
  Logger(Logger const &);
  std::ofstream m_stream;

  static std::map<std::string, Logger*> Loggers;
};

class DataLogger
{
public:
  struct Block
  {
    int m_indent;
    bool m_is_list;
    bool m_at_list_item_start;

    Block(int indent)
      : m_indent(indent), m_is_list(false), m_at_list_item_start(false)
    {  }
  };

  ~DataLogger();
  static DataLogger *GetInstance();
  void OpenLogEntry(const std::string &entryName);
  void CloseLogEntry();
  void WriteLog();
  void SetRank(int rank);

  template<typename T>
  void AddLogData(const std::string key, const T &value)
  {
    WriteIndent();
    this->m_stream << key << ": " << value <<"\n";
    m_at_block_start = false;
  }

  std::stringstream& GetStream() { return m_stream; }
protected:
  DataLogger();
  DataLogger(DataLogger const &);

  void WriteIndent();
  DataLogger::Block& CurrentBlock();
  std::stringstream m_stream;
  static class DataLogger m_instance;
  std::stack<Block> m_blocks;
  std::stack<Timer> m_timers;
  std::stack<std::map<std::string,int>> m_key_counters;
  bool m_at_block_start;
  int m_rank;
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

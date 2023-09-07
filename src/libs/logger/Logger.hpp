#ifndef VTK_H_LOGGER_HPP
#define VTK_H_LOGGER_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/Timer.hpp>
#include <vtkh/utils/StreamUtil.hpp>

#include <stack>
#include <sstream>
//from rover logging
namespace vtkh
{

class VTKH_API Logger
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

class VTKH_API DataLogger
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

#ifdef VTKH_ENABLE_LOGGING
#define VTKH_INFO(msg) vtkh::Logger::GetInstance("info")->GetStream()<<msg<<std::endl;
#define VTKH_WARN(msg) vtkh::Logger::GetInstance("warning")->GetStream()<<msg<<std::endl;
#define VTKH_ERROR(msg) vtkh::Logger::GetInstance("error")->GetStream()<<msg<<std::endl;
#define VTKH_DATA_OPEN(key) vtkh::DataLogger::GetInstance()->OpenLogEntry(key);
#define VTKH_DATA_CLOSE() vtkh::DataLogger::GetInstance()->CloseLogEntry();
#define VTKH_DATA_ADD(key,value) vtkh::DataLogger::GetInstance()->AddLogData(key, value);

#else
#define VTKH_INFO(msg)
#define VTKH_WARN(msg)
#define VTKH_ERROR(msg)
#define VTKH_DATA_ADD(key,value)
#define VTKH_DATA_OPEN(key)
#define VTKH_DATA_CLOSE()
#endif


}; // namespace vtkh

#endif //VTK_H_LOGGER_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_loggin_h
#define rover_loggin_h

#include <fstream>
#include <stack>
#include <sstream>

namespace rover {

class Logger
{
public:
  ~Logger();
  static Logger *get_instance();
  void write(const int level, const std::string &message, const char *file, int line);
  std::ofstream & get_stream();
protected:
  Logger();
  Logger(Logger const &);
  std::ofstream m_stream;
  static class Logger* m_instance;
};

class DataLogger
{
public:
  ~DataLogger();
  static DataLogger *GetInstance();
  void OpenLogEntry(const std::string &entryName);
  void CloseLogEntry(const double &entryTime);

  template<typename T>
  void AddLogData(const std::string key, const T &value)
  {
    this->Stream<<key<<" "<<value<<"\n";
  }

  std::stringstream& GetStream();
  void WriteLog();
protected:
  DataLogger();
  DataLogger(DataLogger const &);
  std::stringstream Stream;
  static class DataLogger* Instance;
  std::stack<std::string> Entries;
};

#ifdef ROVER_ENABLE_LOGGING
#define ROVER_INFO(msg) rover::Logger::get_instance()->get_stream() <<"<Info>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define ROVER_WARN(msg) rover::Logger::get_instance()->get_stream() <<"<Warn>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define ROVER_ERROR(msg) rover::Logger::get_instance()->get_stream() <<"<Error>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;

#define ROVER_DATA_OPEN(name) rover::DataLogger::GetInstance()->OpenLogEntry(name);
#define ROVER_DATA_CLOSE(time) rover::DataLogger::GetInstance()->CloseLogEntry(time);
#define ROVER_DATA_ADD(key,value) rover::DataLogger::GetInstance()->AddLogData(key, value);

#else
#define ROVER_INFO(msg)
#define ROVER_WARN(msg)
#define ROVER_ERROR(msg)

#define ROVER_DATA_OPEN(name)
#define ROVER_DATA_CLOSE(name) (void)name;
#define ROVER_DATA_ADD(key,value)
#endif
} // namespace rover

#endif

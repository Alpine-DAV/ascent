#ifndef DRAY_DATA_LOGGER_HPP
#define DRAY_DATA_LOGGER_HPP

#include <dray/utils/yaml_writer.hpp>
#include <dray/utils/timer.hpp>

#include <fstream>
#include <map>
#include <stack>

namespace dray
{

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
  static DataLogger *get_instance();
  void open(const std::string &entryName);
  void close();

  template<typename T>
  void add_entry(const std::string key, const T &value)
  {
    write_indent();
    this->m_stream << key << ": " << value <<"\n";
    m_at_block_start = false;
  }

  void write_log();
  void set_rank(const int &rank);
  std::stringstream& get_stream() { return m_stream; }
protected:
  DataLogger();
  DataLogger(DataLogger const &);

  void write_indent();
  DataLogger::Block& current_block();
  std::stringstream m_stream;
  static class DataLogger m_instance;
  std::stack<Block> m_blocks;
  std::stack<Timer> m_timers;
  std::stack<std::map<std::string,int>> m_key_counters;
  bool m_at_block_start;
  int m_rank;
};

} // namspace dray

#ifdef DRAY_ENABLE_LOGGING
#define DRAY_INFO(msg) ::dray::Logger::get_instance()->get_stream() <<"<Info>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define DRAY_WARN(msg) ::dray::Logger::get_instance()->get_stream() <<"<Warn>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;

#define DRAY_LOG_OPEN(name) ::dray::DataLogger::get_instance()->open(name);
#define DRAY_LOG_CLOSE() ::dray::DataLogger::get_instance()->close();
#define DRAY_LOG_ENTRY(key,value) ::dray::DataLogger::get_instance()->add_entry(key,value);
#define DRAY_LOG_VALUE(value) ::dray::DataLogger::get_instance()->add_value(value);
#define DRAY_LOG_WRITE() ::dray::DataLogger::get_instance()->write_log();

#else
#define DRAY_INFO(msg)
#define DRAY_WARN(msg)

#define DRAY_LOG_OPEN(name)
#define DRAY_LOG_CLOSE()
#define DRAY_LOG_ENTRY(key,value)
#define DRAY_LOG_VALUE(value)
#define DRAY_LOG_WRITE()
#endif

#endif

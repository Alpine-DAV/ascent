#ifndef ASCENT_DATA_LOGGER_HPP
#define ASCENT_DATA_LOGGER_HPP

#include <ascent_exports.h>
#include <flow_timer.hpp>

#include <string>
#include <stack>
#include <map>
#include <sstream>

//from rover logging
namespace ascent
{

class ASCENT_API DataLogger
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
  static DataLogger *instance();
  void open_entry(const std::string &entryName);
  void close_entry();
  void rank(int rank);

  template<typename T>
  void add_data(const std::string key, const T &value)
  {
    write_indent();
    this->m_stream << key << ": " << value <<"\n";
    m_at_block_start = false;
  }

  std::stringstream& stream() { return m_stream; }
protected:
  DataLogger();
  DataLogger(DataLogger const &);

  void write_log();
  void write_indent();
  DataLogger::Block& current_block();
  std::stringstream m_stream;
  static class DataLogger m_instance;
  std::stack<Block> m_blocks;
  std::stack<flow::Timer> Timers;
  std::stack<std::map<std::string,int>> m_key_counters;
  bool m_at_block_start;
  int m_rank;
};

#define ASCENT_DATA_OPEN(key) vtkh::DataLogger::instance()->open_entry(key);
#define ASCENT_DATA_CLOSE() vtkh::DataLogger::instance()->close_entry();
#define ASCENT_DATA_ADD(key,value) vtkh::DataLogger::instance()->add_data(key, value);

}; // namespace ascent

#endif //ASCENT_DATA_LOGGER_HPP

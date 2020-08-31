#include "ascent_data_logger.hpp"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

namespace ascent
{

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
#ifdef ASCENT_LOGGING_ENABLED
  write_log();
#endif
  m_stream.str("");
}

DataLogger*
DataLogger::instance()
{
  return &DataLogger::m_instance;
}

DataLogger::Block&
DataLogger::current_block()
{
  return m_blocks.top();
}

void
DataLogger::rank(int rank)
{
  m_rank = rank;
}

void
DataLogger::write_indent()
{
#ifdef ASCENT_LOGGING_ENABLED
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
    current_block().m_at_list_item_start = false;
  }
#endif
}

void
DataLogger::write_log()
{
#ifdef ASCENT_LOGGING_ENABLED
  std::stringstream log_name;

  std::string log_prefix = "ascent_data";
  if(const char* log_p = std::getenv("ASCENT_LOG_PREFIX"))
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
    std::cerr<<"Warning: could not open the ascent data log file\n";
    return;
  }
  stream<<m_stream.str();
  stream.close();
#endif
}

void
DataLogger::open_entry(const std::string &entryName)
{
#ifdef ASCENT_LOGGING_ENABLED
    write_indent();
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

    int indent = this->current_block().m_indent;
    m_blocks.push(Block(indent+1));
    m_key_counters.push(std::map<std::string,int>());

    flow::Timer timer;
    Timers.push(timer);
    m_at_block_start = true;
#else
    (void) entryName;
#endif
}
void
DataLogger::close_entry()
{
#ifdef ASCENT_LOGGING_ENABLED
  write_indent();
  this->m_stream<<"time : "<<Timers.top().elapsed()<<"\n";
  Timers.pop();
  m_blocks.pop();
  m_key_counters.pop();
  m_at_block_start = false;
#endif
}

};// namespace ascent

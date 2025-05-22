//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_logging.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_logging.hpp"

// standard includes
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <string>

// thirdparty includes
#include <conduit.hpp>
#include <conduit_fmt/conduit_fmt.h>

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// main logger instance
//-----------------------------------------------------------------------------
Logger                        Logger::m_instance;

//-----------------------------------------------------------------------------
Logger::Scope::Scope(Logger &lgr, const std::string &name)
 : m_lgr(lgr),
   m_name(name)
{
    m_lgr.log_block_begin(m_name);
}

//-----------------------------------------------------------------------------
Logger::Scope::~Scope()
{
    m_lgr.log_block_end(m_name);
}

//-----------------------------------------------------------------------------
Logger::Logger()
 : m_log_open(false),
   m_indent_level(0),
   m_rank(-1),
   m_log_threshold(Logger::LOG_INFO_ID),
   m_echo_threshold(Logger::LOG_NONE_ID)
{
    m_key_counters.push(std::map<std::string,int>());
}

//-----------------------------------------------------------------------------
Logger::~Logger()
{
    close();
}

//-----------------------------------------------------------------------------
void
Logger::reset()
{
    m_indent_level = 0;
    m_log_threshold  = Logger::LOG_INFO_ID;
    m_echo_threshold = Logger::LOG_NONE_ID;
    // reset our stacks
    m_timers = std::stack<Timer>();
    m_key_counters = std::stack<std::map<std::string,int>>();
    m_key_counters.push(std::map<std::string,int>());
}

//-----------------------------------------------------------------------------
void
Logger::open(const std::string &ofpattern)
{
    if(is_log_open())
    {
        std::string emsg = conduit_fmt::format("[FATAL_ERROR] failed to open log with pattern: {}, logger already has {} open.",
                                               ofpattern, m_log_fname);
        throw conduit::Error(emsg,
                             __FILE__,
                             __LINE__);
    }

    // multi node case, assumes file pattern includes "rank"
    if(rank() > -1)
    {
        m_log_fname = conduit_fmt::format(ofpattern,
                                          conduit_fmt::arg("rank",rank()));
    }
    else
    {
        m_log_fname = ofpattern;
    }

    // open append (consequtive ascent open use cases)
    m_log_stream.open(m_log_fname.c_str(),std::ios_base::app);

    if(!m_log_stream.is_open())
    {
        throw conduit::Error(conduit_fmt::format("[FATAL_ERROR] Failed to open log file: {} ", m_log_fname),
                             __FILE__,
                             __LINE__);
    }
    m_log_open = true;
    log_message(Logger::LOG_DEBUG_ID,conduit_fmt::format("opened log file: {}", m_log_fname));
}

//-----------------------------------------------------------------------------
bool
Logger::is_log_open()
{
    return m_log_open;
}

//-----------------------------------------------------------------------------
void
Logger::close()
{
    if(is_log_open())
    {
        log_message(Logger::LOG_DEBUG_ID,conduit_fmt::format("closing log file: {}", m_log_fname));
        m_log_stream.close();
        m_log_open = false;
        m_log_fname = "";
    }

    reset();
}

//-----------------------------------------------------------------------------
void
Logger::flush()
{
    if(is_log_open())
    {
        log_stream() << std::flush;
    }
}

//-----------------------------------------------------------------------------
void
Logger::log_block_begin(const std::string &name)
{
    // skip if log is not open
    if(!is_log_open())
    {
        return;
    }

    // make sure we have a unique key name
    int key_count = m_key_counters.top()[name]++;
    
    log_stream() << m_indent_string <<"-\n";
    set_indent_level(indent_level()+1);

    if(key_count == 0)
    {
        log_stream() << m_indent_string << name << ":\n";
    }
    else
    {
        log_stream() << m_indent_string << name << "_" << key_count <<":\n";
    }
    set_indent_level(indent_level()+1);
    // add timer for new level
    m_timers.push(Timer());
    // add key counter for new level
    m_key_counters.push(std::map<std::string,int>());
}

//-----------------------------------------------------------------------------
void
Logger::log_block_end(const std::string &name)
{
    // skip if log is not open
    if(!is_log_open())
    {
        return;
    }

    log_stream() << m_indent_string <<"-\n";
    log_stream() << m_indent_string << "  time_elapsed: " << m_timers.top().elapsed() << "\n";
    set_indent_level(indent_level()-2);
    m_key_counters.pop();
    m_timers.pop();
}

//-----------------------------------------------------------------------------
void
Logger::log_message(int level,
                    const std::string &msg,
                    const std::string &file,
                    int line)
{
    // echo if equal or above echo threshold
    if(level >= echo_threshold())
    {
        log_message(level, msg, file, line, std::cout, false);
    }

    // log if equal or above logging threshold
    if(level >= log_threshold())
    {
        log_message(level, msg, file, line, log_stream(), true);
    }
}


//-----------------------------------------------------------------------------
void
Logger::log_message(int level,
                    const std::string &msg,
                    const std::string &file,
                    int line,
                    std::ostream &os,
                    bool detailed)
{
    /*
    {parent_indent}-
    {parent_indent}{indent}level:
    {parent_indent}{indent}file:
    {parent_indent}{indent}line:
    ... msg txt
    */
    os << m_indent_string <<"-\n";
    os << m_indent_string << "  level: " << level_id_to_string(level) << "\n";
    if(detailed)
    {
        os << m_indent_string << "  file: "  << file  << "\n";
        os << m_indent_string << "  line: "  << line  << "\n";
        os << m_indent_string << "  timestamp: \"" << timestamp()  << "\"\n";
    }
    log_message_inner(msg, os);
}

//-----------------------------------------------------------------------------
void
Logger::log_message(int level,
                    const std::string &msg)
{
    // echo if equal or above echo threshold
    if(level >= echo_threshold())
    {
        log_message(level, msg, std::cout, false);
    }

    // log if equal or above logging threshold
    if(level >= log_threshold())
    {
        log_message(level, msg, log_stream(), true);
    }
}

//-----------------------------------------------------------------------------
void
Logger::log_message(int level,
                    const std::string &msg,
                    std::ostream &os,
                    bool detailed)
{
    /*
    {parent_indent}-
    {parent_indent}{indent}level:
    ... msg txt
    */
    os << m_indent_string <<"-\n";
    os << m_indent_string << "  level: " << level_id_to_string(level) << "\n";
    if(detailed)
    {
        os << m_indent_string << "  timestamp: \"" << timestamp()  << "\"\n";
    }
    log_message_inner(msg, os);
}


//-----------------------------------------------------------------------------
void
Logger::log_message_inner(const std::string &msg, 
                          std::ostream &os)
{
    /*
    {parent_indent}{indent}msg: |
    {parent_indent}{indent}{indent} msg line
    ...
    {parent_indent}{indent}{indent} msg line
    */
    os << m_indent_string << "  msg: |\n";
    std::istringstream input;
    input.str(msg);
    for (std::string line; std::getline(input, line);)
    {
        os << m_indent_string << "    " << line << "\n";
    }
}

//-----------------------------------------------------------------------------
int
Logger::indent_level() const
{
    return m_indent_level;
}

//-----------------------------------------------------------------------------
void
Logger::set_indent_level(int level)
{
    m_indent_level  = level;
    m_indent_string = std::string(m_indent_level*2, ' ');
}

//-----------------------------------------------------------------------------
int
Logger::rank() const
{
    return m_rank;
}

//-----------------------------------------------------------------------------
void
Logger::set_rank(int rank)
{
    m_rank = rank;
}

//-----------------------------------------------------------------------------
void
Logger::set_log_threshold(int level)
{
    m_log_threshold = level;
}

//-----------------------------------------------------------------------------
void
Logger::set_log_threshold(const std::string &level_str)
{
    set_log_threshold(level_string_to_id(level_str));
}

//-----------------------------------------------------------------------------
int
Logger::log_threshold() const
{
    return m_log_threshold;
}

//-----------------------------------------------------------------------------
void
Logger::set_echo_threshold(int level)
{
    m_echo_threshold = level;
}

//-----------------------------------------------------------------------------
void
Logger::set_echo_threshold(const std::string &level_str)
{
    set_echo_threshold(level_string_to_id(level_str));
}


//-----------------------------------------------------------------------------
int
Logger::echo_threshold() const
{
    return m_echo_threshold;
}

//-----------------------------------------------------------------------------
std::ostream &
Logger::log_stream()
{
    return m_log_stream;
}

//-----------------------------------------------------------------------------
Logger &
Logger::instance()
{
    return m_instance;
}

//-----------------------------------------------------------------------------
std::string
Logger::level_id_to_string(int level)
{
    if(level < Logger::LOG_ALL_ID )
    {
        level = Logger::LOG_ALL_ID;
    }
    else if(level < Logger::LOG_DEBUG_ID)
    {
        level = Logger::LOG_DEBUG_ID;
    }
    else if(level > Logger::LOG_ERROR_ID)
    {
        level = Logger::LOG_ERROR_ID;
    }
    switch(level)
    {
        case Logger::LOG_ALL_ID:   return "all";   break;
        case Logger::LOG_DEBUG_ID: return "debug"; break;
        case Logger::LOG_INFO_ID:  return "info";  break;
        case Logger::LOG_WARN_ID:  return "warn";  break;
        case Logger::LOG_ERROR_ID: return "error"; break;
        case Logger::LOG_NONE_ID:  return "none";  break;
        default: return "unknown"; break;
    }
}

//-----------------------------------------------------------------------------
int
Logger::level_string_to_id(const std::string &level_str)
{
    // strip, lower?
    if(level_str == "all" )
    {
        return Logger::LOG_ALL_ID;
    }
    else if(level_str == "debug" )
    {
        return Logger::LOG_DEBUG_ID;
    }
    else if(level_str == "info" )
    {
        return Logger::LOG_INFO_ID;
    }
    else if(level_str == "warn" )
    {
        return Logger::LOG_WARN_ID;
    }
    else if(level_str == "error" )
    {
        return Logger::LOG_ERROR_ID;
    }
    else if(level_str == "none" )
    {
        return Logger::LOG_NONE_ID;
    }
    else
    {
        // Unknown, default to all.
        return Logger::LOG_ALL_ID;
    }
}

//-----------------------------------------------------------------------------
std::string
Logger::timestamp()
{
    std::time_t time = std::time(nullptr);
    auto tm = *std::localtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





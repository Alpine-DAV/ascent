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

Logger                        Logger::m_instance;
Logger                       *Logger::m_active_instance = nullptr;
std::vector<std::string>      Logger::m_level_strings = {"unset",
                                                         "debug",
                                                         "info",
                                                         "warn",
                                                         "error",
                                                         "legendary"};

//-----------------------------------------------------------------------------
Logger::Scope::Scope(Logger *lgr, const std::string &name)
 : m_lgr(lgr),
   m_name(name)
{
    if(m_lgr != nullptr)
    {
        m_lgr->log_block_begin(m_name);
    }
}

//-----------------------------------------------------------------------------
Logger::Scope::~Scope()
{
    if(m_lgr != nullptr)
    {
        m_lgr->log_block_end(m_name);
    }
}

//-----------------------------------------------------------------------------
Logger::Logger()
 : m_indent_level(0),
   m_rank(-1),
   m_level_threshold(INFO),
   m_echo_level_threshold(LEGENDARY)
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
Logger::open(const std::string &ofpattern)
{
    // multi node case, assumes file pattern includes "rank"
    std::string ofname;
    if(rank() > -1)
    {
        ofname = conduit_fmt::format(ofpattern,
                                     conduit_fmt::arg("rank",rank()));
    }
    else
    {
        ofname = ofpattern;
    }

    m_ofstream.open(ofname.c_str());

    if(!m_ofstream.is_open())
    {
        std::cerr << "[ERROR] Failed to open log file: "  << ofname << std::endl;
    }
}

//-----------------------------------------------------------------------------
bool
Logger::is_open()
{
    return m_ofstream.is_open();
}

//-----------------------------------------------------------------------------
void
Logger::close()
{
    if(m_ofstream.is_open())
    {
        m_ofstream.close();
    }
}

//-----------------------------------------------------------------------------
void
Logger::flush()
{
    m_ofstream << std::flush;
}

//-----------------------------------------------------------------------------
void
Logger::log_block_begin(const std::string &name)
{
    // make sure we have a unique key name
    
    int key_count = m_key_counters.top()[name]++;
    
    stream() << m_indent_string <<"-\n";
    set_indent_level(indent_level()+1);

    if(key_count == 0)
    {
        stream() << m_indent_string << name << ":\n";
    }
    else
    {
        stream() << m_indent_string << name << "_" << key_count <<":\n";
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
    stream() << m_indent_string <<"-\n";
    stream() << m_indent_string << "  time_elapsed: " << m_timers.top().elapsed() << "\n";
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
    // log if equal or above logging threshold
    if(level >= log_threshold())
    {
        log_message(level, msg, file, line, stream(), true);
    }

    // echo if equal or above echo threshold
    if(level >= echo_threshold())
    {
        log_message(level, msg, file, line, std::cout, false);
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
    os << m_indent_string << "  level: " << level_string(level) << "\n";
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
    // log if equal or above logging threshold
    if(level >= log_threshold())
    {
        log_message(level, msg, stream(), true);
    }

    // echo if equal or above echo threshold
    if(level >= echo_threshold())
    {
        log_message(level, msg, std::cout, false);
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
    os << m_indent_string << "  level: " << level_string(level) << "\n";
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
    m_level_threshold = level;
}

//-----------------------------------------------------------------------------
int
Logger::log_threshold() const
{
    return m_level_threshold;
}

//-----------------------------------------------------------------------------
void
Logger::set_echo_threshold(int level)
{
    m_echo_level_threshold = level;
}

//-----------------------------------------------------------------------------
int
Logger::echo_threshold() const
{
    return m_echo_level_threshold;
}

//-----------------------------------------------------------------------------
std::ostream &
Logger::stream()
{
    return m_ofstream;
}

//-----------------------------------------------------------------------------
Logger *
Logger::instance()
{
    return m_active_instance;
}

//-----------------------------------------------------------------------------
void 
Logger::activate()
{
    m_active_instance = &m_instance;
}

//-----------------------------------------------------------------------------
void 
Logger::deactivate()
{
    m_active_instance = nullptr;
}

//-----------------------------------------------------------------------------
const std::string &
Logger::level_string(int level)
{
    if(level < Logger::UNKNOWN )
    {
        level = Logger::UNKNOWN;
    }
    else if(level > Logger::LEGENDARY)
    {
        level = Logger::LEGENDARY;
    }
    return m_level_strings[level];
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





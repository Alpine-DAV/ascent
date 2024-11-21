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

Logger                       *Logger::m_active_logger = nullptr;
std::map<std::string,Logger>  Logger::m_loggers;
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
{}

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
    stream() << m_indent_string << name << ":\n";
    set_indent_level(indent_level()+1);
}

//-----------------------------------------------------------------------------
void
Logger::log_block_end(const std::string &name)
{
    set_indent_level(indent_level()-1);
}

//-----------------------------------------------------------------------------
void
Logger::log_message(int level,
                    const std::string &msg,
                    const std::string &file,
                    int line)
{
    /*
    {parent_indent}-
    {parent_indent}{indent}level:
    {parent_indent}{indent}file:
    {parent_indent}{indent}line:
    ... msg txt
    */
    stream() << m_indent_string <<"-\n";
    stream() << m_indent_string << "  level: " << level_string(level) << "\n";
    stream() << m_indent_string << "  file: "  << file  << "\n";
    stream() << m_indent_string << "  line: "  << line  << "\n";
    log_message_inner(msg);
}

//-----------------------------------------------------------------------------
void
Logger::log_message(int level,
                    const std::string &msg)
{
    /*
    {parent_indent}-
    {parent_indent}{indent}level:
    ... msg txt
    */
    stream() << m_indent_string <<"-\n";
    stream() << m_indent_string << "  level: " << level_string(level) << "\n";
    log_message_inner(msg);
}


//-----------------------------------------------------------------------------
void
Logger::log_message_inner(const std::string &msg)
{
    /*
    {parent_indent}{indent}msg: |
    {parent_indent}{indent}{indent} msg line
    ...
    {parent_indent}{indent}{indent} msg line
    */
    stream() << m_indent_string << "  msg: |\n";
    std::istringstream input;
    input.str(msg);
    for (std::string line; std::getline(input, line);)
    {
        stream() << m_indent_string << "    " << line << "\n";
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
Logger::set_level_threshold(int level)
{
    m_level_threshold = level;
}

//-----------------------------------------------------------------------------
int
Logger::level_threshold() const
{
    return m_level_threshold;
}

//-----------------------------------------------------------------------------
void
Logger::set_echo_level_threshold(int level)
{
    m_echo_level_threshold = level;
}

//-----------------------------------------------------------------------------
int
Logger::echo_level_threshold() const
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
Logger::active_instance()
{
    return m_active_logger;
}

//-----------------------------------------------------------------------------
Logger *
Logger::activate_instance(const std::string &ofile_pattern)
{
    m_active_logger = &m_loggers[ofile_pattern];
    return m_active_logger;
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
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





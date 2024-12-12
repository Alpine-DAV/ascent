//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_logging.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_LOGGING_HPP
#define ASCENT_LOGGING_HPP

#include <ascent_logging_exports.h>
#include <ascent_logging_config.h>
#include <ascent_annotations.hpp>
#include <ascent_logging_timer.hpp>

#include <string>
#include <stack>
#include <conduit.hpp>

//-----------------------------------------------------------------------------
/*

Unified Logging Macros used by all libraries.alignas

macros:

ASCENT_LOG_OPEN( output_file_name_pattern ) // serial
ASCENT_LOG_OPEN( output_file_name_pattern, rank ) // mpi par

ASCENT_LOG_DEBUG( msg ) (w/ line, file)
ASCENT_LOG_INFO( msg )  (w/ line, file)
ASCENT_LOG_WARN( msg )  (w/ line, file)
ASCENT_LOG_ERROR( msg ) (w/ line, file)

ASCENT_LOG_SCOPE( name ) --> increase indent
ASCENT_LOG_MARK_BEGIN( name ) --> increase indent
ASCENT_LOG_MARK_END( name ) --> decrease indent
ASCENT_LOG_MARK_FUNCTION ()

ASCENT_LOG_FLUSH()
ASCENT_LOG_CLOSE()
*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#define ASCENT_LOG_OPEN( ofname_pattern )                                     \
{                                                                             \
    ascent::Logger::instance().open(ofname_pattern);                           \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_OPEN_RANK( ofname_pattern , rank )                         \
{                                                                             \
    ascent::Logger &lgr = ascent::Logger::instance();                         \
    lgr.set_rank(rank);                                                       \
    lgr.open(ofname_pattern);                                                 \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_DEBUG( msg )                                               \
{                                                                             \
    std::ostringstream _ascent_oss_info;                                      \
    _ascent_oss_info << msg;                                                  \
    ascent::Logger::instance().log_message(ascent::Logger::LOG_DEBUG_ID,      \
                                           _ascent_oss_info.str(),            \
                                           std::string(__FILE__),             \
                                           __LINE__);                         \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_INFO( msg )                                                \
{                                                                             \
    std::ostringstream _ascent_oss_info;                                      \
    _ascent_oss_info << msg;                                                  \
    ascent::Logger::instance().log_message(ascent::Logger::LOG_INFO_ID,       \
                                           _ascent_oss_info.str(),            \
                                           std::string(__FILE__),             \
                                           __LINE__);                         \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_WARN( msg )                                                \
{                                                                             \
    std::ostringstream _ascent_oss_info;                                      \
    _ascent_oss_info << msg;                                                  \
    ascent::Logger::instance().log_message(ascent::Logger::LOG_WARN_ID,       \
                                           _ascent_oss_info.str(),            \
                                           std::string(__FILE__),             \
                                           __LINE__);                         \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_ERROR( msg )                                               \
{                                                                             \
    std::ostringstream _ascent_oss_info;                                      \
    _ascent_oss_info << msg;                                                  \
    ascent::Logger::instance().log_message(ascent::Logger::LOG_ERROR_ID,      \
                                           _ascent_oss_info.str(),            \
                                           std::string(__FILE__),             \
                                           __LINE__);                         \
    ascent::Logger::instance().flush();                                       \
    throw conduit::Error(_ascent_oss_info.str(),                              \
                         std::string(__FILE__),                               \
                         __LINE__);                                           \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_FLUSH()                                                    \
{                                                                             \
    ascent::Logger::instance().flush();                                       \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_CLOSE()                                                    \
{                                                                             \
    ascent::Logger::instance().close();                                       \
}

//-----------------------------------------------------------------------------
#define ASCENT_MARK_SCOPE( name ) ASCENT_ANNOTATE_MARK_SCOPE; ascent::Logger::Scope _ascent_lgr_scope(ascent::Logger::instance(), name );

//-----------------------------------------------------------------------------
#define ASCENT_MARK_FUNCTION( name ) ASCENT_ANNOTATE_MARK_FUNCTION; ascent::Logger::Scope _ascent_lgr_func(ascent::Logger::instance(), std::string(__func__));

//-----------------------------------------------------------------------------
#define ASCENT_MARK_BEGIN( name ) ASCENT_ANNOTATE_MARK_BEGIN( name );         \
{                                                                             \
    ascent::Logger::instance().log_block_begin(name);                         \
}

//-----------------------------------------------------------------------------
#define ASCENT_MARK_END( name ) ASCENT_ANNOTATE_MARK_END( name );             \
{                                                                             \
    ascent::Logger::instance().log_block_end(name);                           \
}

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
class ASCENT_API Logger
{
public:

    //-------------------------------------------------------------------------
    typedef enum
    {
        LOG_ALL_ID   = -1,   // lowest
        LOG_DEBUG_ID =  1,
        LOG_INFO_ID  =  2,
        LOG_WARN_ID  =  3,
        LOG_ERROR_ID =  4,
        LOG_NONE_ID  =  127, // highest
    } MessageLevel;

    //-------------------------------------------------------------------------
    class ASCENT_API Scope
    {
    public:
         Scope(Logger &lgr, const std::string &name);
        ~Scope();
    private:
        Logger      &m_lgr;
        std::string  m_name;
    };

    //
    // file pattern examples:
    //  "ascent_log_out.yaml"
    //  "ascent_log_out_{rank}.yaml"
    //  "ascent_log_out_{rank:05d}.yaml"
    //
    void open(const std::string &ofile_pattern);
    bool is_log_open();
    void reset(); // reset to defaults
    void close();
    void flush();

    void log_message(int level,
                     const std::string &msg,
                     const std::string &file,
                     int line);

    void log_message(int level,
                     const std::string &msg);

    void log_block_begin(const std::string &name);
    void log_block_end(const std::string &name);

    int  rank() const;
    void set_rank(int rank);

    int  indent_level() const;
    void set_indent_level(int level);

    // any msgs >= log_threshold will be logged
    void set_log_threshold(int level);
    void set_log_threshold(const std::string &level_str);
    int  log_threshold() const;

    // any msgs >= echo_threshold will sent to std out as well as log
    void set_echo_threshold(int level);
    void set_echo_threshold(const std::string &level_str);
    int  echo_threshold() const;

    std::ostream &log_stream();
    static Logger &instance();

private:
    //-------------------------------------------------------------------------
    // constructor + destructor
    // ------------------------------------------------------------------------
    Logger();
    ~Logger();

    // ------------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------------
    static std::string level_id_to_string(int level);
    static int         level_string_to_id(const std::string &level_str);
    static std::string timestamp();

    void log_message(int level,
                     const std::string &msg,
                     const std::string &file,
                     int line,
                     std::ostream &os,
                     bool detailed);

    void log_message(int level,
                     const std::string &msg,
                     std::ostream &os,
                     bool detailed);

    void log_message_inner(const std::string &msg,
                           std::ostream &os);

    void log_block_begin(const std::string &name,
                         std::ostream &os);
    
    void log_block_end(const std::string &name,
                       std::ostream &os);

    // ------------------------------------------------------------------------
    // instance vars
    // ------------------------------------------------------------------------
    std::ofstream m_log_stream;
    bool          m_log_open;
    std::string   m_log_fname;
    int           m_indent_level;   // default = 0
    int           m_rank;           // default = -1
    int           m_log_threshold;  // default = INFO
    int           m_echo_threshold; // default = NONE
    std::string   m_indent_string;  // current indent string
    
    // ------------------------------------------------------------------------
    // logging state stacks
    // ------------------------------------------------------------------------
    // stack of timers
    std::stack<Timer>                       m_timers;
    // stack of block name counters
    // used to make sure we have unique keys in our yaml output
    std::stack<std::map<std::string,int>>   m_key_counters;

    // ------------------------------------------------------------------------
    // static members
    // ------------------------------------------------------------------------
    static Logger                        m_instance;
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------



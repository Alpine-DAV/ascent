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
#include <ascent_annotations.hpp>

#include <string>
#include <conduit.hpp>

/*
ASCENT_LOG_OPEN( output_file_name_pattern ) // serial
ASCENT_LOG_OPEN( output_file_name_pattern, rank ) // mpi par

ASCENT_LOG_DEBUG( msg ) (w/ line, file, etc)
ASCENT_LOG_INFO( msg )  (w/ line, file, etc)
ASCENT_LOG_WARN( msg )  (w/ line, file, etc)
ASCENT_LOG_ERROR( msg ) (w/ line, file, etc)

ASCENT_LOG_SCOPE( name ) --> increase indent
ASCENT_LOG_MARK_BEGIN( name ) --> increase indent
ASCENT_LOG_MARK_END( name ) --> decrease indent
ASCENT_LOG_MARK_FUNCTION ()

ASCENT_LOG_FLUSH()
*/

//-----------------------------------------------------------------------------
#define ASCENT_LOG_OPEN( ofname_pattern )                                      \
{                                                                              \
    ascent::Logger *lgr = ascent::Logger::activate_instance(ofname_pattern);   \
    lgr->open(ofname_pattern);                                                 \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_OPEN_RANK( ofname_pattern , rank )                          \
{                                                                              \
    ascent::Logger *lgr = ascent::Logger::activate_instance(ofname_pattern);   \
    lgr->set_rank(rank);                                                       \
    lgr->open(ofname_pattern);                                                 \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_ACTIVATE( ofname_pattern )                                  \
{                                                                              \
    Logger::activate_instance(ofname_pattern);                                 \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_DEBUG( msg )                                                \
{                                                                              \
    ascent::Logger *_ascent_lgr = Logger::active_instance();                   \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        std::ostringstream _ascent_oss_info;                                   \
        _ascent_oss_info << msg;                                               \
        _ascent_lgr->log_message(ascent::Logger::DEBUG,                        \
                                 _ascent_oss_info.str(),                       \
                                 std::string(__FILE__),                        \
                                 __LINE__);                                    \
    }                                                                          \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_INFO( msg )                                                 \
{                                                                              \
    ascent::Logger *_ascent_lgr = Logger::active_instance();                   \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        std::ostringstream _ascent_oss_info;                                   \
        _ascent_oss_info << msg;                                               \
        _ascent_lgr->log_message(ascent::Logger::INFO,                         \
                                 _ascent_oss_info.str(),                       \
                                 std::string(__FILE__),                        \
                                 __LINE__);                                    \
    }                                                                          \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_WARN( msg )                                                 \
{                                                                              \
    ascent::Logger *_ascent_lgr = Logger::active_instance();                   \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        std::ostringstream _ascent_oss_info;                                   \
        _ascent_oss_info << msg;                                               \
        _ascent_lgr->log_message(ascent::Logger::WARN,                         \
                                 _ascent_oss_info.str(),                       \
                                 std::string(__FILE__),                        \
                                 __LINE__);                                    \
    }                                                                          \
}

//-----------------------------------------------------------------------------
#define ASCENT_LOG_ERROR( msg )                                                \
{                                                                              \
    ascent::Logger *_ascent_lgr = ascent::Logger::active_instance();           \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        std::ostringstream _ascent_oss_info;                                   \
        _ascent_oss_info << msg;                                               \
        _ascent_lgr->log_message(ascent::Logger::ERROR,                        \
                                 _ascent_oss_info.str(),                       \
                                 std::string(__FILE__),                        \
                                 __LINE__);                                    \
    }                                                                          \
}// TODO EXCEPTION!

//-----------------------------------------------------------------------------
#define ASCENT_FLUSH ()                                                        \
{                                                                              \
    ascent::Logger *_ascent_lgr = ascent::Logger::active_instance();           \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        _ascent_lgr->flush();                                                  \
    }                                                                          \
}

//-----------------------------------------------------------------------------
#define ASCENT_MARK_SCOPE( name ) ASCENT_ANNOTATE_MARK_SCOPE; ascent::Logger::Scope _ascent_lgr_scope(ascent::Logger::active_instance(), name );

//-----------------------------------------------------------------------------
#define ASCENT_MARK_FUNCTION( name ) ASCENT_ANNOTATE_MARK_FUNCTION; ascent::Logger::Scope _ascent_lgr_func(ascent::Logger::active_instance(), std::string(__func__));

//-----------------------------------------------------------------------------
#define ASCENT_MARK_BEGIN( name ) ASCENT_ANNOTATE_MARK_BEGIN( name );          \
{                                                                              \
    ascent::Logger *_ascent_lgr = ascent::Logger::active_instance();           \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        _ascent_lgr->log_block_begin(name);                                    \
    }                                                                          \
}

//-----------------------------------------------------------------------------
#define ASCENT_MARK_END( name ) ASCENT_ANNOTATE_MARK_END( name );              \
{                                                                              \
    ascent::Logger *_ascent_lgr = ascent::Logger::active_instance();           \
    if(_ascent_lgr != nullptr)                                                 \
    {                                                                          \
        _ascent_lgr->log_block_end(name);                                      \
    }                                                                          \
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
        UNKNOWN   = -1,
        DEBUG     =  1,
        INFO      =  2,
        WARN      =  3,
        ERROR     =  4,
        LEGENDARY =  5,
    } MessageLevel;

    //-------------------------------------------------------------------------
    class ASCENT_API Scope
    {
    public:
         Scope(Logger *lgr, const std::string &name);
        ~Scope();
    private:
        Logger      *m_lgr;
        std::string  m_name;
    };

    //-------------------------------------------------------------------------
    Logger();
    ~Logger();

    //
    // "ascent_log_out.yaml"
    // "ascent_log_out_{rank}.yaml"
    // "ascent_log_out_{rank:05d}.yaml"
    //

    void open(const std::string &ofile_pattern);

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

    // any msgs >= level_threshold will be logged
    void set_level_threshold(int level);
    int  level_threshold() const;

    // any msgs >= echo_level_threshold will sent to std out as well as log
    void set_echo_level_threshold(int level);
    int  echo_level_threshold() const;


    std::ostream &stream();

    static Logger *active_instance();
    static Logger *activate_instance(const std::string &ofile_pattern);

private:
    static const std::string &level_string(int level);

    void log_message_inner(const std::string &msg);

    std::ofstream m_ofstream;

    int           m_indent_level;           // default =  0
    int           m_indent_spaces;          // default = 4
    int           m_rank;                   // default = -1
    int           m_level_threshold;        // default = INFO
    int           m_echo_level_threshold;   // default = LEGENDARY

    std::string   m_indent_string;          // current indent string

    static Logger                       *m_active_logger; // default = nullptr
    static std::map<std::string,Logger>  m_loggers;
    static std::vector<std::string>      m_level_strings;
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



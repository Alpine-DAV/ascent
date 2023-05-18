//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_HPP
#define ASCENT_HPP


#include <ascent_config.h>
#include <ascent_exports.h>


#include <ascent_logging.hpp>
#include <ascent_block_timer.hpp>

#include <conduit.hpp>
#include <conduit_blueprint.hpp>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

// Forward Declare the ascent::Runtime interface class.
class Runtime;

//-----------------------------------------------------------------------------
/// Ascent Interface
//-----------------------------------------------------------------------------
class ASCENT_API Ascent
{
public:
           Ascent();
          ~Ascent();

    void   open(); // open with default options
    void   open(const conduit::Node &options);
    void   publish(const conduit::Node &data);
    void   execute(const conduit::Node &actions);
    void   info(conduit::Node &info_out);
    void   close();

private:
    
    void           set_status(const std::string &msg);
    void           set_status(const std::string &msg,
                              const std::string &details);
    Runtime       *m_runtime;
    bool           m_verbose_msgs;
    bool           m_forward_exceptions;
    std::string    m_actions_file;
    conduit::Node  m_options;
    conduit::Node  m_status;
};


//-----------------------------------------------------------------------------
std::string ASCENT_API about();

//-----------------------------------------------------------------------------
void        ASCENT_API about(conduit::Node &node);

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


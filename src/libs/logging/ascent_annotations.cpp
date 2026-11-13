// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: ascent_annotations.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_annotations.hpp"
#include "ascent_logging.hpp"

//-----------------------------------------------------------------------------
// -- standard lib includes --
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
#include "caliper/Annotation.h"
#include "caliper/cali-manager.h"
#include "caliper/cali.h"
#endif

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::annotations --
//-----------------------------------------------------------------------------
namespace annotations
{

//-----------------------------------------------------------------------------
Scope::Scope(const std::string &name)
  : m_name(name),
    m_started(false)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    cali_begin_region(m_name.c_str());
    m_started = true;
#endif
}

//-----------------------------------------------------------------------------
Scope::~Scope()
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    if(m_started)
    {
        cali_end_region(m_name.c_str());
    }
#endif
}

//-----------------------------------------------------------------------------
struct Value::Impl
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    std::string name;
    cali::Annotation *annotation;
    bool active;

    explicit Impl(const std::string &name)
      : name(name),
        annotation(nullptr),
        active(false)
    {
    }

    ~Impl()
    {
        delete annotation;
    }

    void update(const cali::Variant &value, bool store_as_value)
    {
        if(annotation == nullptr)
        {
            int properties = CALI_ATTR_SCOPE_THREAD | CALI_ATTR_SKIP_EVENTS;
            if(store_as_value)
            {
                properties |= CALI_ATTR_ASVALUE;
            }
            annotation = new cali::Annotation(name.c_str(), properties);
        }

        if(active)
        {
            annotation->set(value);
        }
        else
        {
            annotation->begin(value);
            active = true;
        }
    }
#else
    explicit Impl(const std::string &)
    {
    }
#endif
};

//-----------------------------------------------------------------------------
Value::Value(const std::string &name)
  : m_impl(nullptr)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    m_impl = new Impl(name);
#else
    (void) name;
#endif
}

//-----------------------------------------------------------------------------
Value::~Value()
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    if(m_impl != nullptr)
    {
        if(m_impl->active)
        {
            m_impl->annotation->end();
        }
        delete m_impl;
    }
#endif
}

//-----------------------------------------------------------------------------
void
Value::set(const std::string &value)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    // Store strings in Caliper's context tree to ensure they remain valid,
    // rather than relying on a pointer to the original string's data.
    m_impl->update(cali::Variant(value.c_str()), false);
#else
    (void) value;
#endif
}

//-----------------------------------------------------------------------------
void
Value::set(const char *value)
{
    set(std::string(value == nullptr ? "" : value));
}

//-----------------------------------------------------------------------------
void
Value::set(std::int64_t value)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    m_impl->update(cali::Variant(CALI_TYPE_INT, &value, sizeof(value)), true);
#else
    (void) value;
#endif
}

//-----------------------------------------------------------------------------
void
Value::set(std::uint64_t value)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    m_impl->update(cali::Variant(value), true);
#else
    (void) value;
#endif
}

//-----------------------------------------------------------------------------
void
Value::set(double value)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    m_impl->update(cali::Variant(value), true);
#else
    (void) value;
#endif
}

#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
static cali::ConfigManager *cali_cfg_manager = NULL;
#endif


//-----------------------------------------------------------------------------
bool
supported()
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    return true;
#else
    return false;
#endif
}

//-----------------------------------------------------------------------------
void
initialize()
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    conduit::Node opts;
    initialize(opts);
#endif
}

//-----------------------------------------------------------------------------
void
initialize(const conduit::Node &opts)
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)

    if(cali_cfg_manager != NULL)
    {
        ASCENT_LOG_ERROR("Ascent Caliper Config Manager already initialized.")
    }

    // check opts
    std::string config;
    std::string services;
    std::string ofname;

    if(opts.has_child("config") && opts["config"].dtype().is_string())
    {
        config = opts["config"].as_string();
    }

    if(opts.has_child("services") && opts["services"].dtype().is_string())
    {
        services = opts["services"].as_string();
    }

    if(opts.has_child("output_file") && opts["output_file"].dtype().is_string())
    {
        ofname = opts["output_file"].as_string();
    }

    cali_cfg_manager = new cali::ConfigManager();
    if(!ofname.empty())
    {
        cali_cfg_manager->set_default_parameter("output", ofname.c_str());
    }

    if(!config.empty())
    {
        cali_cfg_manager->add(config.c_str());
    }

    if(!services.empty())
    {
        cali_config_preset("CALI_SERVICES_ENABLE", services.c_str());
    }
    cali_cfg_manager->start();

#endif
}


//-----------------------------------------------------------------------------
void
flush()
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    if(cali_cfg_manager != NULL)
    {
        cali_cfg_manager->flush();
    }
#endif
}


//-----------------------------------------------------------------------------
void
finalize()
{
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
    flush();
    if(cali_cfg_manager != NULL)
    {
        delete cali_cfg_manager;
        cali_cfg_manager = NULL;
    }
#endif
}

}
//-----------------------------------------------------------------------------
// -- end ascent::annotations --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

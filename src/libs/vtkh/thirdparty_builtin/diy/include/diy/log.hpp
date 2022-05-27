#ifndef DIY_LOG_HPP
#define DIY_LOG_HPP

#ifndef DIY_USE_SPDLOG

#include <memory>
#include "fmt/format.h"
#include "fmt/ostream.h"

namespace vtkhdiy
{

namespace spd
{
    struct logger
    {
        // logger.info(cppformat_string, arg1, arg2, arg3, ...) call style
        template <typename... Args> void trace(const char* fmt, const Args&... args)    {}
        template <typename... Args> void debug(const char* fmt, const Args&... args)    {}
        template <typename... Args> void info(const char* fmt, const Args&... args)     {}
        template <typename... Args> void warn(const char* fmt, const Args&... args)     {}
        template <typename... Args> void error(const char* fmt, const Args&... args)    {}
        template <typename... Args> void critical(const char* fmt, const Args&... args) {}
    };
}

inline
std::shared_ptr<spd::logger>
get_logger()
{
    return std::make_shared<spd::logger>();
}

inline
std::shared_ptr<spd::logger>
create_logger(std::string)
{
    return std::make_shared<spd::logger>();
}

template<class... Args>
std::shared_ptr<spd::logger>
set_logger(Args... args)
{
    return std::make_shared<spd::logger>();
}

}   // diy

#else // DIY_USE_SPDLOG

#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>

#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/bundled/ostream.h>

namespace vtkhdiy
{

namespace spd = ::spdlog;

inline
std::shared_ptr<spd::logger>
get_logger()
{
    auto log = spd::get("diy");
    if (!log)
    {
        auto null_sink = std::make_shared<spd::sinks::null_sink_mt> ();
        log = std::make_shared<spd::logger>("null_logger", null_sink);
    }
    return log;
}

inline
std::shared_ptr<spd::logger>
create_logger(std::string log_level)
{
    auto log = spd::stderr_logger_mt("diy");
    int lvl;
    for (lvl = spd::level::trace; lvl < spd::level::off; ++lvl)
        if (spd::level::level_names[lvl] == log_level)
            break;
    log->set_level(static_cast<spd::level::level_enum>(lvl));
    return log;
}

template<class... Args>
std::shared_ptr<spd::logger>
set_logger(Args... args)
{
    auto log = std::make_shared<spdlog::logger>("diy", args...);
    return log;
}

}   // diy
#endif


#endif // DIY_LOG_HPP

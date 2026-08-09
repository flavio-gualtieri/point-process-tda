#ifndef TIME_MEASUREMENT_H
#define TIME_MEASUREMENT_H
#include "typedefs.hpp"
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <vector>
#include <chrono>
#include <optional>

/// Classes and functions for measuring and pretty printing run time for functions and blocks.
namespace Timing {
    using namespace std::chrono;
    using clock = system_clock;
    struct Context {
        std::string message;
        std::optional<time_point<clock>> start;
    };
    extern std::vector<Context> contexts;
    extern bool muted;
    const uint indentation = 4;
    const uint msg_width = 80;
    const std::string bold = "\33[1m";
    const std::string reverse = "\33[7m";
    const std::string reset = "\33[0m";
    const uint time_width = 6;
    extern enum LineStartsWith {
        _none,
        _context,
        _info
    } line_starts_with;

    /// Don't print any output until umute() is called.
    inline void mute(){
        muted = true;
    }
    /// See mute().
    inline void unmute(){
        muted = false;
    }
    /// Start a context with given message.
    void start(std::string message);
    /// Stop a context and print elapsed time. If other things have been printed in between, re-print the message.
    void stop();
    void section(std::string);

    /** Time measurement by scope.
    *
    * Wrapper around Timer::start and Timer::stop that starts measurng on construction
    * and stops on destruction. Use, e.g., as 
    * 
    *     Timer::time("message"), f();`.
    */
    struct time{
        time(std::string msg){Timing::start(msg);}
        ~time(){Timing::stop();}
    };
    /// Use this instead of std::endl with info().
    template< class CharT, class Traits>
    std::basic_ostream<CharT, Traits>& endl( std::basic_ostream<CharT, Traits>& os){
        if(muted)
            return os;
        line_starts_with = _none;
        return os << std::endl << std::string(indentation * contexts.size(), ' ');
    }
    /** Use for printing status messages. 
     * 
     * Use as
     * 
     *     Timing::info() << message << Timing::endl;
     */
    std::ostream& info();
};

#endif

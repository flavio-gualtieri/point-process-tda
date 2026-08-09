#include "time_measurement.hpp"
#include <iomanip>
#include <iostream>
#include <assert.h>
namespace Timing {
    std::vector<Context> contexts;
    bool muted;
    bool on_new_line = true;
    LineStartsWith line_starts_with = _none;
    void start(std::string message) {
        if(muted)
            return;
        auto begin = clock::now();
        if(line_starts_with != _none) {
            std::cout << reset << endl;
        }
        std::cout << std::left << std::setw(msg_width - indentation*contexts.size()) << message  << std::flush;
        contexts.push_back({message, begin});
        line_starts_with = _context;
    }
    void section(std::string message) {
        if(muted)
            return;
        if(line_starts_with != _none) {
            std::cout << reset << endl;
        }
        std::cout << std::left << std::setw(msg_width - indentation*contexts.size()) << message  << std::flush;
        contexts.push_back({message, {}});
        line_starts_with = _context;
    }
    void stop() {
        if(muted)
            return;
        auto end = clock::now();
        assert(!contexts.empty());
        auto c = contexts.back();
        contexts.pop_back();
        if(line_starts_with != _context) {
            if(line_starts_with != _none){
                std::cout << reset << endl;
            }
            else {
                std::cout << "\b\b\b\b";
            }
            if(c.start)
                std::cout << std::left << std::setw(msg_width - indentation*contexts.size()) << c.message  << std::flush;
        }
        if(c.start){
            std::cout
                << std::right
                << std::setw(6)
                << duration_cast<milliseconds>(end - *c.start).count() << "ms"
                << endl;
            line_starts_with = _none;
        }
    }


    class NullStream : public std::ostream {
        public:
        NullStream() :
            std::ostream(&m_sb)
        {}

        private:
        class NullBuffer : public std::streambuf {
        public:
            int overflow(int c) {
                return c;
            }
        } m_sb;
    } null_stream;
    std::ostream& info() {
        if(muted)
            return null_stream;
        if(line_starts_with != _none && line_starts_with != _info) {
            std::cout << reset << endl;
        }
        line_starts_with = _info;
        return std::cout;
    }
} // namespace Timing

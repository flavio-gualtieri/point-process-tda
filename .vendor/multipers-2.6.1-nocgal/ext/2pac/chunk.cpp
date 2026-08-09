#include "chunk.hpp"
#include "minimize.hpp"
#include "time_measurement.hpp"
#include <string>
#include <vector>
Chunk::Chunk(std::shared_ptr<Complex> _complex, uint max_dimension) :
    complex(std::move(_complex)),
    d(0),
    max_dimension(max_dimension)
{
    if(max_dimension > 0){
        auto next = complex->next_matrix();
        if(is_cochain())
            Timing::time("Chunk (cochain complex): minimize D^1"),
            std::tie(nlr, std::ignore, on_hold) = minimize(next);
        else
            Timing::time("Chunk (chain complex): minimize D_1"),
            std::tie(std::ignore, nlr, on_hold) = minimize(next);

    }
}

std::string Chunk::format_D(uint d){
    return std::string("D") + (is_cochain() ? "^" : "_") + std::to_string(d);
}
GradedMatrix Chunk::next_matrix() {
    ++d;
    if (d < max_dimension) {
        // fetch next matrix, truncate columns, minimize, truncate columns of matrix on hold.
        // if there are no more matrices; there's still one matrix on hold.
        GradedMatrix next;
        try {
            next = complex->next_matrix();
        }
        catch(EOI) {
            max_dimension = d;
            goto yield_on_hold;
        }
        Timing::time t(is_cochain() ? "Chunk (cochain complex)" : "Chunk (chain complex)");
        std::vector<size_t> nlc;

        if(is_cochain()){
            Timing::time("truncate columns of D^" + std::to_string(d+1)),
            next = std::move(next).get_columns(nlr);
            Timing::time("minimize D^" + std::to_string(d+1)),
            std::tie(nlr, nlc, next) = minimize(next);
            Timing::time("truncate rows of D^" + std::to_string(d)),
            on_hold = on_hold.get_rows(nlc);
        }
        else{
            Timing::time("truncate rows of D_" + std::to_string(d+1)),
            next = next.get_rows(nlr);
            Timing::time("minimize D_" + std::to_string(d+1)),
            std::tie(nlc, nlr, next) = minimize(next);
            Timing::time("truncate columns of D_" + std::to_string(d)),
            on_hold = std::move(on_hold).get_columns(nlc);
        }
        std::swap(next, on_hold);
        return next;
    } else if (d == max_dimension) {
        // the required matrix is already on hold; just yield.
        yield_on_hold:
        Timing::time t(is_cochain() ? "Chunk (cochain complex)" : "Chunk (chain complex)");
        Timing::info() << format_D(d) << " is already minimized." << Timing::endl;
        return on_hold;
    } else if (d == max_dimension + 1 && d > 1) {
        // fetch next matrix and truncate columns.
        Timing::time t(is_cochain() ? "Chunk (cochain complex)" : "Chunk (chain complex)");
        GradedMatrix next = complex->next_matrix();
        if(is_cochain())
            return (
                Timing::time("truncate columns of D^" + std::to_string(d)),
                std::move(next).get_columns(nlr)
            );
        else
            return (
                Timing::time("truncate rows of D_" + std::to_string(d)),
                next.get_rows(nlr)
            );
    } else {
        // just yield unchanged.
        GradedMatrix next = complex->next_matrix();
        return next;
    }
}

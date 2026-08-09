#ifndef CHUNK_H
#define CHUNK_H
#include "matrices.hpp"
#include "complexes.hpp"
#include <string>

/** Applies chunk preprocessing to (co)chain complexes.
 *
 * Because to remove all local rows from D[i] (the matrix from dimension i-1 to i),
 * including the local (i, i+1)-pairs, one has also to consider D[i+1]. Therefore,
 * a Chunk-instances always keeps a matrix on hold until it reaches `max_dimension`.
 */
class Chunk: public Complex{
    public:
    /** Creates a Chunk preprocessor.
     *
     * Fetches and minimizes the first coboundary matrix.
     *
     * @param matrices An instance from where to fetch the matrices. Must provide `next_coboundary_matrix`.
     * @param max_dimension Apply chunk to the first `max_dimension` matrices (counting from 1).
     *        the n+1-st matrix will have a reduced number of columns. Starting from the n+2nd, the matrices
     *        are untouched.
     */
    Chunk(std::shared_ptr<Complex> complex, uint max_dimension);
    GradedMatrix next_matrix();
    bool is_cochain(){
        return complex->is_cochain();
    }

    private:
    std::string format_D(uint d);
    std::shared_ptr<Complex> complex;
    GradedMatrix on_hold;
    std::vector<size_t> nlr; /// Holds the non-local rows (for cochain complexes) or columns (for chain complex) from the previous iteration.
    uint d;
    uint max_dimension;
};

#endif

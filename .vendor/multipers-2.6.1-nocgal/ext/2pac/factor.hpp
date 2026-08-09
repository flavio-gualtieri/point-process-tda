#ifndef FACTOR_H
#define FACTOR_H
#include "matrices.hpp"

/** Factors a matrix through another.
 * 
 * Computes a matrix M such that matrix = through * M.
 */
template<class T>
SparseMatrix factor_matrix(T &matrix, T &through);

inline GradedMatrix factor_matrix(GradedMatrix& matrix, GradedMatrix& through){
    assert(matrix.row_grades == through.row_grades);
    return GradedMatrix(through.column_grades, matrix.column_grades, factor_matrix(matrix.data, through.data));
}

/** Sets the indirection level for factor_matrix(). 
 * 
 * Disabled at the moment.
 */
void set_factorization_indirection_level(uint level);
#endif

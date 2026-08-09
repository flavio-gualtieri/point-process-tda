#ifndef MINIMIZE_H
#define MINIMIZE_H
#include "matrices.hpp"
#include "typedefs.hpp"
#include <vector>

/** Splits off homological “balls”.
 * 
 * If \p M is considered as a two step chain complex of free modules, 
 * the function splits off homological balls; i.e., trivial summands.
 * Template instantiations for `SparseMatrix` and `BlockColumnMatrix` are provided.
*/
template<class T>
std::tuple<std::vector<size_t>, std::vector<size_t>, GradedMatrix> minimize(GenericGradedMatrix<T>& M);


#endif

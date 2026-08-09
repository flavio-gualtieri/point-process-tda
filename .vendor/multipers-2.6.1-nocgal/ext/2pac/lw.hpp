#ifndef HOMOLOGY_H
#define HOMOLOGY_H
#include "matrices.hpp"
/* Computes the graded kernel of the matrix, and a minimal
 * generating system of the image and a kernel of it.
 * @returns A triple of three graded matrices:
 * - the kernel
 * - the minimal generating system of the image
 * - the kernel of the minimal generating system
 * The basis elements of the kernel (and also the kernel of the mgs) are returned in lex order (sic).
 */
template<class Ordering = grade::lex_greater>
std::tuple<GradedMatrix, GradedMatrix, GradedMatrix> kernel_mgs(GradedMatrix&);

void set_homology_indirection_level(uint);
#endif

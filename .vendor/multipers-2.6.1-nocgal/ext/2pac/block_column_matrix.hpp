#ifndef BLOCK_COLUMN_MATRIX_H
#define BLOCK_COLUMN_MATRIX_H

#include "typedefs.hpp"
#include "matrices.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
#include "utils.hpp"

/// A matrix that consists of block matrices, stacked vertically.
class BlockColumnMatrix {
    public:
    BlockColumnMatrix(std::vector<SparseMatrix> rows = {});
    BlockColumnMatrix(SparseMatrix);
    explicit operator SparseMatrix() const&;
    explicit operator SparseMatrix() &&;
    BlockColumnMatrix operator*(const SparseMatrix&);
    BlockColumnMatrix operator*(const BlockColumnMatrix&);
    void column_operation(uint from, uint to);
    void column_operation(const BlockColumnMatrix&, uint from, uint to);
    index_t pivot(uint column);
    index_t pop_pivot(uint column);
    uint rows() const;
    uint columns() const;
    void v_extend(SparseMatrix);
    const SparseMatrix& get_block_row(uint index) const;
    BlockColumnMatrix get_columns(const std::vector<size_t>&) const &;
    BlockColumnMatrix get_columns(const std::vector<size_t>&) &&;
    BlockColumnMatrix get_rows(const std::vector<size_t>&);
    bool is_zero();

    uint n_rows;
    std::vector<SparseMatrix> matrices;
};

SparseMatrix operator*(const SparseMatrix&, const BlockColumnMatrix&);
#endif

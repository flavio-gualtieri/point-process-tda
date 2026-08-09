#include "block_column_matrix.hpp"
#include "matrices.hpp"
#include "typedefs.hpp"
#include <cassert>
#include <vector>

BlockColumnMatrix::BlockColumnMatrix(std::vector<SparseMatrix> matrices_):
    n_rows(0), matrices(std::move(matrices_))
{
    assert(std::all_of(matrices.begin(), matrices.end(), [&](SparseMatrix& m){return matrices[0].columns() == m.columns();}));
    for(auto& m : matrices)
        n_rows += m.rows();
}
BlockColumnMatrix::BlockColumnMatrix(SparseMatrix m):
    n_rows(m.rows()),
    matrices({std::move(m)})
{}

BlockColumnMatrix::operator SparseMatrix() const& {
    SparseMatrix result = matrices.front();
    for(auto i = matrices.begin()+1; i != matrices.end(); i++)
        result.v_extend(*i);
    return result;
}

BlockColumnMatrix::operator SparseMatrix() && {
    SparseMatrix &result = matrices.front();
    for(auto i = matrices.begin()+1; i != matrices.end(); i++)
        result.v_extend(*i);
    return result;
}

void BlockColumnMatrix::column_operation(uint from, uint to){
    for(auto& m : matrices)
        m.column_operation(from, to);
}

void BlockColumnMatrix::column_operation(const BlockColumnMatrix &other, uint from, uint to){
    assert(matrices.size() == other.matrices.size());
    auto i2 = other.matrices.begin();
    for(auto i = matrices.begin(); i != matrices.end(); ++i, ++i2)
        i->column_operation(*i2, from, to);
}

index_t BlockColumnMatrix::pivot(uint column){
    if(matrices.empty())
        return no_pivot;
    auto pedestal = n_rows;
    for(auto m = matrices.rbegin(); m != matrices.rend(); ++m){
        pedestal -= m->rows();
        index_t i = m->pivot(column);
        if(i != no_pivot)
            return pedestal + i;
    }
    return no_pivot;
}

index_t BlockColumnMatrix::pop_pivot(uint column){
    if(matrices.empty())
        return no_pivot;
    auto pedestal = n_rows;
    for(auto m = matrices.rbegin(); m != matrices.rend(); ++m){
        pedestal -= m->rows();
        index_t i = m->pop_pivot(column);
        if(i != no_pivot)
            return pedestal + i;
    }
    return no_pivot;
}

uint BlockColumnMatrix::rows() const {
    return n_rows;
};
uint BlockColumnMatrix::columns() const {
    if(matrices.empty())
        return 0;
    else
        return matrices.front().columns();
}
void BlockColumnMatrix::v_extend(SparseMatrix m) {
    assert(matrices.empty() || columns() == m.columns());
    n_rows += m.rows();
    matrices.push_back(std::move(m));
}
const SparseMatrix& BlockColumnMatrix::get_block_row(uint index) const {
    return matrices[index];
}

BlockColumnMatrix BlockColumnMatrix::get_columns(const std::vector<size_t>& indices) const & {
    decltype(matrices) new_matrices;
    new_matrices.reserve(matrices.size());
    for(auto& m : matrices){
        new_matrices.push_back(m.get_columns(indices));
    }
    return BlockColumnMatrix(new_matrices);
}
BlockColumnMatrix BlockColumnMatrix::get_columns(const std::vector<size_t>& indices) && {
    for(auto& m : matrices){
        m = std::move(m).get_columns(indices);
    }
    return std::move(*this);
}

BlockColumnMatrix BlockColumnMatrix::get_rows(const std::vector<size_t>& indices) {
    BlockColumnMatrix result;
    result.matrices.reserve(matrices.size());
    auto i = indices.begin();
    uint u = 0;
    for(auto& m : matrices){
        std::vector<size_t> indices_block;
        for(; *i < u + m.rows(); ++i)
            indices_block.push_back(*i);
        result.v_extend(m.get_rows(indices));
        u += m.rows();
    }
    return result;
}

bool BlockColumnMatrix::is_zero(){
    for(auto &m : matrices)
        if(!m.is_zero())
            return false;
    return true;
}

BlockColumnMatrix BlockColumnMatrix::operator*(const SparseMatrix &other){
    assert(columns() == other.rows());
    BlockColumnMatrix result;
    for(auto &m : matrices)
        result.v_extend(m * other);
    return result;
}

BlockColumnMatrix BlockColumnMatrix::operator*(const BlockColumnMatrix &other){
    assert(columns() == other.rows());
    BlockColumnMatrix result;
    for(auto &m : matrices)
        result.v_extend(m * other);
    return result;
}

SparseMatrix operator*(const SparseMatrix &left, const BlockColumnMatrix& right) {
    assert(left.columns() == right.rows());
    SparseMatrix result(left.rows(), right.columns());
    uint pedestal = 0;
    for(auto &m : right.matrices){
        #pragma omp parallel for
        for(uint j = 0; j < right.columns(); ++j)
            for(auto e : m[j].data)
                result[j] += left[e + pedestal];
        pedestal += m.rows();
    }
    return result;
}
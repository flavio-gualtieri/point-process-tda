#ifndef INDIRECT_H
#define INDIRECT_H
#include "typedefs.hpp"
#include "matrices.hpp"
#include <algorithm>
#ifdef ARRAY_MATRICES
#include <boost/heap/priority_queue.hpp>


/** Indirect representation of matrix columns
 *
 * Given an ArrayColumn, an IndirectColumn represents the linear combination resulting from
 * the addition of further ArrayColumn instances indirectly. The underlying data structure
 * is a heap of pairs, each of which consists of a pointer to an ArrayColumn and the index
 * of the bottommost non-zero entry of that column that does not cancel with an entry of
 * another column in the same linear combination. Column additions are implemented by updating
 * the heap accordingly.
 *
 * It is assumed that the std::vector underlying an ArrayColumn is ordered descendingly
 * (i.e., the pivot comes first), without containing duplicate entries.
 */
struct IndirectColumn {
    IndirectColumn() = default;
    IndirectColumn(const IndirectColumn&) = default;
    IndirectColumn(ArrayColumn& column): size(column.size) {
        entry entry{0, &column};
        if(entry)
            pointers.push(entry);
    }
    IndirectColumn& operator=(ArrayColumn& column){
        pointers.clear();
        size = column.size;
        entry entry{0, &column};
        if(entry)
            pointers.push(entry);
        return *this;
    }
    void cancel_pivot_with(ArrayColumn& other){
        entry top = pointers.top();
        pointers.pop();
        if(++top)
            pointers.push(top);
        entry add{1, &other};
        if(add)
            pointers.push(add);
        consolidate_pivot();
        modified = true;
    }
    void operator+=(ArrayColumn& other){
        if(other.pivot() != no_pivot){
            pointers.push({0, &other});
            consolidate_pivot();
            modified = true;
        }
    }
    void operator+=(IndirectColumn& other){
        for(auto &e : other.pointers)
        {
            assert(e);
            pointers.push(e);
        }
        consolidate_pivot();
    }
    operator ArrayColumn() && {
        ArrayColumn result(size);
        for(;;){
            index_t i = pop_pivot();
            if(i == no_pivot)
                break;
            result.data.push_back(i);
        }
        return result;
    }
    operator ArrayColumn() & {
        return (ArrayColumn)IndirectColumn(*this);
    }
    index_t pivot() const {
        if(pointers.size() == 0)
            return no_pivot;
        else
            return pointers.top().row();
    }
    index_t pop_pivot(){
        if(!pointers.empty()){
            entry top = pointers.top();
            uint pivot = top.row();
            pointers.pop();
            if(++top)
                pointers.push(top);
            consolidate_pivot();
            modified = true;
            return pivot;
        }
        else
            return no_pivot;
    }
    bool was_modified(){
        return modified;
    }
    bool is_well_defined(){
        return std::all_of(pointers.begin(), pointers.end(), [](entry e){return (bool)e;});
    }

    index_t consolidate_pivot(){
        for(;;){
            // If there are no more entries in `pointers`, nothing needs to be consolidated.
            if(pointers.size() == 0)
                return no_pivot;
            if(pointers.size() == 1)
                return pointers.top().row();

            entry top = pointers.top();
            pointers.pop();

            // If the next entry does not have the same row index, we are done.
            if(top.row() != pointers.top().row()){
                pointers.push(top);
                return top.row();
            }
            // Otherwise, the entry in `top` and `pointers.top()` cancel. Both may have further entries.
            if(++top)
                pointers.push(top);
            // This cannot be the same top that's been inserted in the line above, because that has a strictly smaller row index.
            top = pointers.top();
            pointers.pop();
            if(++top)
                pointers.push(top);
        }
    }
    struct entry{
        size_t index;
        ArrayColumn* column;
        entry(size_t index, ArrayColumn *column):
            index(index), column(column)
        {
            assert(true);
        }
        uint row() const {
            return column->data[index];
        }
        bool operator<(entry other) const {
            return row() < other.row();
        }
        operator bool() const {
            // The following assumes that the data of an ArrayColumn is descendingly sorted.
            return index < column->data.size();
        }
        entry operator++(){
            // The following assumes that the data of an ArrayColumn is descendingly sorted.
            ++index;
            return *this;
        }
    };
    boost::heap::priority_queue<entry> pointers;
    uint size;
    bool modified = false;
};

class IndirectMatrix {
    public:
    IndirectMatrix(SparseMatrix &_matrix):
        matrix(_matrix)
    {
        columns.reserve(matrix.columns());
        for(size_t j=0; j < matrix.columns(); j++)
            columns.emplace_back(matrix[j]);
    }
    index_t pivot(uint j){
        return columns[j].pivot();
    }
    index_t pop_pivot(uint j){
        return columns[j].pop_pivot();
    }
    void column_operation(SparseMatrix &other, uint from, uint to){
        columns[to] += other[from];
    }
    void column_operation(uint from, uint to){
        columns[to] += columns[from];
    }
    IndirectColumn& operator[](uint j){
        return columns[j];
    }
    operator SparseMatrix() && {
        SparseMatrix result(matrix.rows(), matrix.columns());
        for(uint j = 0; j < matrix.columns(); j++)
            result[j] = std::move(columns[j]);
        return result;
    }

    private:
    SparseMatrix &matrix;
    std::vector<IndirectColumn> columns;
};

class SemiIndirectMatrix {
    public:
    SemiIndirectMatrix(SparseMatrix &_matrix):
        matrix(_matrix)
    {}
    index_t pivot(uint j){
        return column.pivot();
    }
    index_t pop_pivot(uint j){
        return column.pop_pivot();
    }
    void column_operation(SparseMatrix &other, uint from, uint to){
        column += other[from];
    }
    void column_operation(uint from, uint to){
        column += matrix[from];
    }
    void finalize(){
        if(active_column){
            matrix[*active_column] = std::move(column);
            active_column = {};
        }
    }
    void activate(uint j){
        finalize();
        active_column = j;
        column = matrix[j];
    }


    private:
    std::optional<uint> active_column;
    IndirectColumn column;
    SparseMatrix &matrix;
};

template<uint indirection_level>
using MatrixType =
    std::conditional_t<indirection_level == 1, SemiIndirectMatrix,
    std::conditional_t<indirection_level == 2, IndirectMatrix, SparseMatrix&>>;

constexpr bool using_semiindirect_matrices(uint indirection_level){
    return indirection_level == 1;
}

#else
template<uint indirection_level>
using MatrixType = SparseMatrix&;
constexpr bool using_semiindirect_matrices(uint indirection_level){
    return false;
}
#endif
#endif

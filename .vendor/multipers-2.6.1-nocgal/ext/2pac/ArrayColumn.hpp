#ifndef ARRAYCOLUMN_H
#define ARRAYCOLUMN_H
/* If ARRAY_COLUMNS_ASCENDING is not defined, the vectors underlying a ArrayColumn are sorted descendingly.
 * Otherwise, ascendingly.
 */
// #define ARRAY_COLUMNS_ASCENDING
#include "skip_vector.hpp"

/**
 *  A sparse matrix column implemented by a descendingly sorted vector.
 * 
 * Represents a column of a matrix using a vector that contains the row indices of the
 * nonzero entries without duplication in descendingly sorted order. See HeapColumn for
 * a description of the common methods. 
 */
class ArrayColumn {
    public:
    enum sorting { unsorted, ascending, descending, trivial };
    static ArrayColumn eye(uint size, uint row);
    static ArrayColumn ones(uint size);
    explicit ArrayColumn(uint size = 0);
    explicit ArrayColumn(uint size, uint row);
    ArrayColumn(uint size, std::vector<uint> data, sorting);
    ArrayColumn get_rows(uint size, const std::vector<index_t>& lut) const;
    void get_rows_consolidate(const std::vector<index_t>& lut, ArrayColumn &into) const;
    void get_rows(const std::vector<index_t>& lut, ArrayColumn &into) const;
    void reindex_rows(const std::vector<size_t>& lut);
    ArrayColumn& operator+=(const ArrayColumn& other);
    ArrayColumn& v_extend(uint top, uint bottom);
    ArrayColumn& v_extend(const ArrayColumn& other);
    void clear();
    void add_one(index_t row);
    index_t pivot();
    index_t pop_pivot() ;
    bool operator==(const ArrayColumn& other) const;
    void consolidate() {};
    inline bool is_zero(){
        return data.size() == 0;
    }
    bool is_well_defined() const;
    uint rows() const {
        return size;
    }
    operator std::vector<uint>() && {
        return std::move(data);
    }

    uint size;

    skip_vector<uint> data;
};
#endif

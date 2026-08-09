#ifndef HEAPCOLUMN_H
#define HEAPCOLUMN_H
#include "typedefs.hpp"
#include "priority_queue.hpp"
#include <vector>

/// Matrix column represented by binary heaps.
class HeapColumn {
    public:
    enum sorting {unsorted, ascending, descending, trivial};
    static HeapColumn ones(uint size);                          ///< Column filled with ones
    HeapColumn() = default;                                     ///< Column of size 0.
    HeapColumn(uint size, uint row);                            ///< Column with a single 1 in \p row.
    explicit HeapColumn(int size);                              ///< Column filled with 0.
    HeapColumn(uint size, priority_queue<uint> data);
    HeapColumn(uint size, std::vector<uint> entries, sorting how_sorted = unsorted);

    HeapColumn get_rows(uint size, const std::vector<index_t>& lut) const;
    void get_rows(const std::vector<index_t>& lut, HeapColumn &into) const;
    void get_rows_consolidate(const std::vector<index_t>& lut, HeapColumn &into) &&;
    void reindex_rows(const std::vector<size_t>& indices);      ///< Replace every entry i by indices[i].
    HeapColumn operator+(const HeapColumn& other) const;
    HeapColumn& operator+=(const HeapColumn& other);
    HeapColumn& v_extend(uint top, uint bottom);                ///< Extend by given number of zeros at top and bottom.
    HeapColumn& v_extend(const HeapColumn&);                    ///< Append other column at bottom.
    void clear();                                               ///< Set all entries to zero.
    void add_one(index_t row);                                  ///< Add a single 1 in \p row.
    index_t pivot();                                            ///< Row index of the largest non-zero entry.
    index_t pop_pivot();                                        ///< Remove and return the pivot.
    void consolidate();                                         ///< Add all entries of the same row index. The entries are descendingly sorted afterwards.
    bool is_zero();
    bool is_well_defined() const;                               ///< True if no entry exceeds the \c size of the column.
    uint rows() const {
        return size;
    }
    operator std::vector<uint>() && {
        return std::move(data);
    }

    uint size;
    priority_queue<uint> data;
};


#endif

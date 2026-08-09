#include "ArrayColumn.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
using namespace std;
ArrayColumn ArrayColumn::eye(uint size, uint row) {
    assert(row < size);
    ArrayColumn result(size);
    result.data.push_back(row);
    return result;
}
ArrayColumn ArrayColumn::ones(uint size) {
    vector<uint> data;
    for(int i = size - 1; i >= 0; i--)
        data.push_back(i);
    return ArrayColumn(size, std::move(data), ArrayColumn::trivial);
}
ArrayColumn::ArrayColumn(uint size): size(size)
{}
ArrayColumn::ArrayColumn(uint size, uint row): size(size), data({row})
{}
ArrayColumn::ArrayColumn(uint size, vector<uint> data_, sorting how_sorted):
    size(size), data(std::move(data_)) 
{
    switch (how_sorted) {
        case trivial:
            break;
    #ifndef ARRAY_COLUMNS_ASCENDING
        case unsorted: 
            std::sort(data.begin(), data.end(), std::greater<uint>{}); 
            break;
        case descending:
            break;
        case ascending:
            std::reverse(data.begin(), data.end());
            break;
    #else
        case unsorted: 
            std::sort(data.begin(), data.end(), std::less<uint>{}); 
            break;
        case ascending:
            break;
        case descending:
            std::reverse(data.begin(), data.end());
            break;
    #endif
    }
}
ArrayColumn ArrayColumn::get_rows(uint size, const vector<index_t>& lut) const {
    ArrayColumn result(size);
    get_rows(lut, result);
    return result;
}
void ArrayColumn::get_rows_consolidate(const vector<index_t>& lut, ArrayColumn &into) const {
    get_rows(lut, into);
}
void ArrayColumn::get_rows(const vector<index_t>& lut, ArrayColumn &into) const {
    for(uint i : data)
        if(lut[i] != no_pivot)
            into.data.push_back(lut[i]);
    #ifndef ARRAY_COLUMNS_ASCENDING
    std::sort(into.data.begin(), into.data.end(), std::greater<uint>{});
    #else
    std::sort(into.data.begin(), into.data.end(), std::less<uint>{});
    #endif
}
void ArrayColumn::reindex_rows(const vector<size_t> &lut) {
    for(auto &i : data)
        i = lut[i];
    #ifndef ARRAY_COLUMNS_ASCENDING
    std::sort(data.begin(), data.end(), std::greater<uint>{});
    #else
    std::sort(data.begin(), data.end(), std::less<uint>{});
    #endif
}
ArrayColumn& ArrayColumn::operator+=(const ArrayColumn& other) {
    assert(size == other.size);
    skip_vector<uint> result;
    result.reserve(data.size() + other.data.size());
    #ifndef ARRAY_COLUMNS_ASCENDING
    std::set_symmetric_difference(data.begin(), data.end(), other.data.begin(), other.data.end(), std::back_inserter(result.data), std::greater<uint>());
    #else
    std::set_symmetric_difference(data.begin(), data.end(), other.data.begin(), other.data.end(), std::back_inserter(result.data), std::less<uint>());
    #endif
    std::swap(data, result);
    return *this;
}
ArrayColumn& ArrayColumn::v_extend(uint top, uint bottom) {
    size += top + bottom;
    if(top > 0) {
        vector<uint> new_data;
        for(auto& i : data)
            i += top;
    }
    return *this;
}
ArrayColumn& ArrayColumn::v_extend(const ArrayColumn& other) {
    // data.front() is the pivot. So the pivot of the new column will be
    // the pivot of `other`. Therefore, increase all entries of `other` by
    // `this->size` and append `this->data` to `other.data`.
    #ifndef ARRAY_COLUMNS_ASCENDING
    vector<uint> result;
    result.reserve(data.size() + other.data.size());
    for(uint i : other.data)
        result.push_back(i + size);
    for(uint i : data)
        result.push_back(i);
    data = std::move(result);
    #else
    for(uint i : other.data)
        data.push_back(i + size);
    #endif
    size += other.size;
    return *this;
}
void ArrayColumn::clear() {
    data.clear();
}
void ArrayColumn::add_one(index_t row) {
    *this += ArrayColumn::eye(size, row);
}

bool ArrayColumn::operator==(const ArrayColumn& other) const {
    return size == other.size && data == other.data;
}
bool ArrayColumn::is_well_defined() const {
    bool in_bounds = std::all_of(data.begin(), data.end(), [&](uint i) { return i < size; });
    #ifdef ARRAY_COLUMNS_ASCENDING
    bool is_sorted = std::all_of(data.begin(), data.end(), [&](uint i) { return i < size; }) && std::is_sorted(data.begin(), data.end(), std::less<uint>());
    #else
    bool is_sorted = std::is_sorted(data.begin(), data.end(), std::greater<uint>());
    #endif
    return in_bounds && is_sorted;
}
index_t ArrayColumn::pivot(){
    #ifdef ARRAY_COLUMNS_ASCENDING
    return data.size() ? data.back() : no_pivot;
    #else
    return data.size() ? data.front() : no_pivot;
    #endif
}

index_t ArrayColumn::pop_pivot() {
    if(data.size()) {
        #ifndef ARRAY_COLUMNS_ASCENDING
            index_t result = data.front();
            data.pop_front();
            return result;
        #else
            index_t result = data.back();
            data.pop_back();
            return result;
        #endif
    } 
    else
        return no_pivot;
}
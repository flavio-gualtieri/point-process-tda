#include "HeapColumn.hpp"
#include <algorithm>
#include <cassert>
HeapColumn HeapColumn::ones(uint size) {
    priority_queue<uint> data;
    for(uint i = 0; i < size; i++)
        data.push(i);
    return HeapColumn(size, std::move(data));
}
HeapColumn::HeapColumn(int size): size(size)
{}
HeapColumn::HeapColumn(uint size, priority_queue<uint> data):
    size(size), data(std::move(data))
{}
HeapColumn::HeapColumn(uint size, std::vector<uint> entries, HeapColumn::sorting how_sorted): 
    size(size), data(entries)
{
    switch(how_sorted) {
        case trivial:
        case descending:
            break;
        case unsorted:
        case ascending:
            data.make_heap();
            break;
    }
}
HeapColumn::HeapColumn(uint size, uint row): size(size) {
    data.push(row);
}
HeapColumn HeapColumn::get_rows(uint size, const std::vector<index_t>& lut) const {
    HeapColumn result(size);
    get_rows(lut, result);
    return result;
}
void HeapColumn::get_rows(const std::vector<index_t>& lut, HeapColumn &into) const {
    for(uint i : data)
        if(lut[i] != no_pivot)
            into.data.push(lut[i]);
}
void HeapColumn::get_rows_consolidate(const std::vector<index_t>& lut, HeapColumn &into) && {
    for(;;){
        auto i = pop_pivot();
        if(i != no_pivot)
            into.data.push(lut[i]);
        else
            break;
    }
}
void HeapColumn::reindex_rows(const std::vector<size_t>& indices){
    for(auto &j : data)
        j = indices[j];
    data.make_heap();
}
HeapColumn HeapColumn::operator+(const HeapColumn& other) const {
    HeapColumn result = *this;
    result += other;
    return result;
}
HeapColumn& HeapColumn::operator+=(const HeapColumn& other) {
    assert(size == other.size);
    for(index_t entry : other.data)
        data.push(entry);
    return *this;
}
HeapColumn& HeapColumn::v_extend(uint top, uint bottom) {
    size += top + bottom;
    if(top > 0) {
        priority_queue<uint> new_data;
        for(auto i : data)
            new_data.push(i + top);
        data = std::move(new_data);
    }
    return *this;
}
HeapColumn& HeapColumn::v_extend(const HeapColumn& other) {
    for(auto i : other.data) {
        data.push(size + i);
    }
    size += other.size;
    return *this;
}
void HeapColumn::clear() {
    data.clear();
}
void HeapColumn::add_one(index_t row) {
    data.push(row);
}
index_t HeapColumn::pivot() {
    for(;;){
        if(data.empty())
            return no_pivot;
        else if((data.size() > 1 && data.q[0] == data.q[1]) || (data.size() > 2 && data.q[0] == data.q[2])){
            data.pop();
            data.pop();
        }
        else
            return data.top();
    }
}
index_t HeapColumn::pop_pivot() {
    auto i = pivot();
    if(i != no_pivot)
        data.pop();
    return i;
}
void HeapColumn::consolidate() {
    priority_queue<uint> consolidated_data;
    for(;;) {
        index_t entry = pop_pivot();
        if(entry == no_pivot)
            break;
        consolidated_data.push(entry);
    }
    swap(data, consolidated_data);
}
bool HeapColumn::is_zero() {
    return pivot() == no_pivot;
}
bool HeapColumn::is_well_defined() const {
    bool is_heap            = std::is_heap(data.begin(), data.end()),
         entries_in_bounds  = std::all_of(data.begin(), data.end(), [&](uint i) { return i < size; });
    return is_heap && entries_in_bounds;
}
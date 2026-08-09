#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <vector>
typedef unsigned int uint;
typedef int index_t;
#include <unordered_map>
template<typename Key, typename Value>
class unordered_map_wrapper : private std::unordered_map<Key, Value>{
    public:
    unordered_map_wrapper() = default;
    unordered_map_wrapper(size_t, Value default_value):
        default_value(default_value)
    {}
    void assign(size_t, Value default_value){
        this->clear();
        this->default_value = default_value;
    }
    Value get(Key index) const {
        auto i = this->find(index);
        if(i == this->end()) return default_value;
        else return i->second;
    }
    void set(Key index, Value value){
        std::unordered_map<Key, Value>::operator[](index) = value;
    }
    Value operator[](Key index) const{
        return get(index);
    }
    private:
    Value default_value;
};
// typedef std::vector<index_t> pivot_map_t;
typedef unordered_map_wrapper<size_t, index_t> pivot_map_t;

const index_t no_pivot = -1;

extern uint indirection_level;

#endif

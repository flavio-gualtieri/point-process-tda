
#ifndef SKIP_VECTOR_H
#define SKIP_VECTOR_H
#include <algorithm>
#include <iterator>
#include <vector>
#include "typedefs.hpp"
/** Vector with `pop_front()`.
 *
 * Adapter to std::vector that maintains a number of entries at the beginning to be skipped.
 * This allows to implement a method pop_front().
 */
template<typename T> class skip_vector{
    private:
    std::vector<T> data;
    uint skip=0;
    friend class ArrayColumn;

    public:
    typedef typename decltype(data)::iterator iterator;
    typedef typename decltype(data)::const_iterator const_iterator;

    skip_vector() = default;

    skip_vector(const skip_vector<T>& from): data(from.data) {};
    skip_vector(skip_vector<T>&& from): data(std::move(from.data)) {};
    skip_vector<T>& operator=(const skip_vector<T>& from){
        skip=0;
        data = from.data;
        return *this;
    }
    skip_vector<T>& operator=(skip_vector<T>&& from){
        skip=0;
        data = std::move(from.data);
        return *this;
    }

    skip_vector(std::vector<T> from): data(std::move(from)) {}
    skip_vector<T>& operator=(std::vector<T> from){
        skip = 0;
        data = std::move(from);
        return *this;
    }
    auto size() const {
        return data.size() - skip;
    }
    auto begin(){
        return data.begin() + skip;
    }
    auto begin() const {
        return data.begin() + skip;
    }
    auto end(){
        return data.end();
    }
    auto end() const {
        return data.end();
    }
    auto rbegin(){
        return data.rbegin();
    }
    auto rend(){
        return data.rend() - skip;
    }
    auto& front(){
        return data[skip];
    }
    auto& back(){
        return data.back();
    }
    auto back_inserter(){
        return std::back_inserter(data);
    }
    auto& operator[](size_t i){
        return data[i + skip];
    }
    void pop_back(){
        data.pop_back();
    }
    void pop_front(){
        ++skip;
    }
    void push_back(const T& t){
        data.push_back(t);
    }
    void push(const T& t){
        data.push_back(t);
    }
    void clear(){
        skip = 0;
        data.clear();
    }
    void reserve(size_t size){
        data.reserve(skip+size);
    }
    template<typename S> bool
    operator==(const S& other) const {
        return size() == other.size()
            && std::equal(begin(), end(), other.begin());
    }
    template< class InputIt >
    constexpr iterator insert(const_iterator pos, InputIt first, InputIt last ){
        return data.insert(pos, first, last);
    }
    void swap(skip_vector<T>& o){
        using std::swap;
        swap(skip, o.skip);
        swap(data, o.data);
    }
    uint size(){
        return data.size() - skip;
    }
    operator std::vector<T>() const & {
        return std::vector<T>(data.begin() + skip, data.end());
    }
    operator std::vector<T>() && {
        if(skip == 0)
            return std::move(data);
        else
            return std::vector<T>(data.begin() + skip, data.end());
    }
};

#endif

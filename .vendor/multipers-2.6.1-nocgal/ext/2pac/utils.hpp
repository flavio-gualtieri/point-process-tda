#ifndef UTILS_H
#define UTILS_H
// #include "matrices.hpp"
#include "typedefs.hpp"
#include <string>
#include <type_traits>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

/** Apples \p f to all members of \p rs. */
template <typename R, typename Function>
auto map(const std::vector<R> &rs, Function f){
    std::vector<decltype(f(rs[0]))> ts;
    std::transform(rs.begin(), rs.end(), std::back_inserter(ts), f);
    return ts;
}

template<class T>
T& unmove(T&& t) {
    return t;
}

/** Returns the indices necessary to sort a container.*/
template<typename T, typename Compare = std::less<T>>
std::vector<size_t> indirect_sort(const std::vector<T> v, Compare c = std::less<T>()){
    std::vector<size_t> result(v.size());
    std::iota(result.begin(), result.end(), 0);
    std::stable_sort(result.begin(), result.end(), [&](int i, int j){return c(v[i], v[j]);});
    return result;
}

template<class RandomIt, class Compare>
std::vector<size_t> indirect_sort(RandomIt first, RandomIt last, Compare comp){
    std::vector<size_t> result(last - first);
    std::iota(result.begin(), result.end(), 0);
    std::stable_sort(result.begin(), result.end(), [&](int i, int j){return comp(*(first+i), *(first+j));});
    return result;
}

/** Left inverse of a injective integer map.
 *
 * Computes a vector v such that v[p[i]]=i for all i. All v[j] for j not in the range of p are set to no_pivot.
 * \param p Injective map of unsigned integral type. Injectivity is not checked.
 */
template<typename T>
std::vector<index_t> inverse_permutation(std::vector<T> p) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    assert(std::all_of(p.begin(), p.end(), [&p](T i){return 0 <= i && i < p.size();}));
    std::vector<index_t> result(p.size(), no_pivot);
    for(uint j = 0; j < p.size(); j++){
        assert(result[p[j]] == no_pivot);
        result[p[j]] = j;
    }
    return result;
}

template<typename T>
std::vector<index_t> inverse_permutation(std::vector<T> p, T max) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    assert(std::all_of(p.begin(), p.end(), [max](T i){return 0 <= i && i < max;}));
    std::vector<index_t> result(max, no_pivot);
    for(uint j = 0; j < p.size(); j++){
        assert(result[p[j]] == no_pivot);
        result[p[j]] = j;
    }
    return result;
}

/** Inverse of a permutation.
 *
 * Bijectivity is not checked.
 */
template<typename T>
std::vector<size_t> inverse_permutation_inj(std::vector<T> permutation) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    assert(std::all_of(permutation.begin(), permutation.end(), [&permutation](T i){return 0 <= i && i < permutation.size();}));
    std::vector<size_t> result(permutation.size(),0);
    for(uint j = 0; j < permutation.size(); j++){
        assert(result[permutation[j]] == 0);
        result[permutation[j]] = j;
    }
    return result;
}

/** Retrieve indices of elements that satisfy predicate. */
template<typename Container, typename Predicate> 
std::vector<size_t> where(Container c, Predicate p){
    std::vector<size_t> result;
    size_t i = 0;
    for(auto it = c.begin(); it != c.end(); it++, i++)
        if(p(*it))
            result.push_back(i);
    return result;
}

/** Retrieve elements from container by indices. */
template<typename T, typename I> std::vector<T> get_elements(const std::vector<T>& v, const std::vector<I> &indices){
    assert(std::all_of(indices.begin(), indices.end(), [&](I i){return i < v.size();}));
    std::vector<T> result;
    result.reserve(indices.size());
    for(auto i : indices)
        result.push_back(v[i]);
    return result;
}

template<typename T, typename I> std::vector<T> get_elements(std::vector<T>&& v, const std::vector<I> &indices){
    assert(std::all_of(indices.begin(), indices.end(), [&](I i){return i < v.size();}));
    std::vector<T> result;
    result.reserve(indices.size());
    for(auto i : indices)
        result.push_back(std::move(v[i]));
    return result;
}

/** STL-compatible comparator for the lexicographic order on pairs. */
template<class L, class R, class CmpL=std::less<L>, class CmpR=std::less<R>>
struct lex_compare {
    lex_compare(CmpL cmp_l = CmpL{}, CmpR cmp_r = CmpR{}): cmp_l(cmp_l), cmp_r(cmp_r) {}
    bool operator()(const std::pair<L, R>& a, const std::pair<L, R>& b) const {
        return cmp_l(a.first, b.first) || (a.first==b.first && cmp_r(a.second, b.second));
    }
    private:
    CmpL cmp_l;
    CmpR cmp_r;

};

/** STL-compatible comparator for the colexicographic order on pairs. */
template<class L, class R, class CmpL=std::less<L>, class CmpR=std::less<R>>
struct colex_compare {
    colex_compare(CmpL cmp_l = CmpL{}, CmpR cmp_r = CmpR{}): cmp_l(cmp_l), cmp_r(cmp_r) {}
    bool operator()(const std::pair<L, R>& a, const std::pair<L, R>& b) const {
        return cmp_l(a.second, b.second) || (a.second==b.second && cmp_r(a.first, b.first));
    }
    private:
    CmpL cmp_l;
    CmpR cmp_r;

};

/** Sorts two vectors together as if they were a vectir of pairs.*/
template<typename T1, typename T2, typename Compare>
void sort_together(std::vector<T1>& v1, std::vector<T2>& v2, Compare c = std::less<std::pair<T1, T2>>()){
    assert(v1.size() == v2.size());
    std::vector<size_t> indices(v1.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(
        // std::execution::par_unseq,
        indices.begin(), indices.end(),
        [&](size_t i, size_t j){return c({v1[i], v2[i]}, {v1[j], v2[j]});}
    );
    // #pragma omp parallel
    {
        v1 = get_elements(v1, indices);
        v2 = get_elements(v2, indices);
    }
}

/** True if all elements of \p array are distinct. */
template<class T>
bool all_distinct(T array){
    std::sort(std::begin(array), std::end(array));
    return std::adjacent_find(std::begin(array), std::end(array)) == std::end(array);
}

/** Returns all indices i such that p(v[i]) is true. */
template<class V, class UnaryPredicate>
std::vector<size_t> find_all(const V& v, UnaryPredicate p){
    std::vector<size_t> result;
    result.reserve(v.size());
    for(size_t i = 0; i != v.size(); i++)
        if(p(v[i]))
            result.push_back(i);
    return result;
}

/** Used to signal end of input. */
struct EOI{};

/** Reads a T from stream. */
template<typename T> T read_stream(std::istream& stream){
    if(!stream) throw EOI();
    T to;
    stream.read((char*) &to, sizeof(T));
    return to;
}

template<typename T> void read_stream(std::istream& stream, T& to){
    if(!stream) throw EOI();
    stream.read((char*) &to, sizeof(T));
}

template<typename T, typename... Ts> void read_stream(std::istream& stream, T& t, Ts&... ts){
    read_stream(stream, t);
    read_stream(stream, ts...);
}

template<typename T> void write_stream(std::ostream& stream, T what){
    if(!stream) throw EOI();
    stream.write((char*) &what, sizeof(T));
}

template<typename T, typename... Ts> void write_stream(std::ostream& stream, T t, Ts... ts){
    write_stream(stream, t);
    write_stream(stream, ts...);
}

/** Convenience operators for ostreams */
template<class S, class T>
std::ostream& operator<< (std::ostream& out, const std::pair<S, T>& v);

template<class T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto i = v.begin(); i != v.end(); i++){
            if(i != v.begin())
                out << ", ";
            out << *i;
            
        }
        out << "]";
    }
    return out;
}

template<class S, class T>
std::ostream& operator<< (std::ostream& out, const std::pair<S, T>& v) {
    out << '(' << v.first << ", " << v.second << ")";
    return out;
}


#endif

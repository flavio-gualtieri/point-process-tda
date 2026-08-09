#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

#include <algorithm>
#include <utility>
#include <vector>

/** Sift down the element at \p start in the heap between \p first and \p last until it satisfies the heap condition. */
template<class I, class C> void sift_down(I first, I last, I start, C &&comp){
    using std::swap;
    auto len = last - first;
    auto child = start - first;
    for(;;){
        // left child exists
        child = 2 * child + 1;
        if(child >= len)
            return;
        auto child_i = first + child;
        // right child exists and is greater than left child 
        if(child + 1 < len && comp(*child_i, child_i[1])){
            child++;
            child_i++;
        }
        // heap order established
        if(comp(*child_i, *start))
            return;
        swap(*child_i, *start);
        start = child_i;
    }
}
template<class I> void sift_down(I first, I last, I start){
	sift_down(first, last, start, std::less<decltype(*first)>());
}
template<class I> void sift_down(I first, I last){
	sift_down(first, last, first, std::less<decltype(*first)>());
}
template<class I, class C> void sift_down(I first, I last, C &&comp){
	sift_down(first, last, first, std::move(comp));
}

/** Extends functionality of \c std::priority_queue. */
template<class T>
class priority_queue {
    public:
    priority_queue() = default;
    /// Heapify a vector.
    priority_queue(const std::vector<T>& q_): q(q_) {
        std::make_heap(q.begin(), q.end());
    }
    priority_queue(std::vector<T>&& q_): q(std::move(q_)) {
        std::make_heap(q.begin(), q.end());
    }
    /// Re-establish heap condition.
    void make_heap(){
        std::make_heap(q.begin(), q.end());
    }
    void push(const T& t){
        q.push_back(t);
        std::push_heap(q.begin(), q.end());
    }
    void pop(){
        std::pop_heap(q.begin(), q.end());
        q.pop_back();
    }
    const T& top() const {
        return q.front();
    }
    T& top(){
        return q.front();
    }
    /// See ::sift_down().
    void sift_down(){
        ::sift_down(q.begin(), q.end());
    }
    void clear(){
        q.clear();
    }
    void reserve(size_t size){
        q.reserve(size);
    }
    /// Iterators to the underlying vector.
    auto begin(){ return q.begin(); }
    auto begin() const { return q.begin(); }
    auto end(){ return q.end(); }
    auto end() const { return q.end(); }
    auto rbegin(){ return q.rbegin(); }
    auto rbegin() const { return q.rbegin(); }
    auto rend(){ return q.rend(); }
    auto rend() const { return q.rend(); }
    size_t size() const {
        return q.size();
    }
    bool empty(){
        return q.empty();
    }

    friend void swap(priority_queue<T>& a, priority_queue<T>& b){
        swap(a.q, b.q);
    }
    operator std::vector<T>() && {
        return std::move(q);
    }
    operator const std::vector<T>&() const {
        return q;
    }
    std::vector<T> q;

};
#endif

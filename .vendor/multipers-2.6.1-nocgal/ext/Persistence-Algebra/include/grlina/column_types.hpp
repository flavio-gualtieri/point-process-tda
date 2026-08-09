/**
 * @file column_types.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.1
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once

#ifndef COLUMN_TYPES_HPP
#define COLUMN_TYPES_HPP

#include <vector>
#include <set>
#include <random>
#include <boost/dynamic_bitset.hpp>

namespace graded_linalg {


template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;
template <typename T>
using pair = std::pair<T, T>;
template <typename T>
using set = std::set<T>;
using bitset = boost::dynamic_bitset<>;


static std::mt19937& get_rng() {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    return gen;
}

template <typename COLUMN, typename index>
struct Column_traits {

    // Define these for your own type which designates a column of a matrix

    /**
     * @brief Changes w to w+v
     */
    static void add_to(const COLUMN& v, COLUMN& w);

    /**
     * @brief Returns true, if the i-th entry of a is non-zero.
     */
    static bool is_nonzero_at(const COLUMN& v, index i);

    /**
     * @brief Returns the index of the last non-zero entry in a or -1
     */
    static index last_entry_index(const COLUMN& v);

    /**
     * @brief Compares v and w.
     */
    static bool is_equal(const COLUMN& v, const COLUMN& w);

    /**
     * @brief Returns true if v is the zero vector.
     */
    static bool is_zero (const COLUMN& v);

    /**
     * @brief Flips the j-th entry of a
     */
    static void set_entry(COLUMN& v, index j);

    /**
     * @brief Computes the scalar product of v and w.
     */
    static bool scalar_product(const COLUMN& v, const COLUMN& w);

    /**
     * @brief Returns the i-th standard vector of length n.
     */
    static COLUMN get_standard_vector(index i, index n);

    /**
     * @brief Returns a random vector of length length with perc percent non-zero entries.
     */
    static COLUMN get_random_vector(index length, float rate);

    static COLUMN get_zero_vector(index length);

}; // Column_traits


/**
 * @brief Returns a+b when treated as F_2 vectors.
 * Needs them to be ordered.
 * 
 */
template <typename index>
vec<index> operator+(const vec<index>& a, vec<index>& b) {
    vec<index> c;
    auto a_it = a.begin();
    auto b_it = b.begin();
    while(a_it!=a.end() || b_it!=b.end()) {
        if(a_it==a.end()) {
            c.push_back(*b_it);
            b_it++;
            continue;
        }
        if(b_it==b.end()) {
            c.push_back(*a_it);
            a_it++;
            continue;
        }
        if(*a_it<*b_it) {
            c.push_back(*a_it);
            a_it++;
        } else if(*a_it>*b_it) {
            c.push_back(*b_it);
            b_it++;
        } else { // *a_it==*b_it
            assert(*a_it==*b_it);
            a_it++;
            b_it++;
        }      
    }
    return c;
}

/**
 * @brief Returns a+b when treated as F_2 vectors.
 * Needs them to be ordered in reverse.
 * 
 */
template <typename index>
vec<index> rev_add(vec<index>& a, vec<index>& b) {
    vec<index> c;
    auto a_it = a.begin();
    auto b_it = b.begin();
    while(a_it != a.end() || b_it != b.end()) {
        if(a_it==a.end()) {
            c.push_back(*b_it);
            b_it++;
            continue;
        }
        if(b_it == b.end()) {
            c.push_back(*a_it);
            a_it++;
            continue;
        }
        if(*a_it > *b_it) {
            c.push_back(*a_it);
            a_it++;
        } else if(*a_it < *b_it) {
            c.push_back(*b_it);
            b_it++;
        } else { // *a_it==*b_it
            assert(*a_it==*b_it);
            a_it++;
            b_it++;
        }      
    }
    return c;
}

/**
* @brief sparse column addition over F_2 for vectors whose entries are stored in reverse order. 
*/
template <typename index>
void rev_add_to(vec<index>& a, vec<index>& b) {
    b = rev_add(a, b);
}

template <typename index>
struct Column_traits<vec<index>, index> {

    static void add_to(const vec<index>& v, vec<index>& w) {
        w = v + w;
    }

    static bool is_nonzero_at(const vec<index>& v, index i) {
        return std::binary_search(v.begin(), v.end(), i);
    }

    static index last_entry_index(const vec<index>& v) {
        if(v.size() == 0) {
            return -1;
        } else {
            return v.back();
        }
    }


    static bool scalar_product(const vec<index>& v,const vec<index>& w){
        auto it_v = v.begin();
        auto it_w = w.begin();
        index count = 0;

        while (it_v != v.end() && it_w != w.end()) {
            if (*it_v < *it_w) {

                ++it_v;
            } else if (*it_w < *it_v) {

                ++it_w; 
            } else {

                ++it_v;
                ++it_w;
                count++;  
            }
        }
        return count % 2 != 0;
    }

    static void set_entry(vec<index>& v, index j) {
        if(last_entry_index(v) < j) {
            v.push_back(j);
        } else {
            auto it = std::lower_bound(v.begin(), v.end(), j);
            if(*it != j) {
                v.insert(it, j);
            } else {
                v.erase(it);
            }
        }  
    }

    static bool is_equal(const vec<index>& v, const vec<index>& w) {
        return v == w;
    }

    static bool is_zero(const vec<index>& v) {
        return v.empty();
    }

    static vec<index> get_standard_vector(index i, index n) {
        return vec<index>{i};
    }

    static vec<index> get_random_vector(index length, float rate) {
        vec<index> v;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        auto& gen = get_rng();
        
        for(index i = 0; i < length; i++) {
            if(dist(gen) < rate) {
                v.push_back(i);
            }
        }
        return v;
    }


}; // Column_traits<vec<index>>
        


// Helper-functions for sets


template <typename index>
set<index> operator+(set<index>& a, set<index>& b) {
    set<index> result;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));
    return result;
}


template <typename index>
struct Column_traits<set<index>, index> {

    static void add_to(set<index>& v, set<index>& w) {
        auto it_v = v.begin();
        auto it_w = w.begin();
    
        while (it_v != v.end() && it_w != w.end()) {
            if (*it_v < *it_w) {
                it_w = w.insert(it_w, *it_v);
                ++it_v;
            } else if (*it_w < *it_v) {
                ++it_w;
            } else {
                it_w = w.erase(it_w);
                ++it_v;
            }
        }
    
        w.insert(it_v, v.end());
    }

    static bool is_nonzero_at(set<index>& v, index i) {
        return (v.find(i) != v.end());
    }

    static index last_entry_index(const set<index>& v) {
        if(v.size() == 0){
            return -1;
        } else {
            return *v.rbegin();
        }
    }

    static bool scalar_product(set<index>& v, set<index>& w) {
        //TO-DO: This isn't efficent, but it works for now.
        set<index> intersection;
        std::set_intersection(v.begin(), v.end(), w.begin(), w.end(), std::inserter(intersection, intersection.begin()));
        return intersection.size() % 2 == 1;
    }

    static void set_entry(set<index>& v, index j) {
        auto ins = v.insert(j);
        if(ins.second == false){
            v.erase(ins.first);
        }   
    }

    static bool is_equal(set<index>& v, set<index>& w) {
        return v == w;
    }

    static set<index> get_standard_vector(index i, index n) {
        return set<index>{i};
    }

    static set<index> get_random_vector(index length, float rate) {
        set<index> result;
        for(index i = 0; i < length; i++){
            if(static_cast<float>(rand()) / RAND_MAX < rate) {
                result.insert(result.end(), i);
            }
        }
        return result;
    }

    static set<index> get_zero_vector(index length) {
        return set<index>();
    }

}; // Column_traits<set<index>>

// Helper-functions for bitsets


template <typename index>
struct Column_traits<bitset, index> {

    static void add_to(bitset& v, bitset& w) {
        assert(v.size() == w.size());
        w ^= v;
    }

    static bool is_nonzero_at(bitset& v, index i) {
        return v[i];
    }

    static index last_entry_index(const bitset& v) {
        for (int i = v.size() - 1; i >= 0; --i) {
            if (v[i]) {
                return i;
            }
        }
        return -1;
    }

    static bool scalar_product(bitset& v, bitset& w) {
        return (v & w).count() % 2;
    }

    static void set_entry(bitset& v, index j) {
        v.flip(j);
    }

    static bool is_equal(bitset& v, bitset& w) {
        return v == w;
    }

    static bitset get_standard_vector(index i, index n) {
        return bitset(n, 0).set(i);
    }

    static bitset get_random_vector(index length, float rate) {
        bitset result(length, 0);
        for(index i = 0; i < length; i++) {
            if( static_cast<float>(rand()) / RAND_MAX < rate) {
                result.set(i);
            }
        }
        return result;
    }

    static bitset get_zero_vector(index length) {
        return bitset(length, 0);
    }

}; // Column_traits<bitset>

}
#endif // COLUMN_TYPES_HPP
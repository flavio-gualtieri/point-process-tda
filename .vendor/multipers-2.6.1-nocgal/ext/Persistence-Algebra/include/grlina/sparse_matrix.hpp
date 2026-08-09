/**
 * @file sparse_matrix.hpp
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


#include "grlina/column_types.hpp"
#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <iostream>
#include <vector>
#include <grlina/matrix_base.hpp>
#include <grlina/dense_matrix.hpp>
#include <omp.h>
#include <boost/dynamic_bitset.hpp>
#include <cmath>

namespace graded_linalg {



// Helperfunctions for index vectors and sparse matrices

    template <typename index>
    struct pair_hash {
        std::size_t operator()(const std::pair<index, index>& p) const {
            auto hash1 = std::hash<index>{}(p.first);
            auto hash2 = std::hash<index>{}(p.second);
            return hash1 ^ (hash2 << 1); // or use another method?
        }
    };


    /**
     * @brief Returns a copy of the target vector with the entries at the indices given in the mask.
     * 
     * @tparam index 
     * @tparam T 
     * @param target 
     * @param mask 
     * @return vec<T>   
     */
    template <typename index, typename T>
    vec<T> vec_restriction(vec<T>& target, vec<index>& mask){
        vec<T> result;
        for(index i : mask){
            result.push_back(target[i]);
        }
        return result;
    }

    /**
     * @brief Inverse to vec_restriction. Returns the indices in "target" of the elements in subset. 
     * Only works if both are sorted.
     * @tparam index 
     * @tparam T 
     * @return vec<T>   
     */
    template <typename index, typename T>
    vec<index> get_index_vector(vec<T> target, vec<T> subset){
        vec<index> result = vec<index>();
        auto itS = subset.begin();
        for(index i = 0; i < target.size() ; i++){
            if(itS != subset.end() && *itS == target[i]){
                result.push_back(i);
                itS++;
            }
        }
        assert(itS == subset.end() && "Not all elements of the subset were found in the target");
        return result;
    }

    /**
     * @brief Delets all entries of v at the indices given while preserving the order of v.
     * 
     * @tparam T 
     * @tparam index 
     * @param v 
     * @param indices 
     */
    template<typename T, typename index>
    void vec_deletion(std::vector<T>& v, const std::vector<index>& indices) {

        assert(std::is_sorted(indices.begin(), indices.end()));

        index read = 0;
        index write = 0;
        index j = 0;
        index n = v.size();
        index m = indices.size();

        while (read < n) {
            if (j < m && read == indices[j]) {
                ++read;
                ++j;
            } else {
                if (read != write) {
                    v[write] = std::move(v[read]);
                }
                ++read;
                ++write;
            }
        }

        v.resize(write);
    }

    /**
     * @brief Attention: This might be slower then just counting parallelized! Returns the xor of all vectors whose indices are in the mask, generalizing the approach of add_to/ + operator. 
     * In essence this computes the product of the matrix with the column vector given by the mask over GF(2)
     * 
     * @tparam index 
     * @param matrix
     * @param mask
     *  
     * @return vec<index> 
     */
    template <typename index>
    vec<index> vectorXORMulti(const vec<vec<index>>& matrix, const vec<index>& mask) {
        vec<index> result;

        int n = mask.size();
        if (n == 1){
            return matrix[mask[0]];
        }

        // Initialize iterators for all vectors
        std::vector<typename vec<index>::const_iterator> iterators;
        for (index i = 0; i < n; ++i) {
            iterators.push_back(matrix[mask[i]].begin());
        }

        // Iterate through all vectors simultaneously
        while (true) {
            bool allEnd = true;
            index minVal = std::numeric_limits<index>::max();
            int minCount = 0;

            // Find the minimum value among all iterators and count occurrences
            for (size_t i = 0; i < n; ++i) {
                if (iterators[i] != matrix[mask[i]].end()) {
                    allEnd = false;
                    index val = *iterators[i];
                    if (val < minVal) {
                        minVal = val;
                        minCount = 1;
                    } else if (val == minVal) {
                        ++minCount;
                    }
                }
            }

            if (allEnd) {
                break;
            }

            // If the number of iterators pointing to the minimum value is odd, include it in the result
            if (minCount % 2 == 1) {
                result.push_back(minVal);
            }

            // Increment iterators pointing to the minimum value
            for (size_t i = 0; i < n; ++i) {
                if (iterators[i] != matrix[mask[i]].end() && *iterators[i] == minVal) {
                    ++iterators[i];
                }
            }
        }

        return result;
    }

    /**
     * @brief Computes A*b over F_2. Only somewhat fast if A is sparse.
     * Returns the xor of all vectors whose indices are in the mask, generalizing the approach of Michaels "+" operator. 
     * 
     * @tparam index 
     * @param mask
     * @return vec<index> 
     */
    template <typename index>
    vec<index> vectorXORMulti(const vec<vec<index>>& matrix, const bitset& mask) {
        vec<index> result;

        assert(mask.size() == matrix.size());

        // Initialize iterators for all vectors
        std::vector<typename vec<index>::const_iterator> iterators;
        std::vector<typename vec<index>::const_iterator> endIterators;
        for (int i = mask.find_first(); i != -1; i = mask.find_next(i)){
            iterators.push_back(matrix[i].begin());
            endIterators.push_back(matrix[i].end());
        }
        // Then count mod 2
        while (true) {
            bool allEnd = true;
            index minVal = std::numeric_limits<index>::max();
            int minCount = 0;

            // Find the minimum value among all iterators and count occurrences
            for (int i = 0; i < iterators.size(); ++i) {
                if (iterators[i] != endIterators[i]) {
                    allEnd = false;
                    index val = *iterators[i];
                    if (val < minVal) {
                        minVal = val;
                        minCount = 1;
                    } else if (val == minVal) {
                        ++minCount;
                    }
                }
            }

            if (allEnd) {
                break;
            }

            // If the number of iterators pointing to the minimum value is odd, include it in the result
            if (minCount % 2 == 1) {
                result.push_back(minVal);
            }

            // Increment iterators pointing to the minimum value
            for (int i = 0; i < iterators.size(); ++i) {
                if (iterators[i] != endIterators[i] && *iterators[i] == minVal) {
                    ++iterators[i];
                }
            }
        }

        return result;
    }
    

/**
 * @brief Erases i from v and returns true if i was found and erased.
 * 
 * @tparam index 
 * @param v 
 * @param i 
 * @return true 
 * @return false 
 */
template <typename index>
bool erase_from_sorted_vector_bool(vec<index>& v, index i){
    auto it = std::lower_bound(v.begin(), v.end(), i);
    if(it != v.end() && *it == i){
        v.erase(it);
        return true;
    }
    return false;
}

/**
 * @brief Tries to delete i from v.
 * 
 * @tparam index 
 * @param v 
 * @param i 
 */
template <typename index>
void erase_from_sorted_vector(vec<index>& v, index i){
    auto it = std::lower_bound(v.begin(), v.end(), i);
    if(it != v.end() && *it == i){
        v.erase(it);
    }
}

/**
 * @brief Removes the intersection of v and w from v. Both must be sorted
 * 
 * @tparam index 
 * @param v 
 * @param w 
 */
template <typename index>
void remove_intersection(vec<index>& v, const vec<index>& w) {
    assert(std::is_sorted(v.begin(), v.end()));
    assert(std::is_sorted(w.begin(), w.end()));
    index i = 0; // Pointer for v
    index j = 0; // Pointer for w
    index k = 0; // Position to write in v

    while (i < v.size() && j < w.size()) {
        if (v[i] < w[j]) {
            v[k++] = v[i++];
        } else if (v[i] > w[j]) {
            ++j;
        } else { // v[i] == w[j], skip v[i]
            ++i;
            ++j;
        }
    }

    // Copy any remaining elements from v (which are greater than all in w)
    while (i < v.size()) {
        v[k++] = v[i++];
    }

    v.resize(k); // Shrink v to keep only the unique values
}

template <typename index>
void insert_into_sorted_vector(vec<index>& v, index i){
    auto it = std::lower_bound(v.begin(), v.end(), i);
    v.insert(it, i);
}


/**
 * @brief This function is used to delete rows in a LOC sparse matrix. 
 * It creates a map which maps the old indices to the new indices.
 * 
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index> 
 */
template <typename index>
std::unordered_map<index, index> shiftIndicesMap(const vec<index>& indices) {
    std::unordered_map<index, index> indexMap;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indexMap[indices[i]] = i;
    }
    return indexMap;
}

/**
 * @brief This function is used to delete rows in a LOC sparse matrix. 
 * It creates a map which maps the old indices to the new indices.
 * 
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index> 
 */
template <typename index>
std::unordered_map< std::pair<index,index>, index, pair_hash<index> > pair_to_index_map(const vec<std::pair<index, index>>& index_pairs) {
    std::unordered_map<std::pair<index, index>, index, pair_hash<index> > indexMap;
    for (std::size_t i = 0; i < index_pairs.size(); ++i) {
        indexMap[index_pairs[i]] = i;
    }
    return indexMap;
}

/**
 * @brief Parallelized function to apply a transformation to a vector of indices.
 *
 * @param target
 * @param indexMap
 * @param needsNoDeletion If the target vector only contains indices which are in the indexMap, this can be set to true.
 */
template <typename index>
void apply_transformation(vec<index>& target, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion = false) {
    if (!needsNoDeletion) {
#pragma omp parallel for
        for (index i = static_cast<index>(target.size()) - 1; i >= 0; --i) {
            const index& element = target[i];
#pragma omp critical
            {
                if (indexMap.find(element) == indexMap.end()) {
                    target.erase(target.begin() + i);
                }
            }
        }
    }

#pragma omp parallel for
    for (index i = 0; i < target.size(); ++i) {
        target[i] = indexMap.at(target[i]);
    }
}

/**
 * @brief Apply a transformation given by a vector to a vector of indices.
 *
 * @param target
 * @param index_vector
 */
template <typename index>
void apply_transformation(vec<index>& target, const vec<index>& index_vector) {
    for (index i = 0; i < target.size(); ++i) {
        target[i] = index_vector[target[i]];
    }
}

/**
 * @brief Re-indxes the entries of v by subtracting the number of entries in w which are smaller than v[i].
 * 
 * @tparam index 
 * @param v 
 * @param w 
 */
template<typename index>
void re_index_non_ocurrences(std::vector<index>& v, const std::vector<index>& w) {
    assert(std::is_sorted(v.begin(), v.end()));
    assert(std::is_sorted(w.begin(), w.end()));
    index j = 0;
    index m = w.size();
    for (index& a : v) {
        while (j < m && w[j] < a) {
            ++j;
        }
        a -= j;
    }
}

/**
 * @brief Re-indexes the entries of S after having deleted the entries in indices_to_delete.
 * 
 * @tparam index 
 * @param S 
 * @param indices_to_delete 
 */
template<typename index>
void re_index_matrix(array<index>& S, const vec<index>& indices_to_delete){
    #pragma omp parallel for
    for(index i = 0; i < S.size(); ++i){
        re_index_non_ocurrences(S[i], indices_to_delete);
    }
}

/**
 * @brief Parallelised function to change a sparse matrix by applying the indexMap to each entry.
 *
 * @param S
 * @param indexMap
 */
template <typename index>
void transform_matrix(array<index>& S, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion) {
#pragma omp parallel for
    for (index i = 0; i < S.size(); ++i) {
        apply_transformation(S[i], indexMap, needsNoDeletion);
    }
}

/**
 * @brief Parallelised function to re-index a sparse matrix. An entry j is changed to index_vector[j].
 *
 * @param S
 * @param indexMap
 */
template <typename index>
void transform_matrix(array<index>& S, const vec<index>& index_vector) {
#pragma omp parallel for
    for (index i = 0; i < S.size(); ++i) {
        apply_transformation(S[i], index_vector);
    }
}

/**
 * @brief For a vector of integers, removes any even number of consecutive multiples. 
 * If the vector is sorted this is mod 2 reduction.
 * 
 * @param v 
 */
template <typename index>
void convert_mod_2(vec<index>& v){
    if(v.size() <= 1){
        return;
    }
    vec<index> result = vec<index>();
    index counter = 1;
    auto it_prev = v.begin();
    for(auto it = v.begin() + 1; it != v.end(); it++, it_prev++){
        if(*it == *it_prev){
            counter++;
        } else {
            if(counter % 2 == 1){
                result.push_back(*it_prev);
            }
            counter = 1;
        }
    }
    if(counter % 2 == 1){
        result.push_back(v.back());
    }
    v = result;
}

/**
 * @brief Checks if the entries of the vectors are strictly increasing.
 * 
 * @tparam index 
 * @param v 
 * @return true 
 * @return false 
 */
template<typename index>
bool is_sorted(vec<index>& v){
    for(index i = 1; i < v.size(); i++){
        if(v[i-1] >= v[i]){
            return false;
        }
    }
    return true;
}

/**
 * @brief Every column is stored as a list of non-zero entries.
 * 
 */
template <typename index>
struct SparseMatrix : public MatrixUtil<vec<index>, index, SparseMatrix<index>>{

    // Stores the rows (sometimes in reverse order!).
    vec<vec<index>> _rows;
    // So that the rows are not computed multiple times by accident.
    bool rows_computed;


    protected:
        SparseMatrix& assign(const SparseMatrix& other) {
            MatrixUtil<vec<index>, index, SparseMatrix<index>>::assign(other); // Call base
            _rows = other._rows;
            rows_computed = other.rows_computed;
            return *this;
        }
    
        SparseMatrix& assign(SparseMatrix&& other) {
            MatrixUtil<vec<index>, index, SparseMatrix<index>>::assign(std::move(other));
            _rows = std::move(other._rows);
            rows_computed = other.rows_computed;
            return *this;
        }
    
    public:
        SparseMatrix& operator=(const SparseMatrix& other) {
            if (this != &other)
                assign(other);
            return *this;
        }
    
        SparseMatrix& operator=(SparseMatrix&& other) {
            if (this != &other)
                assign(std::move(other));
            return *this;
        }


    SparseMatrix() : MatrixUtil<vec<index>, index, SparseMatrix<index>>() { rows_computed = false; }

    SparseMatrix(index m) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(m) {rows_computed = false;}

    SparseMatrix(index m, index n) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(m, n) {rows_computed = false;}

    SparseMatrix(const SparseMatrix& other) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(other)  {}

    SparseMatrix(SparseMatrix&& other) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(std::move(other))  {}

    SparseMatrix(index m, index n, const array<index>& data) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(m, n, data) {rows_computed = false;}
    
    SparseMatrix(index m, index n, const std::string& type, const index percent = -1) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(m, n, type, percent) {rows_computed = false;}

    SparseMatrix(index n, vec<index> indicator) : MatrixUtil<vec<index>, index, SparseMatrix<index>>(n, indicator) {rows_computed = false;}

    /**
     *   To treat a sparse matrix as a big vector for reduction we will need this function.
     *   In this way SparseMatrix becomes itself a sparse column type.
     */
    index last_entry_index() const{
        for(index i = this->get_num_cols()-1; i >= 0; i-- ){
            index p = this->col_last(i);
            if( p != -1){
                return i*this->get_num_rows() + p;
            }
        }
        return -1;
    }

    bool is_nonzero_at(index p){
        index col = p / this->get_num_cols();
        index row = p % this->get_num_cols();
        return this->is_nonzero_entry(col, row);
    }

    /**
     * @brief Sorts the entries of a column
     * 
     * @param i 
     */
    void sort_column(index i){
        std::sort(this->data[i].begin(), this->data[i].end());
    }
    
    /**
     * @brief Sorts all columns
     * 
     */
    void sort_data(){
        assert(this->num_cols == this->data.size());
        for(index i = 0; i < this->num_cols; i++){
            sort_column(i);
        }
    }

    index num_of_entries(){
        index num = 0;
        for(index i = 0; i < this->get_num_cols(); i++){
            num+= this->data[i].size();
        }
        return num;
    }

    /**
     * @brief Computes all rows from column data in reverse order.
     * 
     */
    void compute_revrows() {
        _rows.clear();
        for(index i = this->num_cols - 1; i >= 0 ; i--) {
            for(index j : this->data[i]) {
                if(j >= _rows.size()) {
                    _rows.resize(j+1);
                }
                _rows[j].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief Compute all rows in forward order.
     * 
     */
    void compute_rows_forward(){
        assert(this->num_rows > 0);
        _rows.clear();
        _rows = vec<vec<index>>(this->num_rows, vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[j].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief Computes all rows in forward order given a list of row indices which can appear shifted by the col_indices.
     * 
     * @param row_indices 
     * @param col_indices 
     * 
     */
    void compute_rows_forward(const vec<index>& row_indices, const vec<index>& col_indices) {

        auto row_map = shiftIndicesMap(row_indices);
        _rows.clear();
        _rows = vec<vec<index>>(row_indices.size(), vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                index r = row_map[j];
                _rows[r].push_back(col_indices[i]);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief   Computes all rows in forward order given a list of row indices which can appear without shift.
     * 
     * @param row_indices 
     */
    void compute_rows_forward(const vec<index>& row_indices) {

        auto row_map = shiftIndicesMap(row_indices);
        _rows.clear();
        _rows = vec<vec<index>>(row_indices.size(), vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[row_map[j]].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief   Computes all rows in forward order given a map from entries to row indices as a vector
     * 
     * @param row_indices 
     */
    void compute_rows_forward_map(const vec<index>& row_map){
        _rows = vec<vec<index>>(_rows.size(), vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[row_map[j]].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief   Computes all rows in forward order given a map from entries to row indices as a vector
     * 
     * @param row_indices 
     */
    void compute_rows_forward_map(const vec<index>& row_map, index size){
        _rows = vec<vec<index>>(size, vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[row_map[j]].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief Computes the columns from the rows if the rows contain only 0..num_cols .
     * 
     * @param col_indices 
     */
    void compute_columns_from_rows() {
        this->data.clear();
        this->data.resize(this->num_cols);
        for(index i = 0; i < _rows.size(); i++) {
            for(index j : _rows[i]) {
                this->data[j].push_back(i);
            }
        }
    }

    /**
     * @brief Computes the columns from the rows if the rows contain only indices in col_indices.
     * 
     * @param col_indices 
     */
    void compute_columns_from_rows(vec<index>& col_indices, vec<index>& row_indices) {
        auto shiftMap = shiftIndicesMap(col_indices);
        this->data.clear();
        this->data.resize(this->num_cols);
        for(index i = 0; i < _rows.size(); i++) {
            for(index j : _rows[i]) {
                this->data[shiftMap[j]].push_back(row_indices[i]);
            }
        }
    }

    /**
     * @brief Computes the columns from the rows if the rows contain only 0..num_cols .
     * 
     * @param col_indices 
     */
    void compute_columns_from_rows(vec<index>& row_indices) {
        this->data.clear();
        this->data.resize(this->num_cols);
        for(index i = 0; i < _rows.size(); i++) {
            for(index j : _rows[i]) {
                this->data[j].push_back(row_indices[i]);
            }
        }
    }


    /**
     * @brief If the entries of data are not from 1..n, but lie in row_indices, then this function
     * normalises those entries back to 1..n.
     * 
     * @param col_indices 
     * @param row_indices 
     */
    void compute_normalisation(const vec<index>& row_indices) {
        auto row_map = shiftIndicesMap(row_indices);
        transform_matrix(this->data, row_map, true);
    }

    /**
     * @brief Deletes the rows given by row_indices; row_indices must be sorted!
     * 
     * @param row_indices 
     */
    void delete_rows(const vec<index>& row_indices_to_remove) {
        assert(std::is_sorted(row_indices_to_remove.begin(), row_indices_to_remove.end()));
        if(row_indices_to_remove.empty()){
            return;
        }
        vec<index> remaining_rows = vec<index>();
        for(index i = 0; i < this->num_cols; i++) {
            remove_intersection(this->data[i], row_indices_to_remove);
        }
        auto it = row_indices_to_remove.begin();
        for(index j = 0; j < this->num_rows; j++) {
            if(it != row_indices_to_remove.end() && j == *it){
                it++;
            } else {
                remaining_rows.push_back(j);
            }
        }
        compute_normalisation(remaining_rows);
        this->num_rows = remaining_rows.size();
    }

    /**
     * @brief Deletes the columns given by col_indices; col_indices must be sorted!
     * 
     * @param col_indices_to_remove 
     */
    void delete_columns(const vec<index>& col_indices_to_remove) {
        assert(std::is_sorted(col_indices_to_remove.begin(), col_indices_to_remove.end()));
        if(col_indices_to_remove.empty()){
            return;
        }
        vec_deletion(this->data, col_indices_to_remove);
        this->num_cols = this->data.size();
    }


    void compute_normalisation_with_pivots(const vec<index>& row_indices) {
        assert(row_indices.size() == this->num_rows);
        auto row_map = shiftIndicesMap(row_indices);
        transform_matrix(this->data, row_map, true);
        // Create a new map with updated keys
        this->set_pivots_without_reducing();
    }

    void transform_data(const std::unordered_map<index, index>& indexMap) {
        transform_matrix(this->data, indexMap, true);
    }


    void transform_data(const vec<index>& index_vector){
        transform_matrix(this->data, index_vector);
    }

    /**
     * @brief Prints the _row list.
     * 
     */
    void print_rows(){
        for(index i = 0; i < _rows.size(); i++){
            std::cout << "Row " << i << ": ";
            for(index j : _rows[i]){
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief Computes a row from column data in reverse order.
     * 
     * @param i 
     * @param row 
     */
    void compute_row(index i, vec<index>& row) {
        row.clear();
        for(index j = this->num_cols - 1; j >= 0; j--) {
            if(is_nonzero_at(this->data[j], i)) {
                row.push_back(j);
            }
        }
    }

    /**
     * @brief Copy a row from _rows
     * 
     * @param i 
     * @param row 
     */
    void get_row_copy(index i, vec<index>& row) {
      row.clear();
      std::copy(this->_rows[i].begin(),this->_rows[i].end(),std::back_inserter(row));
      std::sort(row.begin(),row.end());
    }    

    bool is_sorted_sparse(){
        for(auto col : this->data){
            if(!is_sorted<index>(col)){
                return false;
            }
        }
        return true;
    }

    // Adds the i-th row to the j-th. 
    void fast_row_op(index i, index j){
        Column_traits<vec<index>, index>::add_to(this->_rows[i], this->_rows[j]);
    }

    void row_op_with_update(index i, index j){
        fast_row_op(i, j);
        for(index k : this->_rows[i]){
            this->set_entry(k, j);
        }
    }

    // Adds the i-th row to the j-th when rows are stored in reverse order.
    // also updates the columns, but without sorting or removing duplicates.
    void fast_rev_row_op(index i, index j){
        assert(i != j);
        for(index k : this->_rows[i]){
            this->data[k].push_back(j);
        }
        rev_add_to(this->_rows[i], this->_rows[j]);
    }

    // Adds the i-th row to the j-th. This is so expensive because of the reindexing, we might want to use some list type instead.
    // Not finished yet.
    void row_op_on_cols(index i, index j) {
        for(index k = 0; k < this->num_cols; k++) {
            if( this->is_nonzero_entry(k,i) ) {
                this->set_entry(k,j);
            }
        }
    }

    /**
     * @brief Deletes the last entry of each column. This is used in the cokernel computation.
     * 
     */
    void delete_last_entries(){
        #pragma omp parallel for
        for(std::size_t i = 0; i < this->data.size(); ++i) {
            vec<index>& c = this->data[i];
            #pragma omp critical
            if (!c.empty()) {
             c.pop_back();  // Delete the last entry if the column is not empty
            }
        }
    }
    
    /**
     * @brief Deletes either the last threshold rows of the matrix or everything after index threshold
     * 
     * @param threshold 
     */
    void cull_columns(const index& threshold, bool from_end){
        if(from_end){
            this->set_num_rows(this->get_num_rows() - threshold);
        } else {
            this->set_num_rows(threshold);
        }

        for (index j = 0; j < this->num_cols; j++){
            while(this->col_last(j) >= this->num_rows){
                this->data[j].pop_back();
            }
        }

    };

    

    /**
    * @brief Returns a copy with only the columns at the indices given in colIndices.
    * 
    * @param colIndices 
    * @return sparseMatrix 
    */
    SparseMatrix restricted_domain_copy(vec<index>& colIndices) const {
        for(index i : colIndices){
            assert(i < this->num_cols);
        }
        SparseMatrix result(colIndices.size(), this->num_rows);
        for(index i = 0; i < colIndices.size(); i++){
            result.data.push_back(this->data[colIndices[i]]);
        }
        return result;
    }



    /**
    * @brief Returns the transpose.
    * 
    * @return sparseMatrix 
    */
    SparseMatrix transposed_copy() const {
        SparseMatrix result(this->num_rows, this->num_cols);
        result.data.resize(result.num_cols);
        for(index i=0;i<this->num_cols;i++) {
            for(index j : this->data[i]) {
                result.data[j].push_back(i);
            }
        }
        return result;
    }


    /**
     * @brief Brings Matrix in *non*-completely reduced Column Echelon Form and returns the performed operations.
     *  
     * @param performed_ops Applies all column operations to this matrix.
     * @param zero_cols Stores the indices of the columns which are completely zero.
     */
    template< typename smaller_index>
    void column_reduction_triangular_with_memory_int(SparseMatrix<smaller_index>& performed_ops, vec<smaller_index>& zero_cols) {
        for(index j=0; j < this->num_cols; j++) {
            index p = this->col_last(j);
            while( p >= 0) {
                if(this->pivots.count(p)) {
                    index i = this->pivots[p];
                    this->col_op(i, j);
                    performed_ops.col_op(i, j);
                    auto new_p = this->col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    this->pivots[p]=j;
                    break;
                }
            }
            if(p == -1){
                zero_cols.push_back(j);
            }
        }        
    }

    /**
     * @brief ! Compute num_cols ! Computes the Kernel of the matrix by column-reduction. Warning: Changes the matrix!
     * 
     * 
     * @return DERIVED 
     */
    template< typename smaller_index>
    SparseMatrix<smaller_index> get_kernel_int(){
        SparseMatrix<smaller_index> col_operations(this->num_cols, this->num_cols, "Identity");
        vec<smaller_index> zero_cols;
        this->column_reduction_triangular_with_memory_int<smaller_index>(col_operations, zero_cols);
        return col_operations.restricted_domain_copy(zero_cols);
    }

    SparseMatrix<index> get_kernel(){
			//TODO Just added this to be able to compile! 
			return *this;
		}

    /**
     * @brief Computes the cokernel of a sparse matrix over F_2 by column reducing the matrix first
     * Notice that the result must be a cokernel to the non-reduced matrix, too, so we can also use a copy instead, if we want to keep the original matrix.
    *  Careful, this may not be working correctly right now, because it uses another column_reduction algorithm
    * @param S 
    * @param isReduced
    * @return sparseMatrix 
    */
    SparseMatrix coKernel(bool isReduced = false, vec<index>* basisLift = nullptr){
  
        vec<index> quotientBasis;
        try {
            // If isRedcued is true, make sure that pivots are set!
            quotientBasis = this->coKernel_basis(isReduced);
        } catch (std::out_of_range& e) {
            std::cerr << "Error in coKernel Computation: " << e.what() << std::endl;
            this->print();
            std::abort();
        }

		auto indexMap = shiftIndicesMap(quotientBasis);
		SparseMatrix trunc(*this);
		trunc.delete_last_entries();
		transform_matrix(trunc.data, indexMap, true);
	
		index newRows = quotientBasis.size();
		index newCols = this->get_num_rows();
		SparseMatrix result(newCols, newRows);
        result.data = vec<vec<index>>(newCols, vec<index>());
		index j = 0;
		for(index i = 0; i < newCols; i++){
		// Construct Identity Matrix on the generators which descend to the quotient basis. 
			if(j < quotientBasis.size() && quotientBasis[j] == i){
				result.data[i].push_back(j);
				j++;
			} else {
				// Locate the unqiue column with the last entry at i.
				index colForPivot = this->pivots[i];
				result.data[i] = trunc.data[colForPivot];
			}
		}
		assert(j == quotientBasis.size() && "Not all quotient basis elements were used");
		if(basisLift){
			*basisLift = quotientBasis;
		}
		return std::move(result);
    }

    /**
     * @brief Assumes that matrix is reduced and a basislift is computed
     * Computes the cokernel of a sparse matrix over F_2 
    * @return sparseMatrix 
    */
    SparseMatrix coKernel_without_prelim(vec<index>& quotientBasis, vec<index>& row_indices){
  
        // TO-DO: This could be done without computing, storing and then deleting trunc. Need to see if we're too slow.

        auto indexMap = shiftIndicesMap(row_indices);  // Do we need this?
        SparseMatrix trunc(*this);

        // Since the matrix is fully reduced, every pivot row has a unique last entry in some column 
        // and this is still stored in the pivot map. Therefore for computation this re-indexed matrix works better.
        trunc.delete_last_entries();
        transform_matrix(trunc.data, indexMap, true);

        index newRows = quotientBasis.size();
        index newCols = this->num_rows;
        SparseMatrix result(newCols, newRows);

        index j = 0;
        for(index i = 0; i < newCols; i++){
            result.data.push_back(vec<index>());
        // Construct Identity Matrix on the generators which descend to the quotient basis. 
            if(j < quotientBasis.size() && quotientBasis[j] == i){
                result.data[i].push_back(j);
                j++;
            } else {
                // If were in a non-basis-column, compute the entries directly:
                // Locate the unqiue column in the input matrix with the last entry at i.
                // Observe that the matrix trunc is exactly the non-identity part of the matrix as long as we work over F_2
                result.data[i] = trunc.data[this->pivots[i]];
            }
        }
        assert(j == quotientBasis.size() && "Not all quotient basis elements were used");
        return std::move(result);
    }       

    
     /**
     * @brief Assumes that matrix is reduced and a basislift was computed
     * Computes the transpose of a cokernel of a sparse matrix over F_2
    * @return sparseMatrix 
    */
   SparseMatrix coKernel_transposed_without_prelim(vec<index>& basislift){
  
        // TO-DO: Needs testing

        auto index_map = shiftIndicesMap(basislift);  

        index new_rows_t = this->num_rows;
        index new_cols_t = basislift.size();
        SparseMatrix result_t(new_cols_t, new_rows_t);

        result_t.data = vec<vec<index>>(new_cols_t, vec<index>());

        index j = 0;

        for(index i = 0; i < new_rows_t; i++){
            if (j < basislift.size() && basislift[j] == i) {
                result_t.data[j].push_back(i);
                j++;
            } else  {
                index k = this->pivots[i];
                for(index l = 0; this->data[k].size()-1; l++){
                    index entry = this->data[k][l];
                    index row_index = index_map[ entry];
                    result_t.data[row_index].push_back(i);
                }
            }
        }

        return result_t;
    }       

    /**
     * @brief Returns the number of entries in the matrix.
     * 
     * @return long 
     */
    long number_of_entries() {
        long result = 0;
        for(index i=0; i<this->get_num_cols(); i++) {
            // Why was this copied in the original code?
            result+=this->data[i].size();
        }
        return result;
    }

    /**
     * @brief Returns the columns indexed by col_indices, but transformed by a right-multiplication with B. 
     * 
     * @param B 
     * @param col_indices 
     */
    SparseMatrix<index> transformed_restricted_copy(const DenseMatrix& B, vec<index>& col_indices){
        index m = B.num_cols;
        index n = B.num_rows;
        assert(n == col_indices.size());
        SparseMatrix<index> result(m, this->num_rows);
        for(const bitset& v : B.data){
            vec<index> w;
            for(index i = 0; i < n; i++){
                if(v[i]){
                    w.push_back(col_indices[i]);
                }
            }
            result.data.emplace_back(vectorXORMulti(this->data, w));
        }
        return result;
    }

    /**
     * @brief Multiplies from the right with a dense matrix.
     * 
     * @param D 
     */
    void multiply_dense(DenseMatrix& D){
        assert(this->num_cols == D.get_num_rows());
        auto copy = this->data;
        for(index i = 0; i < D.get_num_cols(); i++){
            this->data[i] = vectorXORMulti(copy, D.data[i]);
        }    
    }

    /**
     * @brief Multiplies with D + ID if D is on one side of the diagonal.
     *         Not well optimised. 
     * TO-DO: In practice D will often have at most one entry per column so we should probably 
     * add single vectors iteratively to reduce unnecessary copying
     * 
     * @param D 
     */
    void multiply_id_triangular(DenseMatrix& D){
        assert(this->num_cols == D.get_num_rows());
        for(index i = 0; i < D.get_num_cols(); i++){
            if(D.data[i].any()){
                this->add_to_col(i, vectorXORMulti(this->data, D.data[i]));
            }
        }    
    }

    /**
     * @brief Multiplies from the right with a square dense matrix, but checks if nothing needs to be done.
     * 
     * @param D 
     */
    void multiply_dense_with_e_check(DenseMatrix& D, vec<bitset>& e){
        assert(this->num_cols == D.num_rows);
        assert(D.num_cols == e.size());
        array<index> copy = this->data;
        for(index i = 0; i < D.num_cols; i++){
            if(D.data[i] != e[i]){
                this->data[i] = vectorXORMulti(copy, D.data[i]);
            }
        }    
    }

    /**
     * @brief Multiplies the columns given by col_indices
     *          with a square dense matrix.
     * 
     * @param D 
     */
    void multiply_dense_with_e_check(DenseMatrix& D, vec<bitset>& e_vec, const bitset& col_indices){
        assert(col_indices.count() == D.get_num_rows());
        // D.print();
        // this->print();
        array<index> copy;
        int counter = 0;
        for(auto i = col_indices.find_first(); i != -1; i = col_indices.find_next(i)){
            copy.push_back(this->data[i]);
        }
        for(auto i = col_indices.find_first(); i != -1; i = col_indices.find_next(i)){
            if(D.data[counter] != e_vec[counter]){
                this->data[i] = vectorXORMulti(copy, D.data[counter]);
            }
            counter++;
        }    
        // this->print();
    }



    vec<index> multiply_with_sparse_vector(vec<index>& v){
        return vectorXORMulti(this->data, v);
    }

    vec<index> multiply_with_dense_vector(bitset& v){
        return vectorXORMulti(this->data, v);
    }

    SparseMatrix<index> multiply_right(SparseMatrix<index>& N);

    /**
     * @brief Reduces N as much as possible using column reduction to get a quotient space representation.
     * 
     * @param N 
     */

    bool reduce_fully(SparseMatrix& N, bool is_diagonal = false){
        if(!is_diagonal){
            this->column_reduction();
        }
        vec<index> col_ops;
        for(index i = 0; i < N.get_num_cols(); i++){
            col_ops.clear();
            for(auto it = N.data[i].rbegin(); it != N.data[i].rend(); it++){
                if(this->pivots.count(*it)){
                    col_ops.push_back(this->pivots[*it]);
                }
            }
            Column_traits<vec<index>, index>::add_to(vectorXORMulti(this->data, col_ops), N.data[i]);
        }
        return N.is_zero();
    }

    /**
     * @brief Reduces N as much as possible using column reduction to get a quotient space representation.
     * Alternative version using different addition scheme.
     * 
     * @param N 
     */

    bool reduce_fully_alt(SparseMatrix& N, bool is_diagonal = false){
        if(!is_diagonal){
            this->column_reduction_triangular(true);
        }
        N.pivots.clear();
        for(index j=0; j < N.num_cols; j++) {
            index p = N.col_last(j);
            while( p >= 0) {
                if(this->pivots.count(p)) {
                    index i = this->pivots[p];
                     Column_traits<vec<index>, index>::add_to(this->data[i], N.data[j]);
                    auto new_p = N.col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else if ( N.pivots.count(p)){
                    index i = N.pivots[p];
                    Column_traits<vec<index>, index>::add_to(N.data[i], N.data[j]);
                    auto new_p = N.col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    N.pivots[p]=j;
                    break;
                }
            }
            if (p == -1){
                std::swap(N.data[j], N.data[N.get_num_cols()-1]);
                N.data.pop_back();
                N.num_cols--;
                j--;
            }
        }        
        return N.is_zero();
    }

    /**
     * @brief Reduces N as much as possible using column reduction to get a quotient space representation.
     * 
     * @param N 
     */
    void reduce_fully(vec<index>& N){
        this->column_reduction();
        vec<index> col_ops;
        for(index j: N){
            if(this->pivots.count(j)){
                col_ops.push_back(this->pivots[j]);
            }
        }
        Column_traits<vec<index>, index>::add_to(vectorXORMulti(this->data, col_ops), N);
    }

    //destructor
    ~SparseMatrix(){
        _rows.clear();
    }

}; // SparseMatrix;

template< typename index>
void add_to (const SparseMatrix<index>& A, SparseMatrix<index>& B){
    assert(A.get_num_cols() == B.get_num_cols());
    assert(A.get_num_rows() == B.get_num_rows());
    for(index i = 0; i< A.get_num_cols(); i++){
        Column_traits<vec<index>, index>::add_to(A.data[i],B.data[i]);
    }
}

/**
 * @brief Assumes that M is already transposed, then multiplies as before.
 * 
 */
template <typename index>
SparseMatrix<index> multiply_transpose(const SparseMatrix<index>& M, const SparseMatrix<index>& N){
  SparseMatrix<index> result(N.get_num_cols(), M.get_num_cols());
  result.data.resize(result.get_num_cols());
  // assert(M.get_num_rows() == N.get_num_rows()); Sometimes we dont know.
  for(index i = 0; i < N.get_num_cols(); i++){
    for(index j = 0; j < M.get_num_cols(); j++){
      if(Column_traits<vec<index>, index>::scalar_product(M.data[j], N.data[i])){ 
        result.data[i].push_back(j);
      }
    }
  }
  return result;
}

/**
 * @brief Computes the transpose of M, then multiplies the columns.
 * 
 */
template <typename index>
SparseMatrix<index> multiply(const SparseMatrix<index>& M, const SparseMatrix<index>& N){
    assert(M.get_num_cols() == N.get_num_rows());
    SparseMatrix<index> transpose = M.transposed_copy();
    return multiply_transpose(transpose, N);
}

template <typename index>
SparseMatrix<index>  operator*(const SparseMatrix<index>& M, const SparseMatrix<index>& N){
    return multiply(M, N);
}


template <typename index>
SparseMatrix<index> SparseMatrix<index>::multiply_right(SparseMatrix<index>& N) {
    return multiply(*this, N);
}



/**
 * @brief If all matrices N_map[blocks_to_reduce] contain exactly one row, 
 * then this performs row reduction with these rows for entries after "support".
 * 
 * @tparam index 
 * @tparam DERIVED 
 * @param N_map 
 * @param blocks
 * @param support 
 */
template <typename index>
void simultaneous_row_reduction(std::unordered_map<index, SparseMatrix<index>>& N_map, vec<index>& blocks, bitset& support){
    std::unordered_map<index, index> pivots;
    for(index b : blocks){
        vec<index>& row_b = N_map[b]._rows[0];
        for(index p : row_b){
            if(support.test(p)){
                pivots[p] = b;
                for(index c : blocks){
                    if( b == c){
                        continue;
                    } else {
                        vec<index>& row_c = N_map[c]._rows[0];
                        if(std::find(row_c.begin(), row_c.end(), p) != row_c.end()){
                            add_to(row_b, row_c);
                        }
                    }
                }
                break;
            }
        }
    }
}

/**
 * @brief If all matrices N_map[blocks_to_reduce] contain exactly one row, 
 * then this performs row reduction with these rows for entries after "support".
 * 
 * @tparam index 
 * @tparam DERIVED 
 * @param N_map 
 * @param blocks
 * @param support 
 */
template <typename index>
void simultaneous_row_reduction_on_submatrix(std::unordered_map<index, SparseMatrix<index>>& N_map, vec<index>& blocks, bitset& support, SparseMatrix<index>& A){
    std::unordered_map<index, index> pivots;
    for(index b : blocks){
        vec<index>& row_b = N_map[b]._rows[0];
        for(index p : row_b){
            if(support.test(p)){
                pivots[p] = b;
                for(index c : blocks){
                    if( b == c){
                        continue;
                    } else {
                        vec<index>& row_c = N_map[c]._rows[0];
                        if(std::find(row_c.begin(), row_c.end(), p) != row_c.end()){
                            Column_traits<vec<index>, index>::add_to(row_b, row_c);
                            A.fast_rev_row_op(b, c);
                        }
                    }
                }
                break;
            }
        }
    }
}


/**
 * @brief F_2 Sparse Matrix using std::set / binary trees for its columns.
 * This could work better if the matrix is not very sparse and a lot of single elements need to be changed.
 * @tparam index 
 */
template <typename index>
struct SparseMatrix_set : public MatrixUtil<set<index>, index, SparseMatrix_set<index>>{

    SparseMatrix_set() : MatrixUtil<set<index>, index, SparseMatrix_set<index>>() {}

    SparseMatrix_set(index m) : MatrixUtil<set<index>, index, SparseMatrix_set<index>>(m) {}

    SparseMatrix_set(index m, index n) : MatrixUtil<set<index>, index, SparseMatrix_set<index>>(m, n) {}

    SparseMatrix_set(const SparseMatrix_set& other) : MatrixUtil<set<index>, index, SparseMatrix_set<index>>(other)  {}

    SparseMatrix_set(index m, index n, const vec<set<index>>& data) : MatrixUtil<set<index>, index, SparseMatrix_set<index>>(m, n, data) {}

    SparseMatrix_set(index m, index n, const std::string& type, const index percent = -1) : MatrixUtil<set<index>, index, SparseMatrix_set<index>>(m, n, type, percent) {}

    SparseMatrix_set(SparseMatrix<index>& M) : MatrixUtil<set<index>, index, SparseMatrix_set<index>>(M.get_num_cols(), M.get_num_rows()) {
        for(index i = 0; i < M.get_num_cols(); i++){
            this->data.push_back(set<index>(M.data[i].begin(), M.data[i].end()));
        }
    }

}; // SparseMatrix_set;


} // namespace graded_linalg

#endif

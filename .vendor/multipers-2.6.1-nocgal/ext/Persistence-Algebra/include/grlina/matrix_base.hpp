/**
 * @file matrix_base.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.1
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
 *  This file is part of the AIDA library. 
 *  You can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */


#pragma once


#ifndef MATRIX_BASE_HPP
#define MATRIX_BASE_HPP


#include <iostream>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>
#include <numeric>
#include <cassert>
#include <regex>
#include <grlina/column_types.hpp>

namespace graded_linalg {


template <typename T>
inline std::ostream& operator<< (std::ostream& ostr, const vec<T>& c) {
    for(T i:c) {
      ostr << i << " ";
    }
    return ostr;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::set<T>& v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << *it;
        if (std::next(it) != v.end()) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& p) {
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}

template <typename index>
vec<index> sort_by_permutation(const vec<index>& compare_against){
    vec<index> indices(compare_against.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto comparator = [&compare_against](index a, index b) {
        return compare_against[a] < compare_against[b]; // Change < to > if you want descending order
    };
    std::sort(indices.begin(), indices.end(), comparator);
    return indices;
}

/**
 * @brief Re-coordinatises the positions in the matrix by counting from top to bottom and right to left.
 * @param i column-index from the right
 * @param j row-index
 * @return index 
 */
template<typename index>
index linearise_position_reverse_ext(const index& i, const index& j, const index& ncols, const index& nrows) {
    return (ncols-1-i)*nrows + j;
}

/**
 * @brief Re-coordinatises the positions in the matrix by counting from top to bottom and left to left.
 * @param i column-index from the right
 * @param j row-index
 * @return index 
 */
template<typename index>
index linearise_position_ext(const index& i, const index& j, const index& ncols, const index& nrows) {
    return (i)*nrows + j;
}

/**
 * @brief 
 * 
 * @tparam index 
 * @param k 
 * @param ncols 
 * @param nrows 
 * @return std::pair<index, index> 
 */
template<typename index>
std::pair<index, index> delinearise_position_reverse(index& k, index& ncols, index& nrows) {
    return std::make_pair(ncols-1-k/nrows, k%nrows);
}

// Interface for a matrix
template<typename COLUMN, typename index, typename DERIVED>
class MatrixUtil{

    using CT = Column_traits<COLUMN, index>;

    public:
    vec<COLUMN> data; //stores the columns of the matrix
    std::unordered_map<index,index> pivots; // for the reduction algorithm
    index get_num_rows() const {return num_rows;};
    index get_num_cols() const {return num_cols;};
    void set_num_rows(index m){num_rows = m;};
    void set_num_cols(index n){num_cols = n;};
    void increase_num_cols(index n){num_cols += n;};
    void increase_num_rows(index n){num_rows += n;};

    protected:
    index num_cols;
    index num_rows;
    
    
    MatrixUtil() {};

    MatrixUtil(index m) : num_cols(m), data(vec<COLUMN>()) {
        data.reserve(m);
    }

    MatrixUtil(index m, index n) : num_cols(m), num_rows(n), data(vec<COLUMN>()) {
        data.reserve(m);
    }

    // Copy constructor
    MatrixUtil(const MatrixUtil& other) : data(other.data), num_cols(other.num_cols), num_rows(other.num_rows), pivots(other.pivots) {}

    MatrixUtil(index m, index n, vec<COLUMN> d) : num_cols(m), num_rows(n), data(d) {
        assert(m == d.size());
    }


    protected:
        MatrixUtil& assign(const MatrixUtil& other) {
            if (this != &other) {
                data = other.data;
                num_cols = other.num_cols;
                num_rows = other.num_rows;
                pivots = other.pivots;
            }
            return *this;
        }

        MatrixUtil& assign(MatrixUtil&& other) {
            if (this != &other) {
                data = std::move(other.data);
                num_cols = other.num_cols;
                num_rows = other.num_rows;
                pivots = std::move(other.pivots);
            }
            return *this;
        }

    public:
        MatrixUtil& operator=(const MatrixUtil& other) {
            return assign(other);
        }

        MatrixUtil& operator=(MatrixUtil&& other) {
            return assign(std::move(other));
        }

        

    // Move constructor
    MatrixUtil(MatrixUtil&& other) noexcept : data(std::move(other.data)), num_cols(other.num_cols), num_rows(other.num_rows) {
        // destreoy the source object?

    }

    MatrixUtil(index m, index n, const std::string& type, const index percent = -1) : num_cols(m), num_rows(n), data(vec<COLUMN>()) {
        data.reserve(m);
        if (type == "Identity") {
            assert(m == n);
            for(index i = 0; i < m; i++) {
                this->data.emplace_back( CT::get_standard_vector(i, n) );
            }
        } else if (type == "Random") {
            float fill = percent/100.0;
            if (fill == -1) {
                fill = std::log(n)/n;
                std::cout << "fill rate: " << fill << std::endl;
            }
            for(index i = 0; i < m; i++) {
                this->data.emplace_back( CT::get_random_vector(n, fill) );
            }
        }  else {
                // Check if the type matches "Random" followed by an integer of at most two digits
            std::regex random_regex(R"(Random\s+(\d{1,2}))");
            std::smatch match;
            if (std::regex_match(type, match, random_regex)) {
                if (match.size() == 2) {
                    int fill = std::stoi(match[1].str());
                    for(index i = 0; i < m; i++) {
                        this->data.emplace_back( CT::get_random_vector(n, static_cast<float>(fill)/100 ) );
                    }
                } else {
                    throw std::invalid_argument("Invalid format for Random with an integer: " + type);
                }
            } else {
                throw std::invalid_argument("Unknown matrix type: " + type);
            }
        }  
    }

    MatrixUtil(index n, vec<index> indicator) : num_cols(indicator.size()), num_rows(n), data(vec<COLUMN>()) {
        data.reserve(indicator.size());
        for(index i : indicator) {
            this->data.emplace_back( CT::get_standard_vector(i, n) );
        }
    }


    // Destructor
    ~MatrixUtil() {
        // std::cout << "MatrixUtil Destructor Called on the instance of size" << get_num_cols() << " x "<< get_num_rows() << std::endl;
        data.clear();
        // Do I need to do something else here?
    }
    
    protected:
    /**
     * @brief Set the entry at column i and row j to 1
     * 
     */
    void set_entry(index i, index j){
        CT::set_entry(data[i], j);
    }

    /**
     * @brief Deletes all rows after threshold or from end
     * 
     * @param threshold 
     * @param from_end 
     */
    void cull_columns(const index& threshold, bool from_end = true){};

    void delete_columns(const index& threshold, bool from_end = true){
        assert(threshold <= this->get_num_cols());
        assert(threshold >= 0);

        if(from_end){
            data.erase(data.end() - threshold, data.end());
        } else {
            data.erase(data.begin() + threshold, data.end());
        }
        this->compute_num_cols();
    };

    public:
    /**
     * @brief sets num_cols to th enumber of entries in data. Useful for many subroutines.
     * 
     */
    void compute_num_cols(){
        if(data.size() > 0){
            num_cols = data.size();
        } else {
            num_cols = 0;
        }
        
    };

    public:
    /**
     * @brief Swaps the columns i and j.
     * 
     * @param i 
     * @param j 
     */
    void swap_cols(index i, index j) {
        std::swap(data[i], data[j]);
    };


    COLUMN get_col(index i){
        return data[i];
    }

    /**
     * @brief Adds column i to column j. 
     */
    void col_op(index i, index j){
        CT::add_to(data[i], data[j]);
    };

    /**
     * @brief Adds v to column i.
     */
    void add_to_col(index i, COLUMN v){
        CT::add_to(v, data[i]);
    }

    /**
     * @brief Returns the entry at col i and row j 
     */
    bool is_nonzero_entry(index i, index j){
        return CT::is_nonzero_at(data[i] , j);   
    };

    
    /**
     * @brief Returns the index of the last entry in column i.
     * 
     * @param i 
     * @return index 
     */
    index col_last(index i) const {
        return CT::last_entry_index(data[i]);
    };
    
    public:
    /**
     * @brief Prints the matrix to the console. If suppress_description is set to true, it will not print the number of rows and columns.
     * 
     * @param suppress_description 
     */
    void print(bool suppress_description = false, bool space = false) const {
        if(data.size() != num_cols){
            std::cout << "Data size: " << data.size() << " num_cols: " << num_cols << std::endl;
        }
        assert(data.size() == num_cols);
        
        if(!suppress_description){
            std::cout << "Cols: " << num_cols << " rows: " << num_rows << std::endl;
            }
        for(index i=0;i<num_cols;i++) {
            if(!suppress_description){
                std::cout << "Column " << i << ": ";
            }
            if(space){
                std::cout << "     ";
            }
            std::cout << data[i] << std::endl;
        }
    };
    
    /**
     * @brief Checks if the matrix is the zero matrix.
     * 
     * @return true 
     * @return false 
     */
    bool is_zero(){
        for(auto i = 0; i < this->num_cols; i++){
            if(!CT::is_zero(this->data[i])){
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Returns the indices of the columns which are nonzero.
     * 
     * @return vec<index> 
     */
    vec<index> where_is_nonzero(){
        vec<index> result;
        for(auto i = 0; i < this->num_cols; i++){
            if(!CT::is_zero(this->data[i])){
                result.push_back(i);
            }
        }
        return result;
    }

    /**
     * @brief Returns true if the selected columns are empty.
     * 
     * @return true 
     * @return false 
     */
    bool is_zero(bitset& col_indices){
        for(auto i = col_indices.find_first(); i != bitset::npos ; i = col_indices.find_next(i)){
            if(!CT::is_zero(this->data[i])){
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Checks if the matrix is nonzero
     * 
     * @return true 
     * @return false 
     */
    bool is_nonzero(){
        return !is_zero();
    }

    /**
     * @brief Checks if the matrix is nonzero at col_indices
     * 
     * @return true 
     * @return false 
     */
    bool is_nonzero(bitset& col_indices){
        return !is_zero(col_indices);
    }
   
    bool is_zero(index i){
        return CT::is_zero(this->data[i]);
    }
    
    bool is_nonzero(index i){
        return !CT::is_zero(this->data[i]);
    }

    protected:

    /**
     * @brief Sets the pivot to be the first column which has this entry as the latest non-zero entry in the row.
     */
    void set_pivots_without_reducing() {
        pivots.clear();
        for(index j=0; j<num_cols ; j++) {
            index p = col_last(j);
            if( p >= 0) {
                if(! pivots.count(p)) {
                    pivots[p]=j;
                }
            }
        } 
    }

    public:
   /**
    * @brief Brings Matrix in completly reduced Column Echelon form. Operations go *from* the active column.
    * 
    */
    void column_reduction() {
        for (index j = 0; j < this->num_cols; ++j) {
            index pivotRow = col_last(j);
            if (pivotRow != -1) {
                pivots[pivotRow] = j;
                for (index otherCol = 0; otherCol < num_cols; ++otherCol) {
                    if (j != otherCol && is_nonzero_entry(otherCol, pivotRow)) {
                        col_op(j, otherCol);
                    }
                }
            }
        }
    }
    

    public:
    /**
     * @brief Michael's column reduction algorithm. Brings Matrix in *non*-completely reduced Column Echelon Form. All column operations go from left to right and go *to* the active column.
     *  
     * @param delete_zero_columns If set to true, columns with only zero entries will be deleted.
     */
    void column_reduction_triangular(bool delete_zero_columns = false) {
        pivots.clear();
        for(index j=0; j < this->num_cols; j++) {
            index p = col_last(j);
            while( p >= 0) {
                if(pivots.count(p)) {
                    index i = pivots[p];
                    col_op(i, j);
                    auto new_p = col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    pivots[p]=j;
                    break;
                }
            }
            if (p == -1 && delete_zero_columns){
                std::swap(data[j], data[num_cols-1]);
                data.pop_back();
                num_cols--;
                j--;
            }
        }        
    }

    /**
     * @brief Michael's column reduction algorithm. 
     * Brings Matrix in *non*-completely reduced Column Echelon Form. 
     * All column operations go from left to right and go *to* the active column. 
     * Only reduces columns with a last entry larger than the threshold.
     *  
     * @param delete_zero_columns If set to true, columns with only zero entries will be deleted.
     */
    void column_reduction_triangular(index threshold, bool delete_zero_columns = false) {
        pivots.clear();
        for(index j=0; j < this->num_cols; j++) {
            index p = col_last(j);
            while( p >= threshold) {
                if(pivots.count(p)) {
                    index i = pivots[p];
                    col_op(i, j);
                    auto new_p = col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    pivots[p]=j;
                    break;
                }
            }
            if (p < threshold && delete_zero_columns){
                std::swap(data[j], data[num_cols-1]);
                data.pop_back();
                num_cols--;
                j--;
            }
        }        
    }

    /**
     * @brief Michael's column reduction algorithm. 
     * Brings Matrix in *non*-completely reduced Column Echelon Form. 
     * All column operations go from left to right and go *to* the active column.
     * Only reduces the column indices given by the support bitset.add_to
     * 
     * @param delete_zero_columns If set to true, columns with only zero entries will be deleted.
     */
    void column_reduction_triangular(bitset& support, bitset& zero_cols) {
        assert(support.size() == this->num_cols);
        assert(zero_cols.size() == this->num_cols);
        pivots.clear();
        for(index j = support.find_first(); j < this->num_cols; j = support.find_next(j)) {
            index p = col_last(j);
            while( p >= 0) {
                if(pivots.count(p)) {
                    index i = pivots[p];
                    col_op(i, j);
                    auto new_p = col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    pivots[p]=j;
                    break;
                }
            }
            if(p == -1){
                zero_cols.set(j);
            }
        }        
    }

    

   
    /**
    * @brief Brings Matrix in reduced Column Echelon form and returns the performed operations if output is set to true.
    * 
    * @param performed_ops Pass an empty matrix of the same type as the current matrix. The performed operations will be stored in this matrix.
    * @param comments If set to true, the function will print out the pivot rows and the operations performed.
    * @param with_swaps If set to true, the function will also perform swaps to bring the matrix in standard form.
    */
    void column_reduction_with_memory(DERIVED& performed_ops, bool comments = false, bool with_swaps = false) {
        for (index col = 0; col < num_cols; ++col) {
            index pivotRow = col_last(col);
            if(comments){
                std::cout << "Column " << col << " has pivot at row " << pivotRow << std::endl;
            }
            if(pivotRow >= num_rows){
                // This shouldnt be a mistake, but a warning.
                throw std::out_of_range("There is an index in the column that is larger than the number of rows at column: " + std::to_string(col) + " and the entry is " + std::to_string(pivotRow) + " and the number of rows is " + std::to_string(num_rows));
            } 
            
            if (pivotRow != -1) {
                pivots[pivotRow] = col;
                if(comments){ std::cout << "pivot at row: " << pivotRow << std::endl; }
                for (index otherCol = 0; otherCol < num_cols; ++otherCol) {
                    if(comments){ std::cout << "Checking column " << otherCol << " for " << is_nonzero_entry(otherCol, pivotRow) << std::endl;}
                    if (col != otherCol && is_nonzero_entry(otherCol, pivotRow)) {
                        if(comments){ std::cout << "Performing operation from" << col << " to " << otherCol << std::endl; }
                        col_op(col, otherCol);
                        performed_ops.col_op(col, otherCol);
                    }
                }
            }
        }
        if(comments){
            std::cout << "pivots are";
            for(auto it = pivots.begin(); it != pivots.end(); it++){
                std::cout << it->first << " " << it->second << std::endl;
            }
        }
    }

    /**
     * @brief Brings Matrix in *non*-completely reduced Column Echelon Form and returns the performed operations.
     * 
     * @param performed_ops Applies all column operations to this matrix.
     */
    void column_reduction_triangular_with_memory(DERIVED& performed_ops) {
        for(index j=0; j < this->num_cols; j++) {
            index p = col_last(j);
            while( p >= 0) {
                if(pivots.count(p)) {
                    index i = pivots[p];
                    col_op(i, j);
                    performed_ops.col_op(i, j);
                    auto new_p = col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    pivots[p]=j;
                    break;
                }
            }
        }        
    }

    /**
     * @brief Brings Matrix in *non*-completely reduced Column Echelon Form and returns the performed operations.
     *  
     * @param performed_ops Applies all column operations to this matrix.
     * @param zero_cols Stores the indices of the columns which are completely zero.
     */
    void column_reduction_triangular_with_memory(DERIVED& performed_ops, vec<index>& zero_cols) {
        for(index j=0; j < this->num_cols; j++) {
            index p = col_last(j);
            while( p >= 0) {
                if(pivots.count(p)) {
                    index i = pivots[p];
                    col_op(i, j);
                    performed_ops.col_op(i, j);
                    auto new_p = col_last(j);
                    assert( new_p < p);
                    p = new_p;
                } else {
                    pivots[p]=j;
                    break;
                }
            }
            if(p == -1){
                zero_cols.push_back(j);
            }
        }        
    }
    
    /**
     * @brief Compares the raw data of this matrix to another one, ignoring all other attributes.
     * 
     * @param other 
     * @return true 
     * @return false 
     */
    bool equals(MatrixUtil& other, bool output = false){
        if(num_cols != other.num_cols){
            if(output){
                std::cout << "#columns dont match.";
            }
        return false;
        }
        if(num_rows != other.num_rows){
            if(output){
                std::cout << "#rows dont match.";
            }
            return false;
        }
        for(index i = 0; i< num_cols; i++){
            if( !CT::is_equal(data[i], other.data[i]) ){
                if(output){
                    std::cout << "Column " << i << " does not match.";
                    std::cout << "This: " << data[i] << "\n Other: " << other.data[i] << std::endl;
                }
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Compares with another matrix and returns the index of the first column that does not match
     *  or -1 if they are equal.
     * 
     * @param other 
     * @return true 
     * @return false 
     */
    index equals_with_entry_check(MatrixUtil& other, bool output = false){
        
        for(index i = 0; i< num_cols; i++){
            if( !CT::is_equal (data[i], other.data[i]) ){
                if(output){
                    std::cout << "Column " << i << " does not match.";
                }
                return i;
            }
        }
        return -1;
    }

    

    /**
    * @brief Returns a copy with only the columns at the indices given in colIndices.
    * 
    * @param colIndices 
    * @return sparseMatrix 
    */
    DERIVED restricted_domain_copy(vec<index>& colIndices){
        for(index i : colIndices){
            assert(i < this->num_cols);
        }
        DERIVED result(colIndices.size(), this->num_rows);
        for(index i = 0; i < colIndices.size(); i++){
            result.data[i] = this->data[colIndices[i]];
        }
        return result;
    }


    /**
    * @brief Returns a copy with only the columns at the indices given in colIndices but adding start.
    * 
    * @param colIndices 
    * @return sparseMatrix 
    */
    DERIVED restricted_domain_copy(bitset& colIndices, index start = 0){
        assert(colIndices.size() + start <= this->num_cols);
        DERIVED result(colIndices.count(), this->num_rows);
        index col = 0;
        for(index i = 0; i < colIndices.size(); i++){
            if(colIndices[i]){
                result.data[col] = this->data[i+start];
                col++;
            }
        }
        return result;
    }

    public:
    /**
     * @brief Re-coordinatises the positions in the matrix by counting from top to bottom and right to left.
     * @param i column-index from the right
     * @param j row-index
     * @return index 
     */
    long linearise_position_reverse(index i, index j) {
        long result = ((static_cast<long>(this->num_cols))-1-i)*static_cast<long>(this->num_rows) + j;
        return result;
    }

    

    /**
     * @brief Inverse function to linearize_position_reverse
     * 
     * @param k 
     * @return std::pair<index,index> 
     */
    std::pair<index,index> delinearise_position_reverse(long k) {
        index n = this->num_cols;
        index m = this->num_rows;
        assert( linearise_position_reverse(n-1-k/m,k%m)==k);
        return std::make_pair(n-1-k/m, k%m);
    }

    
    /**
     * @brief Counts the entries in the matrix from left to right and top to bottom.
     * 
     * @param i column-index
     * @param j row-index
     * @return index 
     */
    index linearise_position(index i, index j) {
        return i*this->num_cols + j;
    }

    /**
     * @brief Inverse function to linearise_position
     * 
     * @param k 
     * @return std::pair<index,index> 
     */
    std::pair<index,index> delinearise_position(index k) {
        index n = this->num_cols;
        index m = this->num_rows;
        index i = k/n;
        index j = k%n;
        assert( linearise_position(i,j)==k);
        return std::make_pair(i,j);
    }


    public:
    /**
     * @brief Solves the Linear System (data * x) = N_{target} using triangular column-reduction and returns a solution.
     * 
     * @param N 
     * @param solution Stores the column operations which where used to solve the system.
     * @param get_ops Indicates if we want to know the solution or just solvability.
     */
    bool solve_col_reduction(DERIVED& N, DERIVED&& solution = DERIVED(), bool get_ops = false){
        
        DERIVED pre_performed_ops;
        if (get_ops){
            pre_performed_ops = std::move( DERIVED(num_cols, num_cols, "Identity") );
            this->column_reduction_triangular_with_memory(pre_performed_ops);
            auto solution = DERIVED(N.num_cols, this->num_cols);
        } else {
            this->column_reduction_triangular();
        }

        for(index i = 0; i < N.data.size(); i++){
            auto p = N.col_last(i);
            while(p >= 0){
                if(this->pivots.count(p)){
                    index j = this->pivots[p];
                    CT::add_to(this->data[j], N.data[i]);
                    if (get_ops){
                        CT::add_to(pre_performed_ops.data[j], solution.data[i]);
                    }
                } else {
                    return false;
                }
                p = N.col_last(i);
            }
        }
        return true;
    }

    

    /**
     * @brief Michaels Column-Reduction Solver, modified to return the correct solution. Also deleted non-functional code.
     *      This reducer is better than the other one, because it does not require re-transformation when there are many empty rows.
     *      This is now a function of the object "S" instead of a pure function for no good reason other than trying an object-oriented approach.
     * @param c the target
     * @param threshold 
     * @param solution Stores the column operations which where used to solve the system.
     * @param complete_reduce Indicate if you only want to solve the system until a certain point 
     * @param reduce_S Indicates, if this matrix needs to be reduced. Set to false, if already reduced. Pivots will be computed again.
     * @param get_ops Indicates if we want to know the solution or just solvability.
     */
    bool solve_col_reduction(COLUMN& c, COLUMN& solution, bool complete_reduce = true, bool reduce_S=true, bool get_ops = true) {

        auto pre_performed_ops = DERIVED(this->num_cols, this->num_cols, "Identity");

        if(reduce_S) {
            if(get_ops){
                this->column_reduction_triangular_with_memory(pre_performed_ops);
            } else {
                this->column_reduction_triangular();
            }
        } else {
            set_pivots_without_reducing();
        }


        index p = CT::last_entry_index(c);
        while(p >= 0 || (!complete_reduce)) {
            if( this->pivots.count(p) ) {
                index i = pivots[p];
                CT::add_to(this->data[i], c);
                if(get_ops){
                    if(reduce_S){
                        CT::add_to(pre_performed_ops.data[i], solution);
                    } else {
                        CT::set_entry(solution, i);
                    }
                }
                auto p_new = CT::last_entry_index(c);
                assert(p_new < p);
                p = p_new;
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Returns a basis for the kernel of the matrix via column reduction.
     * 
     * @return DERIVED 
     */
    DERIVED kernel(){
        DERIVED solution = DERIVED(num_cols, num_cols, "Identity");
        vec<index> basis_indices = vec<index>();
        this->column_reduction_triangular_with_memory(solution, basis_indices);
        return solution.restricted_domain_copy(basis_indices);
    }    
    

    /**
     * @brief Computes A, B, such that this*A = B and B is column-reduced and ordered
     * 
     * @param performed_ops Applies all column operations to this matrix.
     */
    void column_gauss_jordan (DERIVED& performed_ops){
        index col_counter = 0;
        for (index r = 0; r < num_rows; r++) {
            bool found_pivot = false;
            for(index c = col_counter; c < num_cols; c++){
                if(is_nonzero_entry(c, r)){
                    if(c != col_counter){
                        swap_cols(col_counter, c);
                        performed_ops.swap_cols(col_counter, c);
                    }
                    found_pivot = true;
                    pivots[r] = col_counter;
                    break;
                }
            }

            if(found_pivot){
                for(index c = 0; c < num_cols; c++){
                    if(c != col_counter && is_nonzero_entry(c, r)){
                        col_op(col_counter, c);
                        performed_ops.col_op(col_counter, c);
                    }
                }
                col_counter++;
            }
        }
    }

    /**
     * @brief Solves (A^T)x=b with Gauss elimination, pretending that this is a row-matrix. 
     * TO-DO : Implement
     * 
     * @param performed_ops 
     */
    void column_gauss_solver (COLUMN& performed_ops){
    }

    bool is_invertible(){
        DERIVED copy(static_cast<const DERIVED&>(*this));
        copy.column_reduction_triangular();
        for(index i = 0; i < copy.num_cols; i++) {
            if(!copy.pivots.count(i)){
                return false;
            }
        }
        return true;
    }
    /**
     * @brief Computes an inverse using column-reduction without changing the matrix.
     * 
     * @return DERIVED 
     */
    DERIVED inverse() {
        if( this->num_cols != this->num_rows) {
            throw std::invalid_argument("Matrix is not square.");
        }
        DERIVED result(this->num_cols, this->num_cols, "Identity");
        DERIVED copy(static_cast<const DERIVED&>(*this));
        copy.column_gauss_jordan(result);
        for(index i = 0; i < copy.num_cols; i++) {
            if(!copy.pivots.count(i)){
                std::cout << "missing row index i: " << i << std::endl;
                this->print();
                std::cout << "was reduced to:";
                copy.print();
                throw std::invalid_argument("Matrix might not be invertible.");
            }
        }
        return result;
    }

    /**
     * @brief Computes an inverse using column-reduction without changing the matrix.
     * 
     * @return DERIVED 
     */
    DERIVED inverse_nocopy() {
        if( this->num_cols != this->num_rows) {
            throw std::invalid_argument("Matrix is not square.");
        }
        DERIVED result(this->num_cols, "Identity");
        this->column_gauss_jordan(result);
        for(index i = 0; i < this->num_cols; i++) {
            if(!this->pivots.count(i)){
                std::cout << "missing row index i: " << i << std::endl;
                std::cout << "was reduced to:";
                this->print();
                throw std::invalid_argument("Matrix might not be invertible.");
            }
        }
        return result;
    }

    /**
     * @brief Computes this*other^{-1} without changing either matrix.
     * Can be computed faster even using column_reduction_with_memory using other as input, extending the getinverse function. (Could implement this).
     * 
     */
    DERIVED divide_right(const DERIVED& other) const {
        DERIVED result = DERIVED(static_cast<DERIVED&>(*this));
        DERIVED other_copy(other);
        other_copy.column_gauss_jordan(result);
        return result;
    }

    

    /**
     * @brief Computes this*other^{-1} without changing either matrix.
     * Can be computed faster even using column_reduction_with_memory using other as input, extending the getinverse function. (Could implement this).
     * 
     */
    void divide_right_nocopy(DERIVED& other) {
        assert(other.is_invertible());
        other.column_gauss_jordan(this);
    }

    
    /**
     * @brief Appends the columns of another matrix.
     * 
     * @param other 
     */
    void append_matrix(const DERIVED& other) {
        assert(this->num_rows == other.num_rows);
        for(index i = 0; i < other.num_cols; i++) {
            this->data.push_back(other.data[i]);
        }
        this->num_cols += other.num_cols;
    }

    public:

    /**
     * @brief Recursively finds a permutation of the column indices such that all diagonal elements are non-zero.
     *        Matrix needs to be square and invertible for this algorithm.
     * @return vec<int> 
     */
    vec<index> rectify_invertible(){
        if(this->num_cols == 1){
            return {0};
        }
        assert(this->num_cols == this->num_rows);
        vec<index> minor_indices = vec<index>(this->num_cols-1);
        vec<index> permutation = vec<index>(this->num_cols);

        // Find a column whose last entry is nonzero and whose associated minor is invertible
        for(index i = 0; i < this->num_cols; i++){
            if(this -> col_last(i) == this->num_rows-1){
                for(index j = 0; j < this->num_cols; j++){
                    if(j < i){
                        minor_indices[j] = j;
                    } else if(j > i){
                        minor_indices[j-1] = j;
                    }
                }
                DERIVED minor = this->restricted_domain_copy(minor_indices);
                minor.cull_columns(1);
                DERIVED minor_copy(minor);
                //This could be speed up by stoping when the reduction deletes a column completly. 
                //TO-DO: If this algorithm needs to be faster, then change this.
                minor_copy.column_reduction_triangular();
                bool minor_is_invertible = true;
                for(index k = 0; k < minor_copy.num_rows; k++){
                    if(minor_copy.pivots.count(k) == 0){
                        minor_is_invertible = false;
                        break;
                    }
                }

                // If the minor is invertible, we can recursively find a permutation for the minor
                if(minor_is_invertible){
                    vec<int> minor_permutation = minor.rectify_invertible();
                    for(index j = 0; j < this->num_cols; j++){
                        if(j < i){
                            permutation[j] = minor_permutation[j];
                        } else if(j > i){
                            permutation[j] = minor_permutation[j-1];
                        } else {
                            permutation[j] = this->num_cols-1;
                        }
                    }
                    return permutation;
                }
            }
        }
        throw std::invalid_argument("Matrix is not invertible.");
    }

    /**
     * @brief Reorder the columns given a permutation as a vector of integers by copying everything.
     * 
     * @param permutation 
     */
    void reorder_columns(vec<index> permutation){
        assert(permutation.size() == this->num_cols);
        DERIVED copy(static_cast<DERIVED&>(*this));
        for(index i = 0; i < this->num_cols; i++){
            this->data[permutation[i]] = copy.data[i];
        }
    }

    /**
     * @brief Given a set of indices to be ordered, reorder the columns in the same way.
     * 
     * @param permutation 
     */
    void reorder_via_comparison(vec<index> compare_against){
        reorder_columns(sort_by_permutation(compare_against));
    }


    /**
     * @brief Verifies if all diagonal entries are non-zero.
     * 
     * @return true 
     * @return false 
     */
    bool test_diagonal_entries(){
        for(index i = 0; i < this->num_cols; i++){
            if(!is_nonzero_entry(i,i)){
                return false;
            }
        }
        return true;
    }

    public:

    


    /**
     * @brief Computes a set of row indices whose images under the quotient map form a basis of the cokernel.
     *          Equivalently, the set of row indices which are not pivots after column-reduction.
     * @return vec<index> 
     */
    vec<index> coKernel_basis(const bool& is_reduced = false){
        vec<index> basis;
        if(!is_reduced){
            column_reduction();
        }
        for(index i = 0 ; i < this->num_rows; i++){
            if(pivots.count(i) == 0){
                basis.push_back(i);
            }
        }
        return basis;
    }

    vec<index> coKernel_basis_local(const vec<index>& row_indices, const bool& no_reduction = false){
        vec<index> basis;
        if(!no_reduction){
            column_reduction();
        }
        for(index i : row_indices){
            if(pivots.count(i) == 0){
                basis.push_back(i);
            }
        }
        return basis;
    }

    /**
     * @brief Computes a set of row indices whose images under the quotient map form a basis of the cokernel.
     *          Equivalently, the set of row indices which are not pivots after column-reduction.
     * @param mask
     * @param row_indices
     * @return vec<index> 
     */
    vec<index> coKernel_basis(const vec<index>& mask, const vec<index>& row_indices, const bool& no_reduction = false){
        vec<index> basis;
        if(!no_reduction){
            column_reduction();
        }
        for(index i : mask){
            if(pivots.count(row_indices[i]) == 0){
                basis.push_back(i);
            }
        }
        return basis;
    }

    vec<index> coKernel_basis(const vec<index>& row_indices, const bool& no_reduction = false){
        vec<index> basis;
        if(!no_reduction){
            column_reduction();
        }
        for(index i = 0 ; i < this->num_rows; i++){
            if(pivots.count(row_indices[i]) == 0){
                basis.push_back(i);
            }
        }
        return basis;
    }

    

    
    // Adds the matrices together, so that we can treat matrices themselves as vectors and do reduction
    
    /**
     * @brief Usual addition of matrices.
     * 
     * @param other 
     * @return DERIVED 
     */
    DERIVED operator+(const DERIVED& other) const {
        // Ensure the matrices have the same dimensions
        assert(this->num_cols == other.num_cols);
        assert(this->num_rows == other.num_rows);

        // Create a new DERIVED object to store the result
        DERIVED result(this->num_cols, this->num_rows);

        // Add the columns of the two matrices
        for (index i = 0; i < this->num_cols; ++i) {
            result.data[i] = this->data[i];
            this->CT::add_to(other.data[i], result.data[i]);
        }

        return result;
    }

    /**
     * @brief Adds this matrix to the other matrix in place.
     * 
     * @param other 
     */
    void add_matrix_to(MatrixUtil& other){
        // Ensure the matrices have the same dimensions
        assert(this->num_cols == other.num_cols);
        assert(this->num_rows == other.num_rows);

        // Add the columns of the two matrices
        for (index i = 0; i < this->num_cols; ++i) {
            CT::add_to(this->data[i], other.data[i]);
        }
    }

    /**
     * @brief Returns the coordinates of the last non-zero entry for reduction
     */
    pair<index> last_entry (){
        for(index i = num_cols-1; i >= 0; i--){
            index p = col_last(i);
            if(p >= 0){
                return std::make_pair(i, p);
            }
        }
        return std::make_pair(-1, -1);
    }

}; //MatrixUtil


/**
 * @brief Performs reduction on a vector of matrices
 * 
 * @tparam COLUMN 
 * @tparam index 
 * @tparam DERIVED 
 * @param matrices 
 * @return vec<index> 
 */
template<typename index, typename T>
vec<index> general_reduction(vec< T > matrices) {
    // Ensure all matrices have the same dimensions
    assert(!matrices.empty());
    index num_cols = matrices[0].get_num_cols();
    index num_rows = matrices[0].get_num_rows();
    for (const T& matrix : matrices) {
        assert(matrix.get_num_cols() == num_cols);
        assert(matrix.get_num_rows() == num_rows);
    }

    vec<index> non_zero_indices;

    for (index j = 0; j < matrices.size(); ++j) {
        T& matrix = matrices[j];
        auto [col, row] = matrix.last_entry();
        if (row != -1) {
            for (index k = j; k < matrices.size(); k++) {
                if (j != k && matrices[k].is_nonzero_entry(col, row)) {
                    matrix.add_matrix_to(matrices[k]);
                }
            }
            non_zero_indices.push_back(j);
        }
    }

    return non_zero_indices;
}

/**
 * @brief Treats all matrices in N_map[all_blocks] as stacked on top and column reduces them diagonally 
 * wrt. the parts given by blocks to reduce on the columns from "support".
 * 
 * @tparam index 
 * @tparam DERIVED 
 * @param N_map 
 * @param blocks_to_reduce 
 * @param all_blocks 
 * @param support 
 * @param zero_cols 
 */
template <typename index, typename DERIVED>
bitset simultaneous_column_reduction(std::unordered_map<index, DERIVED>& N_map, 
        vec<index>& blocks_to_reduce, vec<index>& all_blocks, bitset& support){
    
    
    index num_cols = N_map[all_blocks[0]].get_num_cols();
    assert(support.size() == num_cols);
    bitset non_zero_cols = bitset(num_cols, false);

    for(index col = support.find_first(); col != bitset::npos; col = support.find_next(col)){
        for(index i = blocks_to_reduce.size()-1; i >= 0 ; i--){
            index b = blocks_to_reduce[i];
            DERIVED& N_b = N_map[b];
            index p = N_b.col_last(col);
            if(p >= 0){
                N_b.pivots[p] = col;
                for(index col_target = support.find_first(); col_target != bitset::npos; col_target = support.find_next(col_target)){
                    if(col_target != col && N_b.is_nonzero_entry(col_target, p)){
                        for(index j = 0; j < all_blocks.size(); j++){
                            N_map[all_blocks[j]].col_op(col, col_target);
                        }
                    }
                }
                non_zero_cols.set(col);
                break;
            }
        }
    }
    return non_zero_cols;
}

/**
 * @brief Treats all matrices in N_map[all_blocks] as stacked on top and column reduces them diagonally 
 * wrt. the parts given by blocks to reduce on the columns from "support".
 * 
 * @tparam index 
 * @tparam DERIVED 
 * @param N_map 
 * @param blocks_to_reduce 
 * @param all_blocks 
 * @param support 
 * @param zero_cols 
 */
template <typename index, typename DERIVED>
bitset simultaneous_column_reduction_full_support(std::unordered_map<index, DERIVED>& N_map, 
        vec<index>& blocks_to_reduce, vec<index>& all_blocks){
    
    index num_cols = N_map[all_blocks[0]].get_num_cols();
    bitset non_zero_cols = bitset(num_cols, false);

    for(index col = 0; col < num_cols; col++){
        for(index i = blocks_to_reduce.size()-1; i >= 0 ; i--){
            index b = blocks_to_reduce[i];
            DERIVED& N_b = N_map[b];
            index p = N_b.col_last(col);
            if(p >= 0){
                N_b.pivots[p] = col;
                for(index col_target = 0; col_target < num_cols; col_target ++){
                    if(col_target != col && N_b.is_nonzero_entry(col_target, p)){
                        for(index j = 0; j < all_blocks.size(); j++){
                            N_map[all_blocks[j]].col_op(col, col_target);
                        }
                    }
                }
                non_zero_cols.set(col);
                break;
            }
        }
    }
    return non_zero_cols;
}

/**
 * @brief Pushes all non-zero columns after support to the left.
 * 
 * @param N_map 
 * @param all_blocks 
 * @param support 
 * @param non_zero_cols 
 */
template <typename index, typename DERIVED>
void simultaneous_align(std::unordered_map<index, DERIVED>& N_map, vec<index>& all_blocks, bitset& support, bitset& non_zero_cols){
    index num_cols = N_map[all_blocks[0]].num_cols;
    auto non_zero = non_zero_cols.find_first();
    for(index col = support.find_first(); non_zero != bitset::npos; col = support.find_next(col)){
        if(col == non_zero){
            non_zero = non_zero_cols.find_next(non_zero);
        } else {
            for(index b : all_blocks){
                std::swap( N_map[b].data[col], N_map[b].data[non_zero]);
                non_zero = non_zero_cols.find_next(non_zero);
            }
        }
    }
}



/**
 * @brief returns true if the column spaces of two matrices of equal dimension are isomorphic.
 * 
 * @tparam DERIVED 
 * @param A 
 * @param B 
 * @return true 
 * @return false 
 */
template <typename DERIVED>
bool compare_col_space(const DERIVED& A,const DERIVED& B){
    if(A.get_num_cols() != B.get_num_cols()){
        return false;
    }
    DERIVED copy_A = DERIVED(A);
    DERIVED copy_B = DERIVED(B);
    DERIVED A_inv = DERIVED(A.get_num_cols(), "Identity");
    DERIVED B_inv = DERIVED(B.get_num_cols(), "Identity");
    copy_A.column_gauss_jordan(A_inv);
    copy_B.column_gauss_jordan(B_inv);
    
    return copy_A.equals(copy_B);
}



} // namespace graded_linalg


#endif

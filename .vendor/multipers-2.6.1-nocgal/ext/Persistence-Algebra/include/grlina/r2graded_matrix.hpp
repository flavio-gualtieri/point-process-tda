/**
 * @file r2graded_matrix.hpp
 * @author Jan Jendrysiak
 * @brief The Kernel computation is adapted from MPfree (Michael Kerber)
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

#ifndef R2GRADED_MATRIX_HPP
#define R2GRADED_MATRIX_HPP

#include "grlina/orders_and_graphs.hpp"
#include <grlina/graded_matrix.hpp>
#include <grlina/grid_scheduler.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace graded_linalg {

using r2degree = std::pair<double, double>;

// Vector addition
inline r2degree operator+(const r2degree& a,
    const r2degree& b) {
return {a.first + b.first, a.second + b.second};
}

inline vec<r2degree> operator+(const vec<r2degree>& a,
    const r2degree& b) {
    vec<r2degree> result = a;
    for (auto& elem : result) {
        elem = elem + b;
    }
    return result;
}

// Vector subtraction
inline r2degree operator-(const r2degree& a,
    const r2degree& b) {
return {a.first - b.first, a.second - b.second};
}


// Scalar multiplication (scalar * pair)
inline r2degree operator*(double scalar, const r2degree& p) {
return {scalar * p.first, scalar * p.second};
}

// Scalar multiplication (pair * scalar)
inline r2degree operator*(const r2degree& p, double scalar) {
return {p.first * scalar, p.second * scalar};
}

inline r2degree operator/(const r2degree& p, double scalar) {
    return {p.first / scalar, p.second / scalar};
}

template<>
struct Degree_traits<r2degree> {
    static bool equals(const r2degree& lhs, const r2degree& rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    static bool smaller(const r2degree& lhs, const r2degree& rhs) {
        if(lhs.first < rhs.first) {
            return (lhs.second <= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
        } else {
            return false;
        }
    }

    static bool greater(const r2degree& lhs, const r2degree& rhs) {
        if(lhs.first > rhs.first) {
            return (lhs.second >= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second > rhs.second;
        } else {
            return false;
        }
    }

    static bool greater_equal(const r2degree& lhs, const r2degree& rhs) {
        return (lhs.first >= rhs.first) && (lhs.second >= rhs.second);
    }

    static bool smaller_equal(const r2degree& lhs, const r2degree& rhs) {
        return (lhs.first <= rhs.first) && (lhs.second <= rhs.second);
    }

    static bool lex_order(const r2degree& a, const r2degree& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    }

    static bool colex_order(const r2degree& a, const r2degree& b) {
        if (a.second != b.second) {
            return a.second < b.second;
        } else {
            return a.first < b.first;
        }
    }

    /**
    * @brief Lambda function to compare lexicographically for sorting.
    */
    static std::function<bool(const r2degree&, const r2degree&)> lex_lambda() {
        return [](const r2degree& a, const r2degree& b) {
            return Degree_traits<r2degree>::lex_order(a, b);
        };
    }

    /**
    * @brief Lambda function to compare colexicographically for sorting.
    */
    static std::function<bool(const r2degree&, const r2degree&)> colex_lambda() {
        return [](const r2degree& a, const r2degree& b) {
            return Degree_traits<r2degree>::colex_order(a, b);
        };
    }

    static vec<double> position(const r2degree& a)  {
        return {a.first, a.second};
    }

    static void print_degree(const r2degree& a) {
        std::cout << "(" << a.first << ", " << a.second << ")";
    }

    static r2degree join(const r2degree& a, const r2degree& b)  {
        return {std::max(a.first, b.first), std::max(a.second, b.second)};
    }

    static r2degree meet(const r2degree& a, const r2degree& b) {
        return {std::min(a.first, b.first), std::min(a.second, b.second)};
    }

    
    /**
     * @brief Writes the r2degree to an output stream.
     */
    template <typename OutputStream>
    static void write_degree(OutputStream& os, const r2degree& a) {
        os << a.first << " " << a.second;
    }

    template <typename InputStream>
    static r2degree from_stream(InputStream& iss){
        r2degree deg;
        iss >> deg.first >> deg.second;
        return deg;
    }

    static void add(const r2degree& a, r2degree& b) {
        b.first += a.first;
        b.second += a.second;
    }

    static void subtract(const r2degree& a, r2degree& b){
        b.first -= a.first;
        b.second -= a.second;
    }


}; //Degree_traits<r2degree>




/**
 * @brief A graded matrix with degrees in R^2.
 * 
 * @tparam index 
 */
template <typename index>
struct R2GradedSparseMatrix : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>> {

    // For kernel computation we will need to compute a grid, i.e. a function Z^2 -> R^2, 
    // such that all degrees of columsn and rows are in the image of this function.

    vec<double> x_grid;
    vec<double> y_grid;

    std::unordered_map<double, index> x_to_index;
    std::unordered_map<double, index> y_to_index;

    vec<pair<index>> z2_col_degrees;
    vec<pair<index>> z2_row_degrees;

    // This is also used in kernel computation
    typedef std::priority_queue<index,std::vector<index>,std::greater<index>> PQ;

    Grid_scheduler<index> grid_scheduler;
    std::vector<PQ> pq_row;


    R2GradedSparseMatrix() : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>() {}


    R2GradedSparseMatrix( SparseMatrix<index>&& other) :  GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(std::move(other)) {}
    

    public:

    R2GradedSparseMatrix& operator= (const R2GradedSparseMatrix& other) {
        if (this != &other)
            GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>::assign(other);
        return *this;
    }

    R2GradedSparseMatrix& operator= (R2GradedSparseMatrix&& other) {
        if (this != &other)
            GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>::assign(std::move(other));
        return *this;
    }

    R2GradedSparseMatrix& operator = (const GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>& other) {
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>::assign(other);
        return *this;
    }

    R2GradedSparseMatrix& operator = (GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>&& other) {
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>::assign(std::move(other));
        return *this;
    }



    R2GradedSparseMatrix(index cols, index rows) : 
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(cols, rows) {}
    R2GradedSparseMatrix(index n, vec<index> indicator) : 
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(n, indicator) {} 
    R2GradedSparseMatrix(index cols, index rows, std::string type, const index percent = -1) : 
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(cols, rows, type, percent) {} // Constructor with type
    R2GradedSparseMatrix(index m, index n, vec<r2degree> c_degrees, vec<r2degree> r_degrees) : 
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(m, n, c_degrees, r_degrees) {} // Constructor with degrees
    R2GradedSparseMatrix(index m, index n, const array<index>& data, vec<r2degree> c_degrees, vec<r2degree> r_degrees) : 
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(m, n, data, c_degrees, r_degrees) {} // Constructor with data and degrees

    R2GradedSparseMatrix(const R2GradedSparseMatrix& other)
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(other) {
    } // Copy constructor

    R2GradedSparseMatrix(R2GradedSparseMatrix&& other)
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(std::move(other)) {
    } // Move constructor

    R2GradedSparseMatrix(const GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix>& other) {
        this->assign(other);
    } // Copy constructor from GradedSparseMatrix
    R2GradedSparseMatrix( const SparseMatrix<index>& other) : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(other.get_num_cols(), other.get_num_rows()) {
        this->data = other.data;
    } // Copy constructor from SparseMatrix

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) 
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(filepath, lex_sort, compute_batches) {
    } // Constructor from file


    /**
     * @brief Constructs an R^2 graded matrix from an input file stream.
     * 
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index>>(file_stream, lex_sort, compute_batches) {
    } // Constructor from ifstream

    /**
     * @brief Overwrites restricted_domain_copy from SparseMatrix to also copy the degrees.
     * 
     * @param colIndices 
     * @return R2GradedSparseMatrix 
     */
    R2GradedSparseMatrix restricted_domain_copy(vec<index>& colIndices) const {
        R2GradedSparseMatrix result( this->SparseMatrix<index>::restricted_domain_copy(colIndices) );
        result.col_degrees = vec<r2degree>(colIndices.size());
        for(index i = 0; i < colIndices.size(); i++){
            result.col_degrees[i] = this->col_degrees[colIndices[i]];
        }
        result.row_degrees = this->row_degrees;

        return result;
    }


    /**
     * @brief Sets up the grid scheduler for kernel computation
     * 
     */
    void initialise_grid_scheduler() {
        this->grid_scheduler = Grid_scheduler<index>(*this);
    }

    /**
     * @brief Sorts the row degrees, stores the permutation used and transforms the entries of the sparse col vectors accordingly.
     * 
     */
    void sort_rows_colexicographically() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->row_degrees, Degree_traits<r2degree>::colex_lambda());
        vec<index> reverse = vec<index>(permutation.size());
        for (index i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }
        this->transform_data(reverse);
        this->sort_data();
    }

    vec<index> sort_rows_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->row_degrees, Degree_traits<r2degree>::colex_lambda());
        vec<index> reverse = vec<index>(permutation.size());
        for (index i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }
        this->transform_data(reverse);
        this->sort_data();
        return permutation;
    }

    /**
     * @brief Sorts the column degrees, stores the permutation used and then reorders the date with the same permutation.
     * 
     */
    void sort_columns_colexicographically() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->col_degrees, Degree_traits<r2degree>::colex_lambda());
        array<index> new_data = array<index>(this->data.size());
        for(index i = 0; i < this->data.size(); i++) {
            new_data[i] = this->data[permutation[i]];
        }
        this->data = new_data;
    }

    vec<index> sort_columns_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->col_degrees, Degree_traits<r2degree>::colex_lambda());
        array<index> new_data = array<index>(this->data.size());
        for(index i = 0; i < this->data.size(); i++) {
            new_data[i] = this->data[permutation[i]];
        }
        this->data = new_data;
        return permutation;
    }

    /**
     * @brief Given a vector v where v[i] indicates that we want the second row to be moved to position v[i],
     * applies this permutation to the rows/data and the row degrees.
     * 
     * @param permutation 
     */
    void permute_rows_graded(const vec<index>& permutation) {
        assert(permutation.size() == this->get_num_rows());
        this->transform_data(permutation);
        this->sort_data();
        vec<r2degree> new_row_degrees(this->get_num_rows());
        for(index i = 0; i < permutation.size(); i++) {
            new_row_degrees[permutation[i]] = this->row_degrees[i];
        }
        this->row_degrees = new_row_degrees;
    }

    private:

    template <typename T>
    void merge_unique_elements(const std::vector<std::pair<T, T>>& vec1,
                           const std::vector<std::pair<T, T>>& vec2,
                           std::vector<T>& out,
                           bool useFirst = true) {
        auto it1 = vec1.begin();
        auto it2 = vec2.begin();
        T last_x = T();
        bool has_last_x = false;

        while (it1 != vec1.end() || it2 != vec2.end()) {
            T value;

            if (it1 == vec1.end()) {
                value = useFirst ? it2->first : it2->second;
                if (!has_last_x || value != last_x) {
                    out.push_back(value);
                    last_x = value;
                    has_last_x = true;
                }
                ++it2;
            } 
            else if (it2 == vec2.end()) {
                value = useFirst ? it1->first : it1->second;
                if (!has_last_x || value != last_x) {
                    out.push_back(value);
                    last_x = value;
                    has_last_x = true;
                }
                ++it1;
            } 
            else {
                T val1 = useFirst ? it1->first : it1->second;
                T val2 = useFirst ? it2->first : it2->second;

                if (val1 < val2) {
                    if (!has_last_x || val1 != last_x) {
                        out.push_back(val1);
                        last_x = val1;
                        has_last_x = true;
                    }
                    ++it1;
                } 
                else if (val2 < val1) {
                    if (!has_last_x || val2 != last_x) {
                        out.push_back(val2);
                        last_x = val2;
                        has_last_x = true;
                    }
                    ++it2;
                } 
                else {
                    if (!has_last_x || val1 != last_x) {
                        out.push_back(val1);
                        last_x = val1;
                        has_last_x = true;
                    }
                    ++it1;
                    ++it2;
                }
            }
        }
    }


    public:

    /**
     * @brief Stores all appearing unique x and y values of column and row degrees 
     * in an ordered way in the x_grid and y_grid vectors ordered. 
     * After applying the function, the columns are ordered co(!)lexicographically and the permutation used is returned.
     * 
     */
    vec<index> compute_grid_representation() {

        auto column_degrees_lex = this->col_degrees;
        auto row_degrees_lex = this->row_degrees;
        std::sort(column_degrees_lex.begin(), column_degrees_lex.end(), Degree_traits<r2degree>::lex_lambda());
        std::sort(row_degrees_lex.begin(), row_degrees_lex.end(), Degree_traits<r2degree>::lex_lambda());
   
        x_grid.clear();
        y_grid.clear();
        z2_col_degrees.clear();
        z2_row_degrees.clear();

        x_to_index.clear();
        y_to_index.clear();

        // Reserve space to avoid repeated reallocation
        x_grid.reserve(this->get_num_cols() + this->get_num_rows());
        y_grid.reserve(this->get_num_cols() + this->get_num_rows());
        z2_col_degrees.reserve(this->get_num_cols());
        z2_row_degrees.reserve(this->get_num_rows());
        
        auto itc = column_degrees_lex.begin();
        auto itr = row_degrees_lex.begin();

        double last_x = -1;
        // Store all unique x values
        merge_unique_elements<double>(column_degrees_lex, row_degrees_lex, x_grid, true);

        auto rows_degrees_colex = this->row_degrees;
        std::sort(rows_degrees_colex.begin(), rows_degrees_colex.end(), Degree_traits<r2degree>::colex_lambda());
        vec<index> column_permutation = this->sort_columns_colexicographically_with_output();
        merge_unique_elements<double>(this->col_degrees, rows_degrees_colex, y_grid, false);

        for(index i = 0; i < x_grid.size(); i++) {
            x_to_index[x_grid[i]] = i;
        }

        for(index i = 0; i < y_grid.size(); i++) {
            y_to_index[y_grid[i]] = i;
        }
        
        // Compute Z^2 representation of degrees

        for (const auto& pair : this->col_degrees) {
            z2_col_degrees.emplace_back(x_to_index[pair.first], y_to_index[pair.second]);
        }

        for (const auto& pair : this->row_degrees) {
            z2_row_degrees.emplace_back(x_to_index[pair.first], y_to_index[pair.second]);
        }
        
        return column_permutation;
    }

    private:

        pair<index> snap_degree_to_grid_upper(const r2degree& degree, const vec<double>& x_grid, const vec<double>& y_grid) {
            auto it_x = std::lower_bound(x_grid.begin(), x_grid.end(), degree.first);
            auto it_y = std::lower_bound(y_grid.begin(), y_grid.end(), degree.second);
            index x_index = std::distance(x_grid.begin(), it_x);
            index y_index = std::distance(y_grid.begin(), it_y);
            return {x_index, y_index};
        }
        
        pair<index> snap_degree_to_grid_lower(const r2degree& degree, const vec<double>& x_grid, const vec<double>& y_grid) {
            auto it_x = std::upper_bound(x_grid.begin(), x_grid.end(), degree.first);
            auto it_y = std::upper_bound(y_grid.begin(), y_grid.end(), degree.second);
            index x_index = std::distance(x_grid.begin(), it_x)-1;
            index y_index = std::distance(y_grid.begin(), it_y)-1;
            return {x_index, y_index};
        }

    public:

    /**
     * @brief Returns the indices of the closest *smaller* grid point or -1 if to the left/bottom of the grid.
     * 
     * @param degree 
     * @return pair<index> 
     */
    pair<index> get_closest_smaller_grid_point(r2degree& degree){
        if(x_grid.empty() || y_grid.empty()){
            std::cerr << "Grid representation was not computed. Calling compute_grid_representation()." << std::endl;
            compute_grid_representation();
        }
        return snap_degree_to_grid_lower(degree, x_grid, y_grid);
    }

    /** for any grid i:G \into R^2, if this is a presentation, 
    * computes a presentation of i_! i^* X
    * by snapping all degrees to the next larger degree in x_grid x y_grid
    * (Highly inefficient implementation)
     */
    void snap_to_grid( vec<double>& new_x_grid, vec<double>& new_y_grid){

        assert(!new_x_grid.empty() && !new_y_grid.empty());
        index m = new_x_grid.size();
        index n = new_y_grid.size();
        vec<index> columns_to_remove = vec<index>();
        vec<index> rows_to_remove = vec<index>();
        for(index i = 0; i < this->get_num_cols(); i++){
            auto snapped = snap_degree_to_grid_upper(this->col_degrees[i], new_x_grid, new_y_grid);
            if(snapped.first == m || snapped.second == n){
                columns_to_remove.push_back(i);
            } else {
                this->col_degrees[i] = {new_x_grid[snapped.first], new_y_grid[snapped.second]};
            }
        }

        for(index i = 0; i < this->get_num_rows(); i++){
            auto snapped = snap_degree_to_grid_upper(this->row_degrees[i], new_x_grid, new_y_grid);
            if(snapped.first == m ||snapped.second == n){
                rows_to_remove.push_back(i);
            } else {
                this->row_degrees[i] = {new_x_grid[snapped.first], new_y_grid[snapped.second]};
            }
        }
        this->delete_columns(columns_to_remove);
        this->delete_rows(rows_to_remove);
        this->sort_columns_lexicographically();
        this->sort_rows_lexicographically();
        this->minimize();
    }

    void print_grid(){
        std::cout << "x_grid: ";
        for (const auto& x : x_grid) {
            std::cout << x << " ";
        }
        std::cout << std::endl;

        std::cout << "y_grid: ";
        for (const auto& y : y_grid) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    void print_grid_representation(){
        std::cout << "Z^2 Column Degrees: ";
        for (const auto& pair : z2_col_degrees) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;

        std::cout << "Z^2 Row Degrees: ";
        for (const auto& pair : z2_row_degrees) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief Writes the R^2 graded matrix to an output stream.
     * // print_to_stream works more generally in every dimension.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream_r2(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "2" << std::endl;
        output_stream << this->get_num_cols() << " " << this->get_num_rows() << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->get_num_cols(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->get_num_rows(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->row_degrees[i]);
            output_stream << " ;" << std::endl;
        }
    }

    /**
     * @brief used in graded_kernel. Adapted from MPfree
     * 
     */
    void kernel_column_reduction(index i, pair<index>& curr_gr, SparseMatrix<index>& column_operations, bool store_col_ops=false, bool notify_pq=false){

        index p = this->col_last(i);
        
        // Reduction loop
        while (p != -1 && this->pivots.count(p)) {
            index k = this->pivots[p];
            if( k < i){
              // Get the pivot 'k'
                this->col_op(k, i);
                if (store_col_ops) {
                    column_operations.col_op(k, i);
                }
                p = this->col_last(i);
            } else if (notify_pq) {
                index gr_y_index = this->z2_col_degrees[k].second;  

                this->pq_row[gr_y_index].push(k);
                index gr_x_index = curr_gr.first;  
                this->grid_scheduler.notify(gr_x_index, gr_y_index);
                break;
            } else {
                break;
            }
        }

        // If the column was not reduced to zero, we need to update the pivot (in any case, right?)
        if (p != -1 ) {
            this->pivots[p] = i;
        }
    }
    

    /**
     * @brief Returns a basis for the kernel of a 2d graded matrix as another 2d graded matrix.
     * Assumes that the columns are sorted lexicographically.
     * Adapted from MPfree.
     * 
     * @return SparseMatrix<index> 
     */
    R2GradedSparseMatrix graded_kernel() {
        assert(this->col_degrees.size() == this->get_num_cols());
        assert(this->row_degrees.size() == this->get_num_rows());
        vec<index> column_permutation = this->compute_grid_representation();
        this->initialise_grid_scheduler();
        
        pq_row.resize(this->y_grid.size());
        // This is the "slave" matrix in mpfree
        SparseMatrix<index> col_operations = SparseMatrix<index>(this->get_num_cols(), this->get_num_cols(), "Identity");

        std::vector<r2degree> new_degrees; // Basis for the free module which is part of the kernel
        std::vector<std::vector<index>> new_cols; // representing the matrix given by the kernel
        
        std::vector<bool> indices_in_kernel(this->get_num_cols(), false);

        // Initialize grid scheduler for processing degrees in order
        Grid_scheduler<index>& grid = this->grid_scheduler;

        while (!grid.at_end()) {

            auto new_degree = grid.next_grade();
            index x = new_degree.first;
            index y = new_degree.second;

            auto& pq = this->pq_row[y];  // Priority queue for row reduction
            auto range_xy = grid.index_range_at(x, y);

            index start_xy = range_xy.first;
            index end_xy = range_xy.second;
            assert(start_xy <= end_xy);

            // Add indices in the range to the priority queue
            for (index i = start_xy; i < end_xy; ++i) {
                pq.push(i);
            }

            while (!pq.empty()) {
                index i = pq.top();

                // Remove duplicates
                while (!pq.empty() && i == pq.top()) {
                    pq.pop();
                }

                assert(z2_col_degrees[i].first <= x);
                assert(z2_col_degrees[i].second == y);

                // Reduce the column and check if it's part of the kernel
                kernel_column_reduction(i, new_degree, col_operations, true, true);

                if (!indices_in_kernel[i] && this->is_zero(i)) {
                    std::vector<index> col = col_operations.get_col(i);
                    new_cols.push_back(std::move(col));
                    new_degrees.emplace_back(this->x_grid[x], this->y_grid[y]);
                    indices_in_kernel[i] = true;
                    // what is this for?
                    this->data[i].clear();
                    col_operations.data[i].clear();
                }
            }
        }

        // Build the resulting kernel matrix
        R2GradedSparseMatrix<index> result(new_cols.size(), this->get_num_cols());
        result.data = std::move(new_cols);
        result.col_degrees = std::move(new_degrees);
        result.row_degrees = this->col_degrees;
    
        result.permute_rows_graded(column_permutation);

        return result;
    }


    /**
     * @brief Inefficient - computes a minimization.
     * 
     * @return true 
     * @return false 
     */
    bool is_minimal() const {
        auto copy = *this;
        copy.minimize();
        return (copy.get_num_cols() == this->get_num_cols()) && (copy.get_num_rows() == this->get_num_rows());
    }



    /**
     * @brief Computes a presentation for the submodule generated at the given degree.
     * 
     * @param alpha 
     * @return R2GradedSparseMatrix 
     */
    R2GradedSparseMatrix submodule_generated_at (r2degree alpha) const {
        // Find a list of generators which map to a basis:
        vec<index> basis_lift = this->basislift_at(alpha);
        // Construct the injective graded matrix which represents these generators pushed forward to alpha
        R2GradedSparseMatrix<index> basis_injection = R2GradedSparseMatrix<index>(this->get_num_rows(), basis_lift);
        basis_injection.row_degrees = this->row_degrees;
        basis_injection.col_degrees = vec<r2degree>(basis_lift.size(), alpha);
        // Append this presentation itself
        basis_injection.append_matrix(*this);
        // Obsolete if append_matrix works correctly: basis_injection.col_degrees.insert(basis_injection.col_degrees.end(), this->col_degrees.begin(), this->col_degrees.end());
        // A kernel of this map is the pullback of this presentation along the injection
        R2GradedSparseMatrix<index> presentation = basis_injection.graded_kernel();
        
        // To get the map to the basis, forget all the rows which correspong to relations
        presentation.cull_columns(basis_lift.size(), false);
        presentation.sort_columns_lexicographically();
        presentation.minimize();
        return presentation;
    }   


    pair<r2degree> bounding_box() const{
        r2degree min = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        r2degree max = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};

        for (const auto& degree : this->col_degrees) {
            min = Degree_traits<r2degree>::meet(min, degree);
            max = Degree_traits<r2degree>::join(max, degree);
        }

        for (const auto& degree : this->row_degrees) {
            min = Degree_traits<r2degree>::meet(min, degree);
            max = Degree_traits<r2degree>::join(max, degree);
        }

        return {min, max};
    }


    
    /**
     * @brief Computes an equidistant grid of n x n points in the bounding box of the degrees of the matrix.
     * 
     * @param n 
     * @return vec<r2degree> 
     */
    vec<r2degree> get_equidistant_grid(const int& n) const {
        pair<r2degree> box = this->bounding_box();
        if(n == 0){
            return {};
        }
        if(n == 1){
            return {box.first};
        }
        double x_step = (box.second.first - box.first.first) / (n-1);
        double y_step = (box.second.second - box.first.second) / (n-1);
        vec<r2degree> grid;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                grid.push_back({box.first.first + i * x_step, box.first.second + j * y_step});
            }     
        }  
        return grid;
    }

    void snap_to_equidistant_grid(int n, bool end_inclusive = false){
        pair<r2degree> box = this->bounding_box();
        if(n == 0){
            std::cerr << "Error: Cannot snap to an equidistant grid with 0 points." << std::endl;
        } else {
            r2degree step;
            if(end_inclusive){
                step = (box.second - box.first) / (n - 1);
            } else {
                step = (box.second - box.first) / (n);
            }
            vec<double> new_x_grid;
            vec<double> new_y_grid;
            for(int i = 0; i < n; i++){
                new_x_grid.push_back(box.first.first + i * step.first);
                new_y_grid.push_back(box.first.second + i * step.second);
            }
            this->snap_to_grid(new_x_grid, new_y_grid);
        }
    };

    void cut_above(double x_cutoff, double y_cutoff){
        auto box = this->bounding_box();
        r2degree diagonal = box.second - box.first;
        auto max_degree_x = box.first.first + x_cutoff * diagonal.first;
        auto max_degree_y = box.first.second + y_cutoff * diagonal.second;
        vec<index> rows_to_remove;
        vec<index> cols_to_remove;
        for(index i = 0; i < this->get_num_cols(); i++){
            if(this->col_degrees[i].first > max_degree_x || this->col_degrees[i].second > max_degree_y){
                cols_to_remove.push_back(i);
            }
        }

        for(index i = 0; i < this->get_num_rows(); i++){
            if(this->row_degrees[i].first > max_degree_x || this->row_degrees[i].second > max_degree_y){
                rows_to_remove.push_back(i);
            }
        }

        this->delete_columns(cols_to_remove);
        this->delete_rows(rows_to_remove);
    };

    void bound_support(r2degree bound){
        vec<index> rows_to_remove;
        for(index i = 0; i < this->get_num_rows(); i++){
            if(this->row_degrees[i].first > bound.first || this->row_degrees[i].second > bound.second){
                rows_to_remove.push_back(i);
            } else {
                this->col_degrees.push_back(std::make_pair(bound.first,this->row_degrees[i].second));
                this->data.push_back( std::vector<index>({i}));
                this->col_degrees.push_back(std::make_pair(this->row_degrees[i].first, bound.second));
                this->data.push_back( std::vector<index>({i}));
            }
        }
        this->compute_num_cols();
        this->delete_rows(rows_to_remove);
        this->minimize();
    }

}; // R2GradedSparseMatrix

template<typename index>
struct R2Sequence{

    R2GradedSparseMatrix<index> first;
    R2GradedSparseMatrix<index> second;

    private:
    static std::pair<r2degree, std::vector<index>> parse_line(const std::string& line,  const bool& hasEntries = false) {
        std::istringstream iss(line);
        std::vector<index> rel;

        r2degree deg = Degree_traits<r2degree>::from_stream(iss);

        // Consume the semicolon
        std::string tmp;
        iss >> tmp;
        if(tmp != ";"){
            std::cerr << "Error: Expecting a semicolon. Invalid format in the following line: " << line << std::endl;
            std::abort();
        }

        // Parse relation
        if(hasEntries){
            index num;
            while (iss >> num) {
                rel.push_back(num);
            }
        }

        return std::move(std::make_pair(deg, rel));
    }

    void parse_stream(std::istream& file_stream) {
        std::string line;

        // Read the first line to determine the file type
        std::getline(file_stream, line);
        if (line.find("firep") != std::string::npos) {
            // Skip 2 lines for FIREP
            std::getline(file_stream, line);
            std::getline(file_stream, line);
        } else if (line.find("scc2020") != std::string::npos) {
            // Skip 1 line for SCC2020
            std::getline(file_stream, line);
        } else {
            // Invalid file type
            std::cerr << "Error: Unsupported file format. The first line must contain firep or scc2020." << std::endl;
            std::abort();
        }

        // Parse the first line after skipping
        std::getline(file_stream, line);
        std::istringstream iss(line);
        index a, b, c, zero;
        if (!(iss >> a >> b >> c >> zero) || zero != 0) {
            std::cerr << "Error: expected 4 numbers in header, last one should be 0" << std::endl;
            std::abort();
        }

        this->first.set_num_rows(c);
        this->first.set_num_cols(b);
        this->first.row_degrees.reserve(c);
        this->first.col_degrees.reserve(b);
        this->first.data.reserve(b);

        this->second.set_num_rows(b);
        this->second.set_num_cols(a);
        this->second.row_degrees.reserve(b);
        this->second.col_degrees.reserve(a);
        this->second.data.reserve(a);

        index rel_counter = 0;

        bool first_pass = true;

        while (rel_counter < a + b + c) {
            if(!std::getline(file_stream, line)){
                std::cout << "Error: Unexpected end of file. \n Make sure that the dimensions of the file are correctly given at the beginning of the file." << std::endl;
            }
            std::pair<r2degree, std::vector<index>> line_data;
            if (rel_counter < a) {
                line_data = parse_line(line, true);
                this->second.col_degrees.push_back(line_data.first);
                this->second.data.push_back(line_data.second);
                rel_counter++;
            } else if (rel_counter < a + b) {
                line_data = parse_line(line, true);
                this->second.row_degrees.push_back(line_data.first);
                this->first.col_degrees.push_back(line_data.first);
                this->first.data.push_back(line_data.second);
                rel_counter++;
            } else {
                line_data = parse_line(line, false);
                this->first.row_degrees.push_back(line_data.first);
                rel_counter++;
            }
        }
    } // Constructor from ifstream

    public:
    R2Sequence() {}

    R2Sequence(const R2GradedSparseMatrix<index>& first, const R2GradedSparseMatrix<index>& second) 
        : first(first), second(second) {}
    
    R2Sequence(const std::string& filepath) {
        first = R2GradedSparseMatrix<index>();
        second = R2GradedSparseMatrix<index>();
        std::ifstream file_stream(filepath);
        if (!file_stream.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            std::abort();
        }
        parse_stream(file_stream);
        file_stream.close();
    }

    R2Sequence(std::istream& file_stream) {
        first = R2GradedSparseMatrix<index>();
        second = R2GradedSparseMatrix<index>();
        parse_stream(file_stream);
    }

    R2GradedSparseMatrix<index> get_homology() {
        auto K = first.graded_kernel();
        second.sort_columns_lexicographically();
        second.sort_rows_lexicographically();
        K.quotient_by(second);
        return K;
    }

};


template<typename index>
struct R2Resolution {

    R2GradedSparseMatrix<index> d1;
    R2GradedSparseMatrix<index> d2;

    R2Resolution() {}

    R2Resolution(const R2GradedSparseMatrix<index>& d1, const R2GradedSparseMatrix<index>& d2) 
        : d1(d1), d2(d2) {}
    
        //TO-DO: Test this:
    R2Resolution(const R2GradedSparseMatrix<index>& d1, const bool& is_minimal = false) 
        : d1(d1) {
            // Kernel computation is easy if the presentation is minimal, sorted, and has one generator.
            if(is_minimal && d1.get_num_rows() == 1){
                // assert sorted! Todo
                // This doesnt look right at the moment.
                d2 = R2GradedSparseMatrix<index>(d1.get_num_cols()-1, d1.get_num_cols());
                d2.data = vec< vec<index> >(d1.get_num_cols()-1);
                d2.row_degrees = d1.col_degrees;
                d2.col_degrees = vec<r2degree>(d1.get_num_cols()-1);
                r2degree last_degree = d1.col_degrees[0];
                for(index i = 1; i < d1.get_num_cols(); i++){
                    r2degree join = Degree_traits<r2degree>::join(last_degree, d1.col_degrees[i]);
                    d2.data[i] = {i -1, i};
                    d2.col_degrees[i] = join;
                    r2degree last_degree = d1.col_degrees[i];
                }
            } else {
                auto d1_copy = d1;
                d2 = d1_copy.graded_kernel();
            }
        }
    
    // Copy constructor
    R2Resolution(const R2Resolution& other) 
        : d1(other.d1), d2(other.d2) {
    }
    
    // Copy assignment
    R2Resolution& operator=(const R2Resolution& other){
        if (this != &other) {
            d1 = other.d1;
            d2 = other.d2;
        }
        return *this;
    }
    
    // Move constructor
    R2Resolution(R2Resolution&& other) = default;
    
    // Move assignment
    R2Resolution& operator=(R2Resolution&& other) = default;

    /**
     * @brief Writes the R^2 resolution to an output stream.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "2" << std::endl;
        output_stream << this->d2.get_num_cols() << " " << this->d2.get_num_rows() << " " << this->d1.get_num_rows() << std::endl;

        // Write the syzygies
        for (index i = 0; i < this->d2.get_num_cols(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->d2.col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->d2.data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the relations
        for (index i = 0; i < this->d1.get_num_cols(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->d1.col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->d1.data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the generators
        for (index i = 0; i < this->d1.get_num_rows(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->d1.row_degrees[i]);
            output_stream << " ;" << std::endl;
        }
    }

    
    /**
     * @brief Writes the R^2 resolution to an output stream.
     * 
     * @param alpha 
     * @return index 
     */
    index dim_at (r2degree alpha) const {
        index num_chains_0 = d1.num_rows_before(alpha);
        index num_chains_1 = d1.num_cols_before(alpha);
        index num_chains_2 = d2.num_cols_before(alpha);
        return num_chains_0 - num_chains_1 + num_chains_2;
    }

    /**
     * @brief Computes the dimension at every point in the grid. Unoptimised. Not finished.
     * 
     * @return array<index> 
     */
    array<index> dimension_vector(bool sort = true) const {
        if(sort){
            d1.sort_rows_colexicographically();
            d1.sort_columns_colexicographically();
            d2.sort_rows_colexicographically();
            d2.sort_columns_colexicographically();
        }

        vec<double> x_grid = d1.x_grid;
        vec<double> y_grid = d1.y_grid;
        index num_x = x_grid.size();
        index num_y = y_grid.size();

        auto itc1 = d1.row_degrees.begin();
        auto itc2 = d2.row_degrees.begin();
        auto itc3 = d2.col_degrees.begin();

        for(index i = 0; i < num_y; i++){
            
        }
    }

    array<index> dimension_vector_non_opt(index& max_value){

        d1.compute_grid_representation();
        d2.compute_grid_representation();

            d1.sort_rows_colexicographically();
            d1.sort_columns_colexicographically();
            d2.sort_rows_colexicographically();
            d2.sort_columns_colexicographically();
        

        vec<double> x_grid = d1.x_grid;
        vec<double> y_grid = d1.y_grid;
        index num_x = x_grid.size();
        index num_y = y_grid.size();
        
        const auto& generators = d1.row_degrees;
        const auto& relations = d1.col_degrees;
        const auto& syzygies = d2.col_degrees;
        
        array<index> hilbert = array<index>(num_x, vec<index>(num_y, 0));
        max_value = 0;
        
        for (index j = 0; j < num_y; j++) {
            double y = y_grid[j];
            for (index i= 0; i < num_x; i++) {
                double x = x_grid[i];
                int val = 0;
                
                // Add generators and syzygies
                for (const auto& [gx, gy] : generators) 
                    if (x >= gx && y >= gy) val++;
                for (const auto& [sx, sy] : syzygies) 
                    if (x >= sx && y >= sy) val++;
                
                // Subtract relations
                for (const auto& [rx, ry] : relations) 
                    if (x >= rx && y >= ry) val--;
                
                assert(val >= 0);
                hilbert[i][j] = val;
                max_value = std::max(max_value, val);
            }
        }
        return hilbert;
    }

};

} // namespace graded_linalg

#endif // R2GRADED_MATRIX_HPP

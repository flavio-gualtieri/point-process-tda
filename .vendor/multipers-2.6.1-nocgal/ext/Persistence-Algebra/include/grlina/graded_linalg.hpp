/**
 * @file graded_linalg.hpp
 * @author Jan Jendrysiak
 * @brief This library was created to provide a set of tools for working with (graded) matrices over F_2 for the
 *        AIDA algorithm which decomposes minimal presentations of persistence modules. 
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

#ifndef GRADED_LINALG_HPP
#define GRADED_LINALG_HPP


#include <vector>
#include <grlina/general.hpp>
#include <grlina/sparse_matrix.hpp>
#include <grlina/graded_matrix.hpp>
#include <grlina/matrix_base.hpp>
#include <grlina/graded_matrix.hpp>
#include <grlina/r2graded_matrix.hpp>
#include <grlina/r3graded_matrix.hpp>
#include <grlina/dense_matrix.hpp>
#include <grlina/homomorphisms.hpp>
#include <grlina/to_quiver.hpp>

namespace graded_linalg {

/**
 * @brief Converts a dense matrix to a sparse matrix
 * 
 * @tparam index 
 * @param matrix 
 * @return SparseMatrix<index> 
 */
template <typename index>
SparseMatrix<index> sparse_from_dense(DenseMatrix& matrix){
    SparseMatrix<index> result = SparseMatrix<index>(matrix.get_num_cols(), matrix.get_num_rows());
    for(index i = 0; i < matrix.get_num_cols(); i++){
        result.data.push_back(vec<index>());
        for(index j = 0; j < matrix.get_num_rows(); j++){
            if(matrix.data[i][j] == 1){
                result.data[i].push_back(j);
            }
        }
    }
    return result;
}

template <typename index>
vec<vec<SparseMatrix<index>>> all_sparse_proper_subspaces(index k){
    vec<vec<SparseMatrix<index>>> result = vec<vec<SparseMatrix<index>>>(k);
    for(index i = 0; i < k; i++){
    
        vec<DenseMatrix> i_spaces = all_proper_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            result[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
    return result;
}

template <typename index>
vec<vec<SparseMatrix<index>>> all_sparse_subspaces(index k){
    vec<vec<SparseMatrix<index>>> result = vec<vec<SparseMatrix<index>>>(k);
    for(index i = 0; i < k; i++){
    
        vec<DenseMatrix> i_spaces = all_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            result[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
    return result;
}


template <typename index>
void fill_up_subspaces (vec<vec<SparseMatrix<index>>>& subspaces, index k ){

    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<SparseMatrix<index>>());
        vec<DenseMatrix> i_spaces = all_proper_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            subspaces[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
}

template <typename index>
vec<vec<vec<SparseMatrix<index>>>> sparse_seperated_grassmannians(index k){
    vec<vec<vec<SparseMatrix<index>>>> result = vec<vec<vec<SparseMatrix<index>>>>(k, vec<vec<SparseMatrix<index>>>());
    for(index i = 0; i < k; i++){
        for(index j = 0; j <= i+1; j++){
            result[i].emplace_back(vec<SparseMatrix<index>>());
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                result[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
    return result;
}

template <typename index>
void fill_up_seperated_grassmannians(vec<vec<vec<SparseMatrix<index>>>>& subspaces, index k){
    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<vec<SparseMatrix<index>>>());
        for(index j = 0; j <= i+1; j++){
            subspaces[i].emplace_back(vec<SparseMatrix<index>>());
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                subspaces[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
}

template <typename index>
vec<vec<vec<SparseMatrix<index>>>> all_sparse_grassmannians(index k, index n){
    vec<vec<vec<SparseMatrix<index>>>> result = vec<vec<vec<SparseMatrix<index>>>>(k);
    for(index i = 0; i < k; i++){
        for(index j = 0; j <= i+1; j++){
            result[i].emplace_back(vec<SparseMatrix<index>>());
            if(j > n){
                continue;
            }
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                result[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
    return result;
}

template <typename index>
void fill_up_grassmannians (vec<vec<vec<SparseMatrix<index>>>>& subspaces, index k, index n){
    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<vec<SparseMatrix<index>>>());
        for(index j = 0; j <= i+1; j++){
            subspaces[i].emplace_back(vec<SparseMatrix<index>>());
            if(j > n){
                continue;
            }
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                subspaces[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
}

/**
 * @brief Constructs a vector of R2GradedSparseMatrix objects from an input file stream.
 * 
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @param compute_batches whether to compute the column batches and k_max
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename index>
std::vector<R2GradedSparseMatrix<index>> get_matrices_from_stream(std::ifstream& file_stream, bool lex_sort = false, bool compute_batches = false) {
    std::vector<R2GradedSparseMatrix<index>> matrices;

    while (file_stream) {
        // Construct a new matrix from the stream
        R2GradedSparseMatrix<index> matrix(file_stream, lex_sort, compute_batches);
        matrices.push_back(std::move(matrix));

        // Check if the stream is exhausted
        if (file_stream.eof()) {
            break;
        }
    }

    return matrices;
}

/**
 * @brief Constructs a vector of R2GradedSparseMatrix objects from an input file stream.
 * 
 * @param matrices vector to store the constructed matrices
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @param compute_batches whether to compute the column batches and k_max
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename index, typename InputStream>
void construct_matrices_from_stream(std::vector<R2GradedSparseMatrix<index>>& matrices, InputStream& file_stream, bool lex_sort = false, bool compute_batches = false) {

    while (file_stream) {
        // Construct a new matrix from the stream
        R2GradedSparseMatrix<index> matrix(file_stream, lex_sort, compute_batches);
        matrices.push_back(std::move(matrix));

        // Skip empty lines
        std::string line;
        while (std::getline(file_stream, line)) {
            if (!line.empty() && line.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
                file_stream.seekg(-static_cast<int>(line.length()) - 1, std::ios_base::cur);
                break;
            }
        }

        if (file_stream.eof()) {
            break;
        }
    }

}

/**
 * @brief Writes a vector of numbers (arithmetic type) to a .txt file.
 * 
 * @tparam T 
 * @param vec 
 * @param folder 
 * @param filename 
 */
template <typename T>
void write_vector_to_file(const std::vector<T>& vec, const std::string& folder, const std::string& filename) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a numeric type");

    // Construct the full path by combining the folder and filename
    std::string full_path = folder + "/" + filename;

    std::ofstream outfile(full_path);
    
    if (!outfile) {
        std::cerr << "Error: Could not open the file " << full_path << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        outfile << vec[i];
        if (i != vec.size() - 1) {
            outfile << " ";  // Use space as a separator
        }
    }

    outfile.close();
    
    if (!outfile) {
        std::cerr << "Error: Could not close the file " << full_path << " after writing." << std::endl;
    }
}

/**
 * @brief Constructs matrices from stream with header and type annotations
 *
 * @param matrices vector to store the constructed matrices
 * @param file_stream input stream with format: header, count, then (type, matrix) pairs
 * @param lex_sort whether to sort lexicographically
 * @param compute_batches whether to compute the column batches and k_max
 * @param matrix_types optional vector to store the type of each matrix (e.g., "cyclic")
 */
template <typename index, typename InputStream>
void read_sccsum(
    std::vector<R2GradedSparseMatrix<index>>& matrices, 
    InputStream& file_stream, 
    bool lex_sort = false, 
    bool compute_batches = false,
    std::vector<std::string>* matrix_types = nullptr) {
    
    // Read and validate header
    std::string header;
    std::getline(file_stream, header);
    if (header != "scc2020sum") {
        throw std::runtime_error("Invalid header: expected 'scc2020sum', got '" + header + "'");
    }
    
    // Read matrix count
    size_t count;
    file_stream >> count;
    
    matrices.reserve(count);
    if (matrix_types) {
        matrix_types->reserve(count);
    }
    
    for (size_t i = 0; i < count; ++i) {
        // Skip empty lines and read type
        std::string type;
        while (std::getline(file_stream, type) && type.empty()) {}
        
        if (matrix_types) {
            matrix_types->push_back(type);
        }
        
        // Construct matrix from stream
        R2GradedSparseMatrix<index> matrix(file_stream, lex_sort, compute_batches);
        matrices.push_back(std::move(matrix));
    }
}

/**
 * @brief Writes a vector of sets of numbers to a .txt file.
 * 
 * @tparam T 
 * @param vec 
 * @param relative_folder 
 * @param filename 
 */
template <typename T>
void write_vector_of_sets_to_file(const std::vector<std::set<T>>& vec, const std::string& relative_folder, const std::string& filename) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a numeric type");

    std::cout << "Writing vector of length " << vec.size() << " to " << filename << " in folder " << relative_folder << std::endl;

    // Construct the full path by combining the relative folder and filename
    std::string full_path = relative_folder + "/" + filename;

    std::ofstream outfile(full_path);
    
    if (!outfile) {
        std::cerr << "Error: Could not open the file " << full_path << " for writing." << std::endl;
        return;
    }

    for (const auto& s : vec) {
        outfile << "{";
        for (auto it = s.begin(); it != s.end(); ++it) {
            outfile << *it;
            if (std::next(it) != s.end()) {
                outfile << ", ";
            }
        }
        outfile << "}\n";
    }

    outfile.close();
    
    if (!outfile) {
        std::cerr << "Error: Could not close the file " << full_path << " after writing." << std::endl;
    }
}


/**
 * @brief If the two streams contain lists of graded matrices, then
 * this function returns false if they have non-matching degrees.
 * 
 */
template <typename index, typename DERIVED>
bool compare_streams_of_graded_matrices(std::ifstream& stream1, std::ifstream& stream2) {
    vec<R2GradedSparseMatrix<index>> matrices1;
    construct_matrices_from_stream(matrices1, stream1, false, false);
    vec<R2GradedSparseMatrix<index>> matrices2;
    construct_matrices_from_stream(matrices2, stream2, false, false);
    if(matrices1.size() != matrices2.size()){
        return false;
    }
    std::sort(matrices1.begin(), matrices1.end(), Compare_by_degrees<r2degree, index, DERIVED>());
    std::sort(matrices2.begin(), matrices2.end(), Compare_by_degrees<r2degree, index, DERIVED>());
    for(index i = 0; i < matrices1.size(); i++){
        if( Compare_by_degrees<r2degree, index, DERIVED>::compare_three_way(matrices1[i], matrices2[i]) != 0){
            return false;
        }
    }
    return true;
}

/**
 * @brief If the two files contain lists of graded matrices, then
 * this function returns false if they have non-matching degrees.
 * 
 * @param path1 
 * @param path2 
 * @return true 
 * @return false 
 */
template <typename index>
bool compare_files_of_graded_matrices(std::string path1, std::string path2) {
    std::ifstream stream1(path1);
    std::ifstream stream2(path2);
    return compare_streams_of_graded_matrices<index>(stream1, stream2);
}
    

/**
 * @brief Gets the Minor of M with respect to the row and column indices and saves the result as a DenseMatrix.
 *  Not tested yet!
 * @tparam index 
 * @param M 
 * @param col_indices 
 * @param row_indices 
 * @return DenseMatrix 
 */
/**
template <typename index>
DenseMatrix restricted_dense_copy(const SparseMatrix<index>& M, vec<index> col_indices, vec<index> row_indices) {
    DenseMatrix<int> result(col_indices.size(), row_indices.size());
    for(index i = 0; i < col_indices.size(); i++){
        auto c = col_indices[i];
        auto it = std::lower_bound(M.data[c].begin(), M.data[c].end(), row_indices[0]);
        index j = 0;
        while(it != M.data[c].end() && j < row_indices.size()){
            auto r = row_indices[j];
            if(*it == r){
                result.data[i].set(j);
                j++;
                it++;
            } else if(*it < r){
                it++;
            } else {
                j++;
            }
        }
    }
}
*/

template <typename index>
DenseMatrix from_sparse (SparseMatrix<index>& M) {
    DenseMatrix result(M.get_num_cols(), M.get_num_rows());
    result.data = vec<bitset>(M.get_num_cols(), bitset(M.get_num_rows()));
    for(int i = 0; i < M.get_num_cols(); i++){
        for(index j : M.data[i]){
            result.data[i].set(j);
        }
    }
    return result;
}



} // namespace graded_linalg

#endif // GRADED_LINALG_HPP

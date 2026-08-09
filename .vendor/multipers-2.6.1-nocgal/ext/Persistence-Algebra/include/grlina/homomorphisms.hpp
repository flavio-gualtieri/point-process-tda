/**
 * @file homomorphisms.hpp
 * @author Jan Jendrysiak
 * @brief different methods to compute homomorphisms of persistence modules.
 * @version 0.2
 * @date 2026-02-05
 * 
 * @copyright 2026 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once


#ifndef HOMOMORPHISMS_HPP
#define HOMOMORPHISMS_HPP

#include <grlina/graded_matrix.hpp>
#include "grlina/r2graded_matrix.hpp"
#include "grlina/sparse_matrix.hpp"
#include <cstdlib>
#include <utility>
#include <random>
#include <vector>
#include <boost/timer/timer.hpp>

namespace graded_linalg {

    template <typename index>
double avg_entries_per_column(const SparseMatrix<index>& S){
    double total_entries = 0.0;
    for(const auto& col : S.data){
        total_entries += col.size();
    }
    return total_entries / (double) S.get_num_cols();
}

template <typename index>
void system_stats(boost::timer::cpu_timer& timer, std::string alg_name, SparseMatrix<index>& S, const int rows = -1){
    std::cout << alg_name << ":" << std::endl;
    timer.stop();
    std::cout << "  Time to set up: " << timer.elapsed().wall * 1e-6 << "ms " << std::endl;
    std::cout << "  System size: " << S.get_num_cols() << " cols and ";
    if(rows == -1){
        std::cout << S.get_num_rows() << " rows. " << std::endl;
    } else {
        std::cout << rows << " rows." << std::endl;
    }
    std::cout << "  Average #entries per column: " << avg_entries_per_column(S) << std::endl;
    timer.start();
}

template <typename index>
std::vector<index> generate_random_indices(index number, index range) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index> dist(0, range - 1);
    
    std::vector<index> result(number);
    for (index i = 0; i < number; ++i) {
        result[i] = dist(gen);
    }
    return result;
}

template <typename index>
vec<index>index_pair_to_position(index row_index,
    index j_start, 
    const vec<index>& B_data_j, 
    const vec<std::pair<index,index>>& variable_positions) {

    vec<index> result;
    auto it = B_data_j.begin();
    for(index j = j_start; j < variable_positions.size() && it != B_data_j.end(); j++){
        auto& index_pair = variable_positions[j];
        if(index_pair.first == row_index){
            if(*it == index_pair.second){
                result.push_back(j);
                it++;
            }
        }
    }
    return result;
}

/**
 * @brief Returns the basis of Hom(A, B) as a vector of graded matrices.
 *
 * @tparam D
 * @tparam index
 * @tparam DERIVED
 * @param A
 * @param B
 * @return vec<DERIVED>
 */
template <typename D, typename index, typename DERIVED>
vec<DERIVED> hom_space_basis_new(
    const GradedSparseMatrix<D, index, DERIVED>& A,
    const GradedSparseMatrix<D, index, DERIVED>& B, 
    bool use_hom_exactness = false,
    const bool info = false
    ) {

    boost::timer::cpu_timer timer;
    if(info){
        timer.start();
    }
    vec<DERIVED> result = vec<DERIVED>();
    if(!A.rows_computed){
        std::cerr << "Warning: Rows of A must be computed before usage." << std::endl;
        std::abort();
    }

    if(! use_hom_exactness){

    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    vec<index> variable_positions_separator = vec<index>(A.get_num_rows(), 0);
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    for(index i = 0; i < A.get_num_rows(); i++) {
        variable_positions_separator[i] = S_index;
        for(index j = 0; j < B.get_num_rows(); j++) {
            if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.row_degrees[j])){
                S.data.push_back(vec<index>());
                variable_positions.push_back(std::make_pair(i, j));
                for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            }
        }
    }

    index row_op_threshold = S_index;
    assert( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return result;
    }


    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            }
        }
    }

    

    S.compute_num_cols();
    auto K = S.kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();
    K.column_reduction_triangular(true);

    bool reduce = false;
    if(reduce){
        SparseMatrix<index> N_bar = SparseMatrix<index>(0,K.get_num_cols());
        for(index i = 0; i < A.get_num_rows(); i++){
            for(index j = 0; j < B.get_num_cols(); j++){
                if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.col_degrees[j])){
                    // Add a new homotopy
                    vec<index> h = index_pair_to_position(i,variable_positions_separator[i], B.data[j], variable_positions);
                    N_bar.data.push_back(h);
                }
            }
        }
        N_bar.compute_num_cols();
        N_bar.reduce_fully(K);
        K.column_reduction_triangular(true);
    }

    for(index i = 0; i < K.get_num_cols(); i++){
        auto& current_map = K.data[i];
        DERIVED Q(A.get_num_rows(), B.get_num_rows(),
            vec< vec<int> >( A.get_num_rows(), vec<int>() ),
            A.row_degrees, B.row_degrees);
        for(index j : current_map){
            auto& index_pairs = variable_positions[j];
            Q.data[index_pairs.first].push_back(index_pairs.second);
        }
        assert(Q.is_sorted_sparse());
        result.push_back(Q);
    } 

    //  If the bool is true, next use Hom( -, coker B) on the presentation A to get Hom(coker A, coker B)_0 as the kernel of 
    //  Hom(A.target, coker B)_0 -(A^t)-> Hom(A.domain, coker B)_0
    } else {

        vec<SparseMatrix<index>> B_local_spaces = vec<SparseMatrix<index>>();
        B_local_spaces.reserve(A.get_num_rows() + A.get_num_cols());
        vec< std::pair<vec<index>, vec<index>> > B_local_basislifts =  vec< std::pair<vec<index>, vec<index>> >();
        B_local_basislifts.reserve(A.get_num_rows() + A.get_num_cols());

        // Setting up all structure maps for coker B
        array<index> S_column_partition = array<index>();
        index S_num_col = 0;
        for(size_t p = 0; p < A.get_num_rows() + A.get_num_cols(); p++){
            SparseMatrix<index> presentation;
            vec<index> gens; // the set of generators which appear before the selected degree.
            if(p < A.get_num_rows()){
                std::tie(presentation, gens) = B.map_at_degree_pair(A.row_degrees[p]);
            } else {
                std::tie(presentation, gens) = B.map_at_degree_pair(A.col_degrees[p - A.get_num_rows()]);             
            }
            vec<index> basisLift; // Will contain a set of indices, which indicates a minimal set of (local!) generators that maps to a basis of the local vector space
            B_local_spaces.emplace_back(presentation.coKernel(false, &basisLift));
            B_local_basislifts.push_back(std::make_pair(gens, basisLift));
            if(p < A.get_num_rows()){
                S_num_col += basisLift.size();
            }
        }
        SparseMatrix<index> S(S_num_col, 0, array<index>(S_num_col, vec<index>()));
        index S_column_counter = 0;
        index S_row_counter = 0;
        
        // building the matrix A^t \otimes B_*
        for(size_t i = 0; i < A.get_num_rows(); i++){
            S_row_counter = 0;
            std::pair< vec<index>, vec<index> >& sourceBasis = B_local_basislifts[i];
            assert(B_local_spaces[i].get_num_cols() == sourceBasis.first.size());
            assert(B_local_spaces[i].get_num_rows() == sourceBasis.second.size());
            auto itA = A._rows[i].begin();
            // The following will store the global indices corresponding to a basis of Y_(deg g_i)
            S_column_partition.push_back(vec_restriction(sourceBasis.first, sourceBasis.second));
            for(size_t j = 0; j < A.get_num_cols(); j++){
                if(itA == A._rows[i].end()){
                    break;
                } else if (*itA > j) {
                    index j_shift = A.get_num_rows()+j;
				    auto& targetBasis = B_local_basislifts[j_shift];
                    S_row_counter += targetBasis.second.size();
                    continue;
                } else {
                assert(*itA == j);    
                itA++;
                index j_shift = A.get_num_rows()+j;
				auto& targetBasis = B_local_basislifts[j_shift];
                auto& targetSpace = B_local_spaces[j_shift];
				assert(targetSpace.get_num_cols() == targetBasis.first.size());
                assert(targetSpace.get_num_rows() == targetBasis.second.size());
				vec<index> image_of_basis_indices = get_index_vector<index, index>(targetBasis.first, S_column_partition[i]);
                // ^ First computes the actual indices of the genertors which form a basis of the source vector space.
                // Then computes which subset of the generators of the target this corresponds to.
                
                assert(image_of_basis_indices.size() == sourceBasis.second.size());
				// auto local_map = targetSpace.restricted_domain_copy(image_of_basis_indices);
                // The map (coker B)_{gen[i]->rel[j]} is now given by targetSpace restricted to image_of_basis_indices
                // Since we need to translate this in S, add column and row counter
                for(index k = 0; k < image_of_basis_indices.size(); k++){
                    auto& basis_vector = image_of_basis_indices[k];
                    for(index entry : targetSpace.data[basis_vector]){
                        S.data[S_column_counter + k].push_back(entry+S_row_counter);
                    }
                }
                S_row_counter += targetBasis.second.size();
                }
            }
            S_column_counter += S_column_partition[i].size();
        }
        S.set_num_rows(S_row_counter);

        if(info){
            system_stats(timer, "Alg_B", S);
        }

        SparseMatrix<index> K = S.kernel();

        if(info){
            timer.stop();
            std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
            // std::cout << "Dimension of hom-space: " << K.get_num_cols() << std::endl;
        }

        if(K.get_num_cols() > 100000){
            std::cout << " Hom-space is too large (" << K.get_num_cols() << ") to convert to a vector of matrices." << std::endl;
            return result;
        }
        // Translating the vectors to matrices.
        // TO-DO: The following could easily be parallelised
        for(auto f_vec : K.data){
            DERIVED new_Q(A.get_num_rows(), B.get_num_rows(), A.row_degrees, B.row_degrees);
            result.emplace_back(std::move(new_Q));
            auto& Q = result.back().data;
            Q.resize(A.get_num_rows());
            auto it = f_vec.begin();
            index column_counter = 0;
            // We need to advance the iterator for the next block of size S_column_partition[i]
            for(index i = 0; i < A.get_num_rows(); i++){
                index block_end = column_counter + S_column_partition[i].size();
                while(it != f_vec.end() && *it <= block_end){
                    Q[i].push_back(S_column_partition[i][*it - column_counter]);
                    it++;
                }
                if( it == f_vec.end()){
                    break;
                }
                column_counter += S_column_partition[i].size();
            }
        }
    }

    return result;
}

/**
 * @brief Returns the basis of Hom(A, B) as a vector of graded matrices.
 *
 * @tparam D
 * @tparam index
 * @tparam DERIVED
 * @param A
 * @param B
 * @return vec<DERIVED>
 */
template <typename D, typename index, typename DERIVED>
SparseMatrix<index> Alg_B_test(
    const GradedSparseMatrix<D, index, DERIVED>& A,
    const GradedSparseMatrix<D, index, DERIVED>& B, 
    const bool info = false
    ) {

    boost::timer::cpu_timer timer;
    if(info){
        timer.start();
    }
    vec<DERIVED> result = vec<DERIVED>();
    if(!A.rows_computed){
        std::cerr << "Warning: Rows of A must be computed before usage." << std::endl;
        std::abort();
    }

        vec<SparseMatrix<index>> B_local_spaces = vec<SparseMatrix<index>>();
        B_local_spaces.reserve(A.get_num_rows() + A.get_num_cols());
        vec< std::pair<vec<index>, vec<index>> > B_local_basislifts =  vec< std::pair<vec<index>, vec<index>> >();
        B_local_basislifts.reserve(A.get_num_rows() + A.get_num_cols());

        // Setting up all structure maps for coker B
        array<index> S_column_partition = array<index>();
        index S_num_col = 0;
        for(size_t p = 0; p < A.get_num_rows() + A.get_num_cols(); p++){
            SparseMatrix<index> presentation;
            vec<index> gens; // the set of generators which appear before the selected degree.
            if(p < A.get_num_rows()){
                std::tie(presentation, gens) = B.map_at_degree_pair(A.row_degrees[p]);
            } else {
                std::tie(presentation, gens) = B.map_at_degree_pair(A.col_degrees[p - A.get_num_rows()]);             
            }
            vec<index> basisLift; // Will contain a set of indices, which indicates a minimal set of (local!) generators that maps to a basis of the local vector space
            B_local_spaces.emplace_back(presentation.coKernel(false, &basisLift));
            B_local_basislifts.push_back(std::make_pair(gens, basisLift));
            if(p < A.get_num_rows()){
                S_num_col += basisLift.size();
            }
        }
        SparseMatrix<index> S(S_num_col, 0, array<index>(S_num_col, vec<index>()));
        index S_column_counter = 0;
        index S_row_counter = 0;
        
        // building the matrix A^t \otimes B_*
        for(size_t i = 0; i < A.get_num_rows(); i++){
            S_row_counter = 0;
            std::pair< vec<index>, vec<index> >& sourceBasis = B_local_basislifts[i];
            assert(B_local_spaces[i].get_num_cols() == sourceBasis.first.size());
            assert(B_local_spaces[i].get_num_rows() == sourceBasis.second.size());
            auto itA = A._rows[i].begin();
            // The following will store the global indices corresponding to a basis of Y_(deg g_i)
            S_column_partition.push_back(vec_restriction(sourceBasis.first, sourceBasis.second));
            for(size_t j = 0; j < A.get_num_cols(); j++){
                if(itA == A._rows[i].end()){
                    break;
                } else if (*itA > j) {
                    index j_shift = A.get_num_rows()+j;
				    auto& targetBasis = B_local_basislifts[j_shift];
                    S_row_counter += targetBasis.second.size();
                    continue;
                } else {
                assert(*itA == j);    
                itA++;
                index j_shift = A.get_num_rows()+j;
				auto& targetBasis = B_local_basislifts[j_shift];
                auto& targetSpace = B_local_spaces[j_shift];
				assert(targetSpace.get_num_cols() == targetBasis.first.size());
                assert(targetSpace.get_num_rows() == targetBasis.second.size());
				vec<index> image_of_basis_indices = get_index_vector<index, index>(targetBasis.first, S_column_partition[i]);
                // ^ First computes the actual indices of the genertors which form a basis of the source vector space.
                // Then computes which subset of the generators of the target this corresponds to.
                
                assert(image_of_basis_indices.size() == sourceBasis.second.size());
				// auto local_map = targetSpace.restricted_domain_copy(image_of_basis_indices);
                // The map (coker B)_{gen[i]->rel[j]} is now given by targetSpace restricted to image_of_basis_indices
                // Since we need to translate this in S, add column and row counter
                for(index k = 0; k < image_of_basis_indices.size(); k++){
                    auto& basis_vector = image_of_basis_indices[k];
                    for(index entry : targetSpace.data[basis_vector]){
                        S.data[S_column_counter + k].push_back(entry+S_row_counter);
                    }
                }
                S_row_counter += targetBasis.second.size();
                }
            }
            S_column_counter += S_column_partition[i].size();
        }
        S.set_num_rows(S_row_counter);

        if(info){
            system_stats(timer, "Alg_B", S);
        }

        SparseMatrix<index> K = S.kernel();

        if(info){
            timer.stop();
            std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
            // std::cout << "Dimension of hom-space: " << K.get_num_cols() << std::endl;
        }

    return K;
}

    
/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 *  make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index, typename DERIVED>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_optimised(const GradedSparseMatrix<D, index, DERIVED>& A, const GradedSparseMatrix<D, index, DERIVED>& B, 
    const vec<index>& row_indices_A = vec<index>(), const vec<index>& row_indices_B = vec<index>(),
    const bool info = false)  {
    
    assert(A.rows_computed);
    boost::timer::cpu_timer timer;
    if(info)
        timer.start();

    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    // TO-DO: Right now we compute map_at_degree possibly multiple times! This could be optimised.
    for(index i = 0; i < A.get_num_rows(); i++) {
        // Compute the target space B_alpha for each generator of A to minimise the number of variables.
        auto [B_alpha, rows_alpha] = B.map_at_degree_pair(A.row_degrees[i], false);
        vec<index> basislift;
        if(row_indices_B.size() != 0){
            basislift = B_alpha.coKernel_basis(rows_alpha, row_indices_B );
        } else {
            basislift = B_alpha.coKernel_basis_local(rows_alpha);
        }

        // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
    	    variable_positions.push_back(std::make_pair(i, j));
            for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                auto& column_index = *rit;
                S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
            }
            S_index++;
        }
    }
    
    index row_op_threshold = S_index;
    assert( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }

    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    if(row_indices_B.size() != 0){
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.get_num_cols(), B.get_num_rows()));
                    } else {
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));        
                    }
                }
                S_index++;
            }
        }
    }

    S.compute_num_cols();

    if(info){
        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(Degree_traits<D>::greater_equal(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        system_stats(timer, "Mixed", S, equation_counter);
    }

    auto K = S.kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();

    if(info){
        timer.stop();
        std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        timer.start();
    }

    K.column_reduction_triangular(true);

    if(info){
        timer.stop();
        std::cout << "  Time to reduce: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        // std::cout << "Dimension of hom-space: " << K.get_num_cols() << std::endl;
    }

    return std::make_pair(K, variable_positions);
}


/**
 * @brief Algorithm A of "Homomorphisms of Persistence Modules"
 * Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 * make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index, typename DERIVED>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_full_restriction(const GradedSparseMatrix<D, index, DERIVED>& A, const GradedSparseMatrix<D, index, DERIVED>& B, 
    const vec<index>& row_indices_A = vec<index>(), const vec<index>& row_indices_B = vec<index>(),
    const bool info = false)  {
    
    assert(A.rows_computed);
    boost::timer::cpu_timer timer;
    if(info)
        timer.start();

    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    // TO-DO: Right now we compute map_at_degree possibly multiple times! This could be optimised.
    for(index i = 0; i < A.get_num_rows(); i++) {
        // Compute the target space B_alpha for each generator of A to minimise the number of variables.
        auto [B_alpha, rows_alpha] = B.map_at_degree_pair(A.row_degrees[i], false);
        vec<index> basislift;
        if(row_indices_B.size() != 0){
            basislift = B_alpha.coKernel_basis(rows_alpha, row_indices_B );
        } else {
            basislift = B_alpha.coKernel_basis_local(rows_alpha);
        }

        // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
    	    variable_positions.push_back(std::make_pair(i, j));
            for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                auto& column_index = *rit;
                S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
            }
            S_index++;
        }
    }
    
    index row_op_threshold = S_index;
    assert( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }
    DERIVED B_copy = B;
    DERIVED O = B_copy.graded_kernel();
    // Then all column-operations from B to A, again restricted by the basislift
    for(index i = A.get_num_cols()-1; i > -1; i--){

        // Compute the target space O_alpha for each relation of A to minimise the number of variables.
        auto [O_alpha, rows_alpha] = O.map_at_degree_pair(A.col_degrees[i], false);
        vec<index> basislift = O_alpha.coKernel_basis_local(rows_alpha);
        
         // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
            for(index row_index : B.data[j]){
                if(row_indices_B.size() != 0){
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.get_num_cols(), B.get_num_rows()));
                } else {
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));
                }
            }
            S_index++;
        }
    }

    S.compute_num_cols();

    if(info){
        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(Degree_traits<D>::greater_equal(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        system_stats(timer, "Alg_A", S, equation_counter);
    }


    auto K = S.kernel();

    if(info){
        timer.stop();
        std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        timer.start();
    }

    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();

    return std::make_pair(K, variable_positions);
}

template <typename D, typename index, typename DERIVED>
vec<index> no_opt_system_info(const GradedSparseMatrix<D, index, DERIVED>& A, const GradedSparseMatrix<D, index, DERIVED>& B){
    assert(A.rows_computed);
    boost::timer::cpu_timer timer;
    
    timer.start();
    vec<index> result = vec<index>();

    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    for(index i = 0; i < A.get_num_rows(); i++) {
        for(index j = 0; j < B.get_num_rows(); j++) {
            if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.row_degrees[j])){
                S.data.push_back(vec<index>());
                variable_positions.push_back(std::make_pair(i, j));
                for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            } 
        }
    }
    
    index row_op_threshold = S_index;
    assert( variable_positions.size() == S_index );


    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));        
                }
                S_index++;
            }
        }
    }


    S.compute_num_cols();


        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(Degree_traits<D>::greater_equal(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        result.push_back(S.get_num_cols());
        result.push_back(equation_counter);
        system_stats(timer, "Naive", S, equation_counter);
    return result;
}

/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 *  make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index, typename DERIVED>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_no_opt(
    const GradedSparseMatrix<D, index, DERIVED>& A, 
    const GradedSparseMatrix<D, index, DERIVED>& B, 
    const bool reduce = true,
    const vec<index>& row_indices_A = vec<index>(), 
    const vec<index>& row_indices_B = vec<index>(), 
    const bool info = false)  {
    
    assert(A.rows_computed);
    boost::timer::cpu_timer timer;
    if(info)
        timer.start();
    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    vec<index> variable_positions_separator = vec<index>(A.get_num_rows(), 0); // To not have to search through the entire variable_positions vector
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    for(index i = 0; i < A.get_num_rows(); i++) {
        if(reduce){
            variable_positions_separator[i] = S_index;
        }
        for(index j = 0; j < B.get_num_rows(); j++) {
            if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.row_degrees[j])){
                S.data.push_back(vec<index>());
                
                variable_positions.push_back(std::make_pair(i, j));
                for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            } 
        }
    }
    
    index row_op_threshold = S_index;
    assert( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }

    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    if(row_indices_B.size() != 0){
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.get_num_cols(), B.get_num_rows()));
                    } else {
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));        
                    }
                }
                S_index++;
            }
        }
    }


    S.compute_num_cols();

    if(info){
        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(Degree_traits<D>::greater_equal(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        system_stats(timer, "Naive", S, equation_counter);
    }

    auto K = S.kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();
    K.column_reduction_triangular(true);
    if(info){
        timer.stop();
        std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        // std::cout << "Dimension of hom-space before reduction: " << K.get_num_cols() << std::endl;
        timer.start();
    }
    if(reduce){
        SparseMatrix<index> N_bar = SparseMatrix<index>(0,K.get_num_cols());
        for(index i = 0; i < A.get_num_rows(); i++){
            for(index j = 0; j < B.get_num_cols(); j++){
                if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.col_degrees[j])){
                    // Add a new homotopy
                    vec<index> h = index_pair_to_position(i, variable_positions_separator[i], B.data[j], variable_positions);
                    N_bar.data.push_back(h);
                }
            }
        }
        N_bar.compute_num_cols();
        N_bar.reduce_fully_alt(K);
        if(info){
            timer.stop();
            std::cout << "  Time to reduce to basis: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        }
    }
    return std::make_pair(K, variable_positions);
}

/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(C, B), where Q is a map on the generators. 
 * C, B are both blocks in the large matrix A. Used in AIDA.
*/
template <typename D, typename index, typename DERIVED>
std::pair< SparseMatrix<index>, vec<std::pair<index, index> > > block_hom_space_without_optimisation(const GradedSparseMatrix<D, index, DERIVED>& A, const GradedSparseMatrix<D, index, DERIVED>& C, const GradedSparseMatrix<D, index, DERIVED>& B,
        vec<index>& C_rows, vec<index>& B_rows, bool system_size = false)  { 
    vec<std::pair<index, index>> row_ops; // we store the matrices Q_i which form the basis of hom(C, B) as vectors
    // This translates from entries of the vector to entries of the matrix.
    SparseMatrix K(0,0);
    SparseMatrix S(0,0);
    S.data.reserve( C_rows.size() + B_rows.size() + 1);
    index S_index = 0;
    // First add all row-operations from C to B
    for(index i = 0; i < C_rows.size(); i++){
        for(index j = 0; j < B_rows.size(); j++){
            auto source_row_index = C_rows[i];
            auto target_row_index = B_rows[j];
            if(A.is_admissible_row_operation(source_row_index, target_row_index)){
                row_ops.push_back({source_row_index, target_row_index});
                S.data.push_back(vec<index>());
                for(auto rit = C._rows[i].rbegin(); rit != C._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(A.linearise_position_reverse(column_index, target_row_index));
                }
                S_index++;
            }
        }
    }

    index row_op_threshold = S_index;
    assert( row_ops.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return {SparseMatrix<index>(), row_ops};
    }

    // Then all column-operations from B to C
    for(index i = 0; i < B.columns.size(); i++){
        for(index j = 0; j < C.columns.size(); j++){
            if(A.is_admissible_column_operation(B.columns[i], C.columns[j])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[i]){
                    S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[j], row_index));
                }
                S_index++;
            }
        }
    }

    S.compute_num_cols();

    if(system_size){
        std::cout << "System size: " << S.get_num_cols() << std::endl;
    }

    // If M, N present the modules, then the following computes Hom(M,N), i.e. pairs of matrices st. QM = NP.
    K = S.kernel();
    // To see how much the following reduces K: index K_size = K.data.size();
    // Now we need to delete the entries of K which correspond to the row-operations.
    K.cull_columns(row_op_threshold, false);
    
    // Last we need to quotient out those Q where for every i the column Q_i - with its degree alpha_i -
    // lies in the image of N, that is, it lies in the image of N|alpha_i.
    // That is equivalent to locally reducing every column of Q.
    
    for(index i = 0; i < C_rows.size(); i++){
        D alpha = C.row_degrees[i];
        vec<index> local_admissible_columns;
        auto B_alpha = B.map_at_degree(alpha, local_admissible_columns);
        std::unordered_map<index, index> shiftIndicesMap;
        for(index j : B_rows){
            shiftIndicesMap[j] = A.linearise_position_reverse(C_rows[i], j);
        }
        B_alpha.transform_data(shiftIndicesMap);
        B_alpha.reduce_fully(K);
    }
    
    //  delete possible linear dependencies.
    K.column_reduction_triangular(true);
    return std::make_pair(K, row_ops);
}

/**
 * @brief Input, together with the hom space, is a set of indices A and a matrix B = coker_B.
 *  Computes for every matrix Q in the hom space
 * B*Q*A and reduce the whole space generated by these.
 * Outputs the indices which span the quotient space.
 * 
 * @tparam D 
 * @tparam index 
 * @param full_space 
 * @param coKer_B 
 * @param basislift_C 
 * @return Hom_space_temp<index> 
 */
template <typename index>
vec<index> hom_quotient( const Hom_space_temp<index>& full_space, 
    SparseMatrix<index>& coKer_B, const vec<index>& basislift_C, vec<index>& C_admissible_rows, vec<index>& B_admissible_rows, vec<index>& C_rows){

    // For each map in full_space, we need to compute the induced map at alpha.
    // It is given by first restricting the domain to the rows in basislift_C, 
    // then composing from the left with the coKer_B map, 

    // Because the basislift is given as a subset of the admissible rows, 
    // we need to re-index the map Q wth the help of the admissible_rows.

    vec<SparseMatrix<index>> induced_maps = vec<SparseMatrix<index>>();

    auto& B_row_index_map = shiftIndicesMap(B_admissible_rows);

        for(index i = 0; i < full_space.first.num_cols; i++){
            const vec<index>& current_map = full_space.first.data[i];
            auto& index_pairs = full_space.second; // This is a pair (i,j) where i is the column index, j, the row index, and it is sorted first by i, then by j.
            SparseMatrix<index> Q_basislift_C = SparseMatrix<index>(0,0);
            index k = 0;
            for(index j : basislift_C){
                Q_basislift_C.data.push_back(vec<index>());

                if(k != current_map.size()){
     
                    while( C_rows[ index_pairs[current_map[k]].first ] < C_admissible_rows[j] ){
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }

                    while( C_rows[ index_pairs[current_map[k]].first ] == C_admissible_rows[j]){
                        Q_basislift_C.data.back().push_back( B_row_index_map[ index_pairs[current_map[k]].second ]);
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }
                }
            }
            Q_basislift_C.compute_num_cols();
            // Now we have the restriction to the basislift_C, we need to compose with the coKer_B map.
            //TO-DO: Compute the rows of coKer_B beforehand and then multiply to avoid transposition.
            induced_maps.emplace_back(multiply_transpose(coKer_B, Q_basislift_C));
        }

        // Reduce the vector space of induced homomorphisms
        return general_reduction<index, SparseMatrix<index>> (induced_maps);
}

/**
 * @brief Just as hom_quotient, but returns only "true" if the quotient space is zero.
 * 
 * @tparam index 
 * @param full_space 
 * @param coKer_B 
 * @param basislift_C 
 * @param C_admissible_rows 
 * @param B_admissible_rows 
 * @return true 
 * @return false 
 */
template <typename index>
bool hom_quotient_zero( const Hom_space_temp<index>& full_space, 
    SparseMatrix<index>& coKer_B, const vec<index>& basislift_C, vec<index>& C_admissible_rows, vec<index>& B_admissible_rows, vec<index>& C_rows){

    // For each map in full_space, we need to compute the induced map at alpha.
    // It is given by first restricting the domain to the rows in basislift_C, 
    // then composing from the left with the coKer_B map, 

    // Because the basislift is given as a subset of the admissible rows, 
    // we need to re-index the map Q wth the help of the admissible_rows.

    auto B_row_index_map = shiftIndicesMap(B_admissible_rows);

        for(index i = 0; i < full_space.first.get_num_cols(); i++){
            const vec<index>& current_map = full_space.first.data[i];
            auto& index_pairs = full_space.second; // This is a pair (i,j) where i is the column index, j, the row index, and it is sorted first by i, then by j.
            SparseMatrix<index> Q_basislift_C = SparseMatrix<index>(0,0);
            index k = 0;
            for(index j : basislift_C){
                Q_basislift_C.data.push_back(vec<index>());

                if(k < current_map.size()){
     
                    while( C_rows[ index_pairs[current_map[k]].first ] < C_admissible_rows[j] ){
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }
                    if(k == current_map.size()){
                        break;
                    }
                    
                    while( C_rows[ index_pairs[current_map[k]].first ] == C_admissible_rows[j]){
                        Q_basislift_C.data.back().push_back( B_row_index_map[ index_pairs[current_map[k]].second ]);
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }
                }
            }
            // How many rows does this matrix have?
            Q_basislift_C.compute_num_cols();
            // Now we have the restriction to the basislift_C, we need to compose with the coKer_B map.
            //TO-DO: Compute the rows of coKer_B beforehand and then multiply to avoid transposition.
            SparseMatrix<index> Q_alpha = multiply_transpose(coKer_B, Q_basislift_C);
            if ( Q_alpha.is_nonzero() ) {
                return false;
            }
        }

        // Reduce the vector space of induced homomorphisms
        return true;
}

/**
 * @brief 
 * 
 * @param A 
 * @param B 
 * @param alpha 
 * @return Hom_space_temp<index> 
 */
template <typename D, typename index, typename DERIVED>
Hom_space_temp<index> hom_alpha(const GradedSparseMatrix<D, index, DERIVED>& A, const GradedSparseMatrix<D, index, DERIVED>& B, Hom_space_temp<index>& full_hom_space, const D alpha) {
    
    Hom_space_temp<index> result;
    auto [B_alpha, B_alpha_gens] = B.map_at_degree_pair(alpha);
    auto [A_alpha, A_alpha_gens] = A.map_at_degree_pair(alpha);

    vec<index> B_alpha_basis = B_alpha.coKernel_basis(B_alpha_gens);
    vec<index> A_alpha_basis = A_alpha.coKernel_basis(A_alpha_gens);

    // What should the indexing for this be? TO-DO: Check this.
    SparseMatrix<index> coker_B_alpha = B_alpha.coKernel_without_prelim(B_alpha_basis, B_alpha_gens);

    //TO-DO: Finish this.
}   


} // namespace graded_linalg


#endif // Homomorphisms.hpp


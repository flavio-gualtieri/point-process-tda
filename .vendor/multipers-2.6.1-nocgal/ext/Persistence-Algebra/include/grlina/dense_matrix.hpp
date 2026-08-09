/**
 * @file dense_matrix.hpp
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

#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include "grlina/matrix_base.hpp"
#include "grlina/bitset_algebra.hpp"


namespace graded_linalg {


struct DenseMatrix : public MatrixUtil<bitset, int, DenseMatrix>{
        
    bool rowReduced = false;
    bool completeRowReduced = false;
    boost::dynamic_bitset<> pivot_vector;


    /**
     * @brief Deletes the last i rows of the matrix.
     * 
     * @param i 
     */
    void cull_columns(int i){
        this->num_rows = this->num_rows - i;
        for (auto& column : this->data){
            column.resize(this->num_rows);
        }
    };

    DenseMatrix() : MatrixUtil<bitset, int, DenseMatrix>() {}

    DenseMatrix(int m, int n) : MatrixUtil<bitset, int, DenseMatrix>(m, n) {
        data = vec<bitset>(m, bitset(n, 0));
    }

    DenseMatrix(const vec<bitset>& data) : MatrixUtil<bitset, int, DenseMatrix>(data.size(), data[0].size(), data) {}

    DenseMatrix(const DenseMatrix& other) : MatrixUtil<bitset, int, DenseMatrix>(other)  {}

    DenseMatrix(int m, const std::string& type) : MatrixUtil<bitset, int, DenseMatrix>(m, m) {
        if(type == "Identity") {
            for(int i = 0; i < m; i++) {
                this->data.emplace_back(bitset(m, 0).set(i));
            }
        } else {
            throw std::invalid_argument("Unknown matrix type: " + type);
        }
    }

    /**
     * @brief Construct a new Dense Matrix object from an ifstream if it has the bitsets as 1-0-strings.
     * 
     * @param file 
     */
    DenseMatrix(std::ifstream& file) : MatrixUtil<bitset, int, DenseMatrix>(){
        // deserialize matrix from a file
        num_cols = 0;
        num_rows = 0;
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(int));
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(int));
        data.reserve(num_cols);
        for (int i = 0; i < num_cols; ++i) {
            unsigned long bs_data;
            file.read(reinterpret_cast<char*>(&bs_data), sizeof(bs_data));
            data.emplace_back(bitset(num_rows, bs_data));
        }
    }

    /**
     * @brief Construct a new Dense Matrix consisting of only elementary basis vectors. 
     * 
     * @param i_pivots Marks the elementary basis vectors that are to be included in the matrix.
     */
    DenseMatrix(const boost::dynamic_bitset<>& i_pivots) : 
        MatrixUtil<bitset, int, DenseMatrix>(i_pivots.count(), i_pivots.size(), std::vector<bitset>(i_pivots.count(), bitset(i_pivots.size(), 0)) ),
        rowReduced(true),       // By construction, the matrix is in reduced form
        pivot_vector(i_pivots){ 
        // Populate the matrix to form an identity matrix on the pivot positions
        int pivotint = 0; // To keep track of which pivot we're on
        for (int r = 0; r < num_cols; ++r) {
            // Find the next set bit in i_pivots, which will be our next pivot
            while (!i_pivots[pivotint]) {
                ++pivotint; // Skip over unset bits
            }

            // Set the pivot position to 1
            data[r][pivotint] = 1;
            ++pivotint; // Move to the next pivot for the next row
        }
    }

    /**
     * @brief Construct a Dense Matrix in reduced form from a set of pivots_ and a number. 
     * The binary representation of this number fills the spots in the reduced matrix which are not necessarily zero.
     * 
     * @param pivots_ 
     * @param positions 
     */
    DenseMatrix(const bitset& pivots_, std::vector<std::pair<int,int>> &positions, int filler) : DenseMatrix(pivots_) {
        size_t mul = positions.size();
        for (int j=0; j < mul; j++){
            if (filler & (1 << j)) {
                data[positions[j].first][positions[j].second] = 1;
            }
        }
    }

    

    /**
     * @brief Add this bitset to the i-th column.
     * 
     * @param i 
     * @param bitset 
     */
    void addOutsideRow(int i, const boost::dynamic_bitset<>& bitset) {
        data[i] ^= bitset; // XOR the i-th row with the bitset
    }

    /**
     * @brief This is a second algorithm to reduce column-wise. It finds a lower triangular reduced form of the matrix involving swaps.
     * 
     * @param complete 
     */
    void colReduce(bool complete = false) {
        int lead = 0;
        pivot_vector = boost::dynamic_bitset<>(num_rows, 0);
        for (int r = 0; r < num_cols; ++r) {
            if (lead >= num_rows) {
                break; // No more columns to work on, exit the loop
            }

            int i = r;
            // Find the first non-zero entry in the row(potential pivot)
            while (i < num_cols && !data[i][lead]) {
                ++i;
            }

            if (i < num_cols) {
                // Found a non-zero entry, so this row does have a pivot
                // If the pivot is not in the current col, swap the cols
                if (i != r) {
                    swap_cols(i, r);
                }
                pivot_vector[lead] = true; // Mark this row as having a pivot after confirming pivot

                // Eliminate all non-zero entries below this pivot
                for (int j = r + 1; j < num_cols; ++j) {
                    if (data[j][lead]) {
                        data[j] ^= data[r];
                    }
                }

                if (complete) {
                    // Eliminate all non-zero entries above this pivot
                    for (int j = 0; j < r; ++j) {
                        if (data[j][lead]) {
                            data[j] ^= data[r];
                        }
                    }
                }

                ++lead; // Move to the next row
            } else {
                // No pivot in this row, so we move to the next row without incrementing the col int
                ++lead;
                --r; // Stay on the same col for the next iteration
            }
        }
        rowReduced = true; 
        if(complete){completeRowReduced = true;}
    }

    /**
     * @brief Writes the matrix to an ofstream by converting it to a string.
     * 
     * @param file 
     */
    void serialize(std::ofstream& file) const {
        // Write the size of the vector
        assert(data.size() == num_cols);
        file.write(reinterpret_cast<const char*>(&num_cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_rows), sizeof(int));
        // Serialize each dynamic_bitset in the vector

        for (const auto& bs : data) {
            const auto& bs_data = bs.to_ulong();
            file.write(reinterpret_cast<const char*>(&bs_data), sizeof(bs_data));
        }
    }

    /**
     * @brief Computes this*other. Maybe a bit slow.
     * 
     * @param other 
     */
    DenseMatrix multiply_right(const DenseMatrix& other) const  {
        assert(this->get_num_cols() == other.get_num_rows());
        DenseMatrix result(other.get_num_cols(), this->get_num_rows());

        for (int i = 0; i < other.get_num_cols(); ++i) {
            for (int j = 0; j < this->get_num_rows(); ++j) {
                for (int k = 0; k < this->get_num_cols(); ++k) {
                    if (this->data[k][j] && other.data[i][k]) {
                        result.data[i].flip(j);
                    }
                }
            }
        }

        return result;
    }

    /**
     * @brief Returns a transpose of the matrix.
     * 
     * @return DenseMatrix 
     */
    DenseMatrix transposed_copy() {
        DenseMatrix result(this->get_num_rows(), this->get_num_cols());
        for (int i = 0; i < this->get_num_cols(); i++){
            for (int j = 0; j < this->get_num_rows(); j++){
                result.data[j][i] = this->data[i][j];
            }
        }
        return result;
    }

    DenseMatrix divide_left(DenseMatrix& other) const {
        //TO-DO: Implement row-reduction for dense-matrices and use it here instead of applying column-reduction to get an inverse
        DenseMatrix inverse = other.inverse_nocopy();
        DenseMatrix result = inverse.multiply_right(*this); // Unnecessary copy *and* multiplication
        return result;
    }

}; // end of DenseMatrix


inline std::vector<DenseMatrix > pivotsToEchelon(const boost::dynamic_bitset<> &pivots, std::vector<std::pair<int,int>> &positions ){
    std::vector<DenseMatrix> reducedMatrices;
    
    size_t n = pivots.size();
    size_t mul = positions.size();
    size_t subsetCount = static_cast<size_t>(std::pow(2, mul));

    for (size_t i = 0; i < subsetCount; ++i) {
        reducedMatrices.emplace_back( DenseMatrix(pivots, positions, i) );
    }
    return reducedMatrices;
}

inline void pivotsToEchelon(std::vector<DenseMatrix>& result, const boost::dynamic_bitset<> &pivots, std::vector<std::pair<int,int>> &positions ){
    
    size_t n = pivots.size();
    size_t mul = positions.size();
    size_t subsetCount = static_cast<size_t>(std::pow(2, mul));

    for (size_t i = 0; i < subsetCount; ++i) {
        result.emplace_back( DenseMatrix(pivots, positions, i) );
    }
}




// For a given bitset returns all non-set positions in a col-echelon matrix whose pivots are given by the input.
inline std::vector<std::pair<int, int>> getEchelonPositions(const boost::dynamic_bitset<> &bitset) {
    size_t countOnes = 0;
    std::vector<std::pair<int, int>> positions;

    for (boost::dynamic_bitset<>::size_type i = 0; i < bitset.size(); ++i) {
        if (!bitset.test(i)) {
            for (int j = 0; j < countOnes; j++){
                positions.push_back(std::make_pair(j, i));
            }
        } else {
            countOnes++;
        }
    }
    return positions;
}

inline std::vector<DenseMatrix> all_proper_subspaces(int k){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 1; i < (1ULL << k) - 1; ++i) {
        pivots.emplace_back(bitset(k, i));
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

inline std::vector<DenseMatrix> all_subspaces(int k){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 0; i < (1ULL << k); ++i) {
        pivots.emplace_back(bitset(k, i));
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

inline std::vector<DenseMatrix> grassmannian(int k, int n){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 0; i < (1ULL << k); ++i) {
        bitset b(k, i);
        if(b.count() == n){
            pivots.emplace_back(b);
        }
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

inline std::vector<DenseMatrix> grassmannian_union(int k, int n){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 0; i < (1ULL << k); ++i) {
        bitset b(k, i);
        if(b.count() <= n){
            pivots.emplace_back(b);
        }
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

// A pair of DenseMatrices - the two subspaces of a decomposition.
using VecDecomp = std::pair<DenseMatrix, DenseMatrix>;

// Associates to each integer the subspace whos Plücker coordinates have the integer as a binary representation. 
// Then to each subspace there can be a list of complements.
using DecompBranch = std::vector< std::vector<VecDecomp> >;

// associates to each bitset the subspaces whose reduced form has the entries of the bitset as pivots.
using DecompTree = std::unordered_map<boost::dynamic_bitset<>, DecompBranch , BitsetHash>;

// A transition is an invertible matrix with 1s on the diagonal and a subset of the columns given as a bitset.
// The matrix stores the necessary column-operations to transform a decomposition into another one. 
// The nonzero entries of the bitset give the column-indices associated to the first subspace, 
// the zero entries to the second subspace.
using transition = std::pair<DenseMatrix, bitset>;

/**
 * @brief Writes the DecompTree to a file.
 * 
 * @param tree 
 * @param filename 
 */
inline void saveDecompTree(const DecompTree& tree, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    } else {
        std::cout << "File opened successfully for writing: " << filename << std::endl;
    }

    
    size_t treeSize = tree.size();
    file.write(reinterpret_cast<const char*>(&treeSize), sizeof(treeSize));

    for ( auto& [key, branch] : tree) {

        serializeDynamicBitset(key, file);
        // key gives pivots
        
        size_t branchSize = branch.size();
        file.write(reinterpret_cast<const char*>(&branchSize), sizeof(branchSize));

        for (size_t p_coord = 0; p_coord< branch.size(); p_coord++) {
            
            // p_coord is the integer representation of the Plücker coordinate

            size_t vectorSize = branch[p_coord].size();
            file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

            for (const VecDecomp& vecs : branch[p_coord]) {
                // Careful, we are serializing the second file first, because the reader will reverse this again!
                vecs.second.serialize(file);

                vecs.first.serialize(file);
            }
        }
    }

    file.close();
}

/**
 * @brief Loads a DecompTree from a file.
 * 
 * @param filename 
 * @return DecompTree 
 */
inline DecompTree loadDecompTree(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file '" + filename + "' for reading.");
    } else {
        std::cout << "File opened successfully for reading: " << filename << std::endl;
    }
  
    size_t treeSize;
    file.read(reinterpret_cast<char*>(&treeSize), sizeof(treeSize));

    DecompTree tree;
    for (size_t i = 0; i < treeSize; ++i) {
        boost::dynamic_bitset<> key = deserializeDynamicBitset(file);

        size_t branchSize;
        file.read(reinterpret_cast<char*>(&branchSize), sizeof(branchSize));

        DecompBranch branch;
        for (size_t j = 0; j < branchSize; ++j) {

            size_t vectorSize;
            file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));

            std::vector<VecDecomp> VecDecomps;
            VecDecomps.reserve(vectorSize);
            for (size_t k = 0; k < vectorSize; ++k) {
                //This reverses the order of the pair. Works because the writer has already reversed it, too.
                VecDecomps.emplace_back(DenseMatrix(file), DenseMatrix(file));
            }

            branch.emplace_back(std::move(VecDecomps));
        }

        tree.emplace(std::move(key), std::move(branch));
    }

    file.close();
    return tree;
}

/**
 * @brief Prints the whole content of the DecompTree to the console.
 * 
 * @param tree 
 */
inline void print_tree(DecompTree& tree){
    int num_bits = 1;
    std::cout << "Printing tree with the following branches: " << std::endl;
    for(auto& [pivots_, branch] : tree){
        std::cout << pivots_ << ", ";
    }
    std::cout << std::endl;
    for(auto& [pivots_, branch] : tree){
        std::cout << "Pivots: ";
        print_bitset(pivots_);
        for(int i = 0; i < branch.size(); i++){
            num_bits = static_cast<int>(std::log2(branch.size()));
            bitset binary(num_bits, i);
            std::cout << "Pluecker Coordinate as integer is " << i << ". And as binary is ";
            print_bitset(binary);
            for(auto& [first, second] : branch[i]){
                std::cout << "First: " << std::endl;
                first.print(true);
                std::cout << "Second: " << std::endl;
                second.print(true);
            }
        }
    }
}

/**
 * @brief Saves a transition list to a file.
 * 
 * @param transitions 
 * @param filename 
 */
inline void save_transition_list(const std::vector<transition>& transitions, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    } else {
        std::cout << "Writing transitions to: " << filename << std::endl;
    }

    // Write the size of the vector
    size_t vectorSize = transitions.size();
    file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

    for (const auto& [matrix, bitset] : transitions) {
        matrix.serialize(file);
        serializeDynamicBitset(bitset, file);
    }

    file.close();
}

/**
 * @brief Loads a transition list from a file.
 * 
 * @param filename 
 * @return std::vector<transition> 
 */
inline std::vector<transition> load_transition_list(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename + " for reading.");
    } else {

    }

    size_t vectorSize;
    file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));
    // std::cout << "loading #matrices: " << vectorSize << std::endl;
    std::vector<transition> transitions;
    transitions.reserve(vectorSize);
    for (size_t i = 0; i < vectorSize; ++i) {
        auto T = DenseMatrix(file);
        auto bs = deserializeDynamicBitset(file);
        transitions.emplace_back(std::move(T), std::move(bs));
    }

    file.close();
    return transitions;
}



} // namespace graded_linalg

#endif // DENSE_MATRIX_HPP
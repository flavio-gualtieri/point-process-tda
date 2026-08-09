#pragma once

#ifndef BITSET_ALGEBRA_HPP
#define BITSET_ALGEBRA_HPP

#include "grlina/matrix_base.hpp"
#include <fstream>

namespace graded_linalg {

struct BitsetHash {
    unsigned long operator()(const boost::dynamic_bitset<>& bs) const {
        // Safe to use to_ulong() if the bitset size is guaranteed to be <= 32
        return bs.to_ulong();
    }
};

/**
 * @brief asks if a < b in order of the entries (reverse of standard comparison)
 */
inline bool compareBitsets(const boost::dynamic_bitset<>& a, const boost::dynamic_bitset<>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Bitsets must be of the same size for comparison.");
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.test(i) != b.test(i)) {
            return a.test(i) < b.test(i);
        }
    }

    return false; // The bitsets are equal
}

inline std::ostream& operator<< (std::ostream& ostr, const boost::dynamic_bitset<>& bs) {
    for (int i = 0; i < bs.size(); i++){
      ostr << bs[i] << " ";
    }
    return ostr;
}


/**
 * @brief prints the bitset from first to last entry
 * 
 * @param bs 
 */
inline void print_bitset(const boost::dynamic_bitset<>& bs) {
    std::cout << bs << std::endl;
}

/**
 * @brief prints the bitset from last to first entry
 * 
 * @param bs 
 */
inline void print_bitset_reverse(const boost::dynamic_bitset<>& bs) {
    for (int i = bs.size() - 1; i >= 0; --i) {
        std::cout << bs[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Converts a bitset to a string representation in reverse order
 * 
 * @param bs 
 * @return std::string 
 */
inline std::string bitsetToString_alt(const boost::dynamic_bitset<>& bs) {
    std::string result;
    result.reserve(bs.size());
    for (int i = bs.size() - 1; i >= 0; --i) {
        result.push_back(bs.test(i) ? '1' : '0');
    }
    return result;
}

/**
 * @brief Converts a bitset to a string representation in forward order
 * 
 * @param bs 
 * @return std::string 
 */
inline std::string bitsetToString(const boost::dynamic_bitset<>& bs) {
    std::string result;
    result.reserve(bs.size());
    for (size_t i = 0; i < bs.size(); ++i) {
        result.push_back(bs.test(i) ? '1' : '0');
    }
    return result;
}

/**
 * @brief Writes a dynamic_bitset to a file in reverse order
 * 
 * @param bs 
 * @param file 
 */
inline void serializeDynamicBitset(const boost::dynamic_bitset<>& bs, std::ofstream& file) {
    int length = bs.size();
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    const auto& bs_data = bs.to_ulong();
    file.write(reinterpret_cast<const char*>(&bs_data), sizeof(bs_data));
}


/**
 * @brief Reads a dynamic_bitset from a file in reverse order
 * 
 * @param file 
 * @return boost::dynamic_bitset<> 
 */
inline boost::dynamic_bitset<> deserializeDynamicBitset(std::ifstream& file) {
    int length;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    unsigned long bs_data;
    file.read(reinterpret_cast<char*>(&bs_data), sizeof(bs_data));
    return bitset(length, bs_data);
}


inline vec<bitset> compute_standard_vectors(int k){
    vec<bitset> result;
    for (int i = 0; i < k; i++){
        result.emplace_back(bitset(k, 0).set(i));
    }
    return result;
}

/**
 * @brief returns { 1, 11, 111, ... }
 * 
 * @param k 
 * @return 
 */
inline vec<boost::dynamic_bitset<>> compute_sum_of_standard_vectors(int k){
    auto result = vec<boost::dynamic_bitset<>>();
    for (int i = 0; i < k; i++){
        boost::dynamic_bitset<> bitset(i + 1);
        bitset.set(); // Set all bits to 1
        result.push_back(bitset);
    }
    return result;
}

/**
 * @brief Returns a copy of a with the 1-entries replaced by b.
 * 
 * @param a 
 * @param b 
 * @return bitset 
 */
inline bitset glue(const bitset& a, const bitset& b){
    bitset result = a;
    assert(a.count() == b.size());
    size_t counter = 0;
    for(auto it = result.find_first(); it != bitset::npos; it = result.find_next(it)){
        if(!b[counter]){
            result.reset(it);
        }
        counter++;
    }
    return result;
}

/**
 * @brief Copies b onto the 1-entries of a.
 * 
 * @param a 
 * @param b 
 */
inline void glue_to(bitset& a, const bitset& b){
    assert(a.count() == b.size());
    size_t counter = 0;
    for(auto it = a.find_first(); it != bitset::npos; it = a.find_next(it)){
        if(!b[counter]){
            a.reset(it);
        }
        counter++;
    }
}



inline void generateCombinations(boost::dynamic_bitset<> &bitset, int offset, int k, std::vector<boost::dynamic_bitset<>> &combinations) {
    if (k == 0) {
        combinations.push_back(bitset);
        return;
    }

    for (int i = offset ; i < bitset.size(); i++) {
        bitset.set(i);
        generateCombinations(bitset, i + 1, k - 1, combinations);
        bitset.reset(i);
    }
}

/**
 * @brief Recursively generates all bitsets of length n with k bits set to 1.
 * 
 * @param n 
 * @param k 
 * @return std::vector<boost::dynamic_bitset<>> 
 */
inline std::vector<boost::dynamic_bitset<>> generateAllBitsetsWithKOnes(int n, int k) {
    std::vector<boost::dynamic_bitset<>> combinations;
    boost::dynamic_bitset<> bitset(n, 0);
    generateCombinations(bitset,  0, k, combinations);
    return combinations;
}

/**
 * @brief Generates all bitsets of length n with n/2 bits set to 1, where the first bit is set to 1.
 * 
 * @param n 
 * @return std::vector<boost::dynamic_bitset<>> 
 */
inline std::vector<boost::dynamic_bitset<>> generateHalfBitsets(int n) {
    // Check if n is even
    if (n % 2 != 0 || n <= 0) {
        throw std::invalid_argument("n must be a positive even number.");
    }

    // Generate all bitsets of length n-1 with n/2 - 1 bits set
    std::vector<boost::dynamic_bitset<>> bitsets = generateAllBitsetsWithKOnes(n - 1, n / 2 - 1);

    // Prepend a '1' to each bitset
    for (auto& bitset : bitsets) {
        bitset.resize(n);
        bitset <<= 1;
        bitset[0]=1;
    }

    return bitsets;
}

inline std::vector<boost::dynamic_bitset<>> generateBitsets(int n) {
    std::vector<boost::dynamic_bitset<>> bitsets;

    // Start k from the largest possible value less than n/2 (if n is even) or less than n/2 rounded down (if n is odd)
    int startK = (n % 2 == 0) ? (n / 2 - 1) : (n / 2);

    // Generate and append bitsets with fewer ones
    for (int k = startK; k > 0; --k) {
        std::vector<boost::dynamic_bitset<>> additionalBitsets = generateAllBitsetsWithKOnes(n, k);
        bitsets.insert(bitsets.end(), additionalBitsets.begin(), additionalBitsets.end());
    }

    return bitsets;
}

} // graded_linalg


#endif // BITSET_ALGEBRA_HPP
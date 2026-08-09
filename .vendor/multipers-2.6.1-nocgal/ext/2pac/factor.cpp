#include "factor.hpp"
#include "block_column_matrix.hpp"
#include "matrices.hpp"
#include "time_measurement.hpp"
#include "typedefs.hpp"

template<class T, bool indirect>
inline SparseMatrix factor_matrix_inner2(T &matrix, T &through, pivot_map_t &p){
    Timing::time t("Factorize");
    SparseMatrix result = SparseMatrix(through.columns(), matrix.columns());
    #pragma omp parallel for
    for(uint j = 0; j < matrix.columns(); j++){
        std::vector<uint> factored;
        for(;;){
            index_t i = matrix.pivot(j);
            if(i == no_pivot)
                break;
            assert(p[i] != no_pivot);
            matrix.column_operation(through, p[i], j);
            factored.push_back(p[i]);
            assert(matrix.pivot(j) < i);
        }
        result[j] = ColumnType(through.columns(), std::move(factored), ColumnType::unsorted);
    }
    return result;
}

#ifdef DISABLED
#ifdef ARRAY_MATRICES
template<>
inline SparseMatrix factor_matrix_inner2<SparseMatrix, true>(SparseMatrix &matrix, SparseMatrix &through, pivot_map_t &p){
    SparseMatrix result(through.columns(), matrix.columns());
    IndirectColumn c;
    #pragma omp parallel for private(c)
    for(uint j = 0; j < matrix.columns(); j++){
        std::vector<uint> factored;
        c = matrix.data[j];
        for(;;){
            index_t i = c.pivot();
            if(i == no_pivot)
                break;
            assert(p[i] != no_pivot);
            c += through[p[i]];
            factored.push_back(p[i]);
            assert(c.pivot() < i);
        }
        result[j] = ColumnType(through.columns(), std::move(factored), ColumnType::unsorted);
    }
    return result;
}
#endif
#endif

template<class T, bool indirect>
SparseMatrix factor_matrix_inner(T &matrix, T &through){
    assert(matrix.rows() == through.rows());
    // Determine assignment p: row i ↦ column with pivot i.
    pivot_map_t p(matrix.rows(), -1);
    for(uint j = 0; j < through.columns(); j++){
        index_t i = through.pivot(j);
        if(i != no_pivot){
            assert(p.get(i) == no_pivot);
            p.set(i, j);
        }
    }
    return factor_matrix_inner2<T, indirect>(matrix, through, p);
}

template<class T>
SparseMatrix (*factor_matrix_inner_p)(T &matrix, T &through) = factor_matrix_inner<T, false>;

template<class T>
SparseMatrix factor_matrix(T &matrix, T &through){
    return factor_matrix_inner_p<T>(matrix, through);
}
template SparseMatrix factor_matrix(SparseMatrix &matrix, SparseMatrix &through);
template SparseMatrix factor_matrix(BlockColumnMatrix &matrix, BlockColumnMatrix &through);

void set_factorization_indirection_level(uint level){
    #ifdef ARRAY_MATRICES
    if(level > 0)
        factor_matrix_inner_p<SparseMatrix> = factor_matrix_inner<SparseMatrix, true>;
    else
        factor_matrix_inner_p<SparseMatrix> = factor_matrix_inner<SparseMatrix, false>;
    #endif
}

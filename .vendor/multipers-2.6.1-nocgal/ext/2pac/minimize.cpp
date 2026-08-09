#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>
#include "indirect.hpp"
#include "matrices.hpp"
#include "minimize.hpp"
#include "block_column_matrix.hpp"
#include "time_measurement.hpp"
#include <omp.h>

template<class T>
std::tuple<std::vector<size_t>, std::vector<size_t>, GradedMatrix> minimize(GenericGradedMatrix<T> &matrix){
    Timing::time t("Minimize");
    pivot_map_t p(matrix.rows(), no_pivot);         // pivot row -> column assignment
    std::vector<size_t> nlr, nlc;                            // non-local rows and columns.
    std::vector<grade> nlr_grades, nlc_grades;               // row and column grades of minimized matrix.

    // Identify local columns: reduce columns as far as necessary to discern if they are local or not.
    Timing::start("Find local pairs");
    for(uint j = 0; j < matrix.columns(); ++j){
        for(;;){
            index_t i = matrix.data.pivot(j);
            if(i == no_pivot){
                nlc.push_back(j);
                break;
            }
            if(matrix.row_grades[i] != matrix.column_grades[j]){
                nlc.push_back(j);
                break;
            }
            if(p.get(i) == no_pivot){
                p.set(i, j);
                break;
            }
            matrix.data.column_operation(p[i], j);
            assert(matrix.data.pivot(j) < i);
        }
    }
    Timing::stop();


    // Determine non-local rows and new row-indices.
    std::vector<index_t> new_row_indices(matrix.rows(), no_pivot);
    for(uint i = 0; i < matrix.rows(); i++){
        if(p.get(i) == no_pivot){
            new_row_indices[i] = nlr.size();
            nlr.push_back(i);
        }
    }
    
    Timing::info() << "Input size: " << matrix.rows() << "x" << matrix.columns() << "; "
                   << "local pairs: " << matrix.columns() - nlc.size() << "; "
                   << "Output size: " << nlr.size() << "x" << nlc.size() << Timing::endl;

    // Eliminate entries in local rows from non-local columns.
    Timing::start("Eliminate local entries.");
    std::vector<std::vector<uint>> minimized_entries(nlc.size());
    #pragma omp parallel for schedule(guided, 1)
    for(uint k = 0; k < nlc.size(); ++k){
        for(;;){
            index_t i = matrix.data.pivot(nlc[k]);
            if(i == no_pivot)
                break;
            else if(p.get(i) == no_pivot){
                matrix.data.pop_pivot(nlc[k]);
                minimized_entries[k].push_back(new_row_indices[i]);
            }
            else
                matrix.data.column_operation(p[i], nlc[k]);
            assert(matrix.data.pivot(nlc[k]) < i);
        }
    }
    Timing::stop();
    GradedMatrix result(
        get_elements(matrix.row_grades, nlr), 
        get_elements(matrix.column_grades, nlc), 
        SparseMatrix(nlr.size(), std::move(minimized_entries), ColumnType::descending)
    );
    return {std::move(nlr), std::move(nlc), std::move(result)};
}

template std::tuple<std::vector<size_t>, std::vector<size_t>, GradedMatrix> minimize<>(GenericGradedMatrix<SparseMatrix> &matrix);
template std::tuple<std::vector<size_t>, std::vector<size_t>, GradedMatrix> minimize<>(GenericGradedMatrix<BlockColumnMatrix> &matrix);

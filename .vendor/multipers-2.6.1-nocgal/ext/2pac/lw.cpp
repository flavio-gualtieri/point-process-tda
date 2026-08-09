#include "lw.hpp"
#include "matrices.hpp"
#include "typedefs.hpp"
#include <algorithm>
#include <boost/heap/priority_queue.hpp>
#include <type_traits>
#include "time_measurement.hpp"

/** Computes the graded kernel of the matrix, and a minimal
 * generating system of the image and a kernel of it.
 * @returns A triple of three graded matrices:
 * - the kernel
 * - the minimal generating system of the image
 * - the kernel of the minimal generating system
 */
template<class Ordering>
std::tuple<GradedMatrix, GradedMatrix, GradedMatrix> kernel_mgs(GradedMatrix& M){
    Timing::time t("Kernel & MGS");
    assert((std::is_same_v<Ordering, grade::lex_greater> && std::is_sorted(M.column_grades.begin(), M.column_grades.end(), grade::colex_less())
        || std::is_same_v<Ordering, grade::colex_greater> && std::is_sorted(M.column_grades.begin(), M.column_grades.end(), grade::lex_less())
        ));
    // Queue of pairs (bigrade, column index) of grades and columns to consider for reduction.
    // The quee is sorted first by (lex) grade, then by column index.
    // Initially, just the matrix columns and their column grades.
    using compare = boost::heap::compare<lex_compare<grade, uint, Ordering, std::greater<uint>>>;
    boost::heap::priority_queue<std::pair<grade, uint>, compare> Q;
    for(uint j=0; j < M.column_grades.size(); j++)
        Q.push({M.column_grades[j], j});

    // Reduction matrix, from which the kernel will be a submatrix.
    SparseMatrix V = SparseMatrix::eye(M.column_grades.size());
    std::vector<grade> kernel_grades;
    std::vector<size_t> kernel_columns;

    // Assignment pivot row to column.
    pivot_map_t p(M.rows(), no_pivot);

    // The following is relevant to compute also a minimal generating system:
    // Reduction matrix and kernel column indices and grades for mgs
    SparseMatrix V_mgs(0, 0);
    std::vector<grade> mgs_column_grades, mgs_kernel_grades;
    std::vector<index_t> mgs_column_indices(M.columns(), no_pivot);
    std::vector<size_t> mgs_kernel_columns;
    std::vector<ColumnType> mgs_columns;

    grade z{0, 0};
    uint j;
    auto &m = M.data;
    assert(m.is_well_defined());
    while(Q.size()){
        std::tie(z, j) = Q.top();
        Q.pop();
        assert(m[j].is_well_defined());
        // Reduce column j; all reductions are valid at z (and not earlier).
        for(;;){
            index_t i = m.pivot(j);
            if(i == no_pivot){
                // Column has been reduced to zero; i.e., kernel generator found.
                kernel_columns.push_back(j);
                kernel_grades.push_back(z);
                break;
            }
            if(p.get(i) == no_pivot){
                // Column cannot be reduced further.
                p.set(i, j);
                break;
            }
            if(!(uint(p[i]) < j && M.column_grades[p[i]] <= z)){
                // Column can reduce another column p[i] at a larger grade.
                Q.push({M.column_grades[p[i]] | z, p[i]});
                p.set(i, j);
                break;
            }
            m.column_operation(p[i], j);
            V.column_operation(p[i], j);
        assert(m[j].is_well_defined());
            // If we compute a mgs, perform the same operation on the respective reduction matrix.
            if(mgs_column_indices[j] != no_pivot)
                V_mgs.column_operation(mgs_column_indices[p[i]], mgs_column_indices[j]);
            assert(m.pivot(j) < i);
        }
        // If we compute a mgs...
        if(m.pivot(j) != no_pivot && z == M.column_grades[j]){
            // The column represents a new minimal generator of the image.
            // Extend V_mgs by a new row and column with a single 1 on the diagonal.
            mgs_columns.push_back(m[j]);
            mgs_column_grades.push_back(z);
            assert(mgs_column_indices[j] == no_pivot);
            mgs_column_indices[j] = mgs_columns.size() - 1;
            V_mgs.v_extend(0, 1);
            V_mgs.h_extend(ColumnType(V_mgs.rows(), V_mgs.rows() - 1));
        }
        else if (m.pivot(j) == no_pivot && mgs_column_indices[j] != no_pivot) {
            mgs_kernel_columns.push_back(mgs_column_indices[j]);
            mgs_kernel_grades.push_back(z);
        }
    }
    return {
        GradedMatrix(M.column_grades, kernel_grades, std::move(V).get_columns(kernel_columns)),
        GradedMatrix(M.row_grades, mgs_column_grades, SparseMatrix(M.rows(), std::move(mgs_columns))),
        GradedMatrix(mgs_column_grades, mgs_kernel_grades, std::move(V_mgs).get_columns(mgs_kernel_columns))
    };
}
template std::tuple<GradedMatrix, GradedMatrix, GradedMatrix> kernel_mgs<grade::lex_greater>(GradedMatrix&);
template std::tuple<GradedMatrix, GradedMatrix, GradedMatrix> kernel_mgs<grade::colex_greater>(GradedMatrix&);


void set_homology_indirection_level(uint level){
}
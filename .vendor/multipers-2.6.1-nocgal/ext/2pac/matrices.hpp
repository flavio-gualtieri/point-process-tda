#ifndef MATRICES_H
#define MATRICES_H
#include <iterator>
#include <utility>
#include <vector>
#include "typedefs.hpp"
#include "utils.hpp"
#include "grade.hpp"

#ifdef ARRAY_MATRICES
    #include "ArrayColumn.hpp"
    typedef ArrayColumn ColumnType;
    const bool array_columns = true;
#else
    #include "HeapColumn.hpp"
    typedef HeapColumn ColumnType;
    const bool array_columns = false;
#endif

/** Matrix in column sparse format.
 * 
 * As of now, the underlying ColumnType can be HeapColumn or ArrayColumn. */
class SparseMatrix {
    public:
    static SparseMatrix eye(uint rows);                           ///< Matrix with ones on the diagonal
    static SparseMatrix ones(uint rows, uint columns);            ///< Matrix filled with ones
    explicit SparseMatrix(uint rows = 0, uint columns = 0);       ///< Zero matrix of given size

    // Other constructors and methods.
    SparseMatrix(uint rows, std::vector<std::vector<uint>> entries, ColumnType::sorting how_sorted = ColumnType::unsorted);
    SparseMatrix(uint rows, std::vector<ColumnType> data);

    void column_operation(uint from_column, uint to_column);
    void column_operation(const SparseMatrix& other, uint from_column, uint to_column);
    void add_one(uint row, uint column);
    index_t pivot(uint column){
        return data[column].pivot();
    }
    index_t pop_pivot(uint column){
        return data[column].pop_pivot();
    }
    void consolidate();                                         ///< Only relevant if ColumnType = HeapColumn. See HeapColumn::consolidate().
    ColumnType operator*(const ColumnType& other) const;
    SparseMatrix operator*(const SparseMatrix& other) const;
    SparseMatrix& operator+=(const SparseMatrix& other);
    SparseMatrix operator+(const SparseMatrix& other) const&;
    SparseMatrix operator+(const SparseMatrix& other) &&;
    bool operator==(const SparseMatrix& other) const;
    SparseMatrix transpose() const;
    SparseMatrix anti_transpose() const;                          ///< The antitranspose $M^\bot$ of an $m \times n$-matrix $M$ has entries $M^\bot_{ij} = M_{n-j, m-i}$.
    ColumnType& operator[](uint j){
        return data[j];
    }
    const ColumnType& operator[](uint j) const {
        return data[j];
    }
    bool has_unique_pivots();                                   ///< Checks if all *non-zero* columns have distinct pivots.
    std::vector<index_t> pivot_rows();
    std::vector<size_t> nonzero_pivot_rows();
    pivot_map_t pivot_columns();                       ///< Returns an array \c a such that \c a[i] th column is the leftmost column with pivot \c i.
    std::vector<size_t> zero_column_indices();                  ///< Indices of columns that are zero.
    std::vector<size_t> non_zero_column_indices();
    SparseMatrix non_zero_columns() &;
    SparseMatrix non_zero_columns() &&;
    bool is_zero();
    SparseMatrix get_columns(const std::vector<size_t>& indices) const &;
    SparseMatrix get_columns(const std::vector<size_t>& indices) &&;
    void reindex_columns(const std::vector<size_t>& indices);   ///< The jth column becomes the indices[j]-th column. Inverse to get_columns().
    SparseMatrix get_rows(const std::vector<size_t>& indices) const;
    SparseMatrix get_rows_consolidate(const std::vector<size_t>& indices) &&;
    void reindex_rows(const std::vector<size_t>& indices);      ///< The ith row becomes the indices[i]-th row. Inverse to get_rows().
    SparseMatrix& h_extend(SparseMatrix other);                     ///< Extend matrix horizontally to the right.
    SparseMatrix& h_extend(ColumnType column);
    SparseMatrix& v_extend(const SparseMatrix& other);              ///< Extend matrix vertically at the bottom.
    SparseMatrix& v_extend(uint top = 0, uint bottom = 0);
    std::string stats() const;                                  ///< Give some information about the matrix.
    bool is_well_defined() const;                               ///< Checks if all rows have the same size, and runs HeapColumn::is_well_defined() or ArrayColumn::is_well_defined() on all.
    operator std::vector<std::vector<uint>>() && {
        return std::vector<std::vector<uint>>(std::make_move_iterator(data.begin()), std::make_move_iterator(data.end()));
    }
    uint rows() const {
        return n_rows;
    }
    uint columns() const {
        return data.size();
    }
    private:
    uint n_rows;
    std::vector<ColumnType> data;
};


/** Generic Graded Matrix with template-specified underlying (ungraded) matrix type. */
template<class T>
class GenericGradedMatrix {
    public:
    GenericGradedMatrix():
        GenericGradedMatrix({}, {}, T())
    {}
    GenericGradedMatrix(std::vector<grade> row_grades,  std::vector<grade>col_grades, T data);
    template<class S>
    GenericGradedMatrix(const GenericGradedMatrix<S> &other):
        row_grades(other.row_grades), column_grades(other.column_grades), data((T)other.data)
    {}
    size_t rows() const;
    size_t columns() const;
    auto get_columns(std::vector<size_t> indices) const & {
        return GenericGradedMatrix(
            row_grades,
            get_elements(column_grades, indices),
            data.get_columns(indices)
        );
    }
    auto get_columns(std::vector<size_t> indices) && {
        return GenericGradedMatrix(
            std::move(row_grades),
            get_elements(std::move(column_grades), indices),
            std::move(data).get_columns(indices)
        );
    }
    auto get_rows(std::vector<size_t> indices) && {
        return GenericGradedMatrix(
            get_elements(std::move(row_grades), indices),
            std::move(column_grades),
            std::move(data).get_rows(indices)
        );
    }
    template<class S>
    GenericGradedMatrix<T> operator*(const S &other){
        assert(column_grades == other.row_grades);
        return GenericGradedMatrix<T>(row_grades, other.column_grades, data * other.data);
    }
    /** Checks if the matrix represent a map of free graded modules.
     *
     * This is the case if the row grade of any non-zero entry is <= its row grade.
     * @param transpose if true, check if the transpose is well-defined.
     */
    bool is_well_defined(bool transpose=false);
    
    /** Checks if the matrix, if considered as a chain complex, contains no balls as summands. 
     * 
     * This is the case if the row grade of any non-zero entry is < its row grade. 
     */
    bool is_minimal();
    std::vector<grade> row_grades, column_grades;
    T data;
};

/**
 * Additional methods for a graded matrix with an underlying SparseMatrix.
 */
class GradedMatrix: public GenericGradedMatrix<SparseMatrix> {
    public:
    static GradedMatrix ones(std::vector<grade> row_grades, std::vector<grade> column_grades);
    GradedMatrix();
    GradedMatrix(std::vector<grade> row_grades, std::vector<grade> column_grades);
    GradedMatrix(std::vector<grade> row_grades, std::vector<grade> column_grades, SparseMatrix);
    template<class T> GradedMatrix(GenericGradedMatrix<T> m):
        GradedMatrix(std::move(m.row_grades), std::move(m.column_grades), (SparseMatrix)std::move(m.data))
    {}
    GradedMatrix(std::initializer_list<GradedMatrix>);
    GradedMatrix operator*(const GradedMatrix&);
    GradedMatrix& operator+=(const GradedMatrix&);
    GradedMatrix operator+(const GradedMatrix&);

    /** Dual matrix.  This has as
     * - row grades: the negated column grades of the original matrix in reverse order,
     * - column grades: the negated row grades of the original matrix in reverse order,
     * - entries: the anti-transpose of the original matrix.
     * The dual matrix is well-defined iff the original is.
     */
    GradedMatrix D() const;
    /** Naive transposition */
    GradedMatrix T() const;
    GradedMatrix& h_extend(GradedMatrix);
    GradedMatrix& v_extend(const GradedMatrix&);
    GradedMatrix& v_extend(GradedMatrix&&);
    GradedMatrix& v_extend(const std::vector<grade> &top = {}, const std::vector<grade> &bottom = {}){
        row_grades.insert(row_grades.begin(), top.begin(), top.end());
        row_grades.insert(row_grades.end(), bottom.begin(), bottom.end());
        data.v_extend(top.size(), bottom.size());
        return *this;
    }
    void append_column(ColumnType column, grade z){
        data.h_extend(std::move(column));
        column_grades.push_back(z);
    }
    void reindex_rows(const std::vector<size_t> &indices);
    void reindex_columns(const std::vector<size_t> &indices);
    GradedMatrix get_rows(std::vector<size_t> indices) const;
    GradedMatrix get_columns(std::vector<size_t> indices) const &;
    GradedMatrix get_columns(std::vector<size_t> indices) &&;
    template<class C = grade::colex_less> std::vector<size_t> sort_columns(C comp = C());       ///< Sort columns by grade::colex_less or grade::lex_less, and return the permutation.
    template<class C = grade::colex_less> std::vector<size_t> sort_rows(C comp = C());          ///< Sort rows by grade::colex_less or grade::lex_less, and return the permutation. 
};


#endif

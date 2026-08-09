#include "matrices.hpp"
#include "block_column_matrix.hpp"
#include "time_measurement.hpp"
#include "typedefs.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <cassert>
#include <string>
#include <vector>

SparseMatrix SparseMatrix::eye(uint rows) {
    SparseMatrix result(rows);
    result.data.reserve(rows);
    for(uint j = 0; j < rows; j++)
        result.data.emplace_back(rows, j);
    return result;
}
SparseMatrix SparseMatrix::ones(uint rows, uint columns) {
    return SparseMatrix(rows, std::vector<ColumnType>(columns, ColumnType::ones(rows)));
}
SparseMatrix::SparseMatrix(uint rows, uint columns):
    n_rows(rows), data(columns, ColumnType(rows)) {}
SparseMatrix::SparseMatrix(uint rows, std::vector<std::vector<uint>> entries, ColumnType::sorting how_sorted): 
    n_rows(rows) 
{
    data.assign(entries.size(), ColumnType());
    #pragma omp parallel for
    for(size_t j = 0; j < data.size(); j++){
        data[j] = ColumnType(rows, std::move(entries[j]), how_sorted);
    }
}
SparseMatrix::SparseMatrix(uint _rows, std::vector<ColumnType> _data):
    n_rows(_rows), data(std::move(_data)) {
    assert(std::all_of(data.begin(), data.end(), [&](auto &column){return rows() == column.rows();}));
}
void SparseMatrix::add_one(uint row, uint column) {
    data[column].add_one(row);
}
void SparseMatrix::consolidate() {
    #pragma omp parallel for
    for(uint j = 0; j < data.size(); j++) {
        data[j].consolidate();
    }
}
ColumnType SparseMatrix::operator*(const ColumnType& other) const {
    ColumnType result(n_rows);
    for(uint entry : other.data)
        result += data[entry];
    return result;
}
SparseMatrix SparseMatrix::operator*(const SparseMatrix& other) const {
    assert(columns() == other.n_rows);
    SparseMatrix result(n_rows, other.columns());
    #pragma omp parallel for
    for(uint j = 0; j < other.data.size(); j++) {
        result.data[j] = *this * other.data[j];
    }
    return result;
}

SparseMatrix& SparseMatrix::operator+=(const SparseMatrix& other) {
    assert(columns() == other.columns() && n_rows == other.n_rows);
    #pragma omp parallel for
    for(size_t j = 0; j < columns(); j++) {
        data[j] += other.data[j];
    }
    return *this;
}
SparseMatrix SparseMatrix::operator+(const SparseMatrix& other) const& {
    SparseMatrix result(*this);
    result += other;
    return result;
}
SparseMatrix SparseMatrix::operator+(const SparseMatrix& other) && {
    return *this += other;
}
bool SparseMatrix::operator==(const SparseMatrix& other) const {
    return (*this + other).is_zero();
}
SparseMatrix SparseMatrix::transpose() const {
    Timing::time t1("Transpose");
    std::vector<std::vector<uint>> entries(rows());
    {
        Timing::time t1("Collect entries");
        for(uint j = 0; j < columns(); j++)
            for(auto i : data[j].data)
                entries[i].push_back(j);
    }{
        Timing::time t2("Sort entries");
        return SparseMatrix(columns(), entries);
    }
}
SparseMatrix SparseMatrix::anti_transpose() const {
    Timing::time t("Anti-transpose");
    std::vector<std::vector<uint>> entries(rows());
    {
        Timing::time t1("Collect entries");
        for(uint j = 0; j < columns(); j++)
            for(auto i : data[j].data)
                entries[rows() - 1 - i].push_back(columns() - 1 - j);
    }{
        Timing::time t2("Sort entries");
        return SparseMatrix(columns(), entries, ColumnType::descending);
    }
}
bool SparseMatrix::has_unique_pivots() {
    std::set<index_t> pivots;
    for(uint j = 0; j < columns(); j++) {
        index_t i = pivot(j);
        if(i != no_pivot && pivots.count(i))
            return false;
        pivots.insert(i);
    }
    return true;
}
std::vector<index_t> SparseMatrix::pivot_rows() {
    std::vector<index_t> result(columns());
    for(uint j = 0; j < columns(); j++)
        result[j] = pivot(j);
    return result;
}
std::vector<size_t> SparseMatrix::nonzero_pivot_rows() {
    std::vector<size_t> result;
    result.reserve(columns());
    for(uint j = 0; j < columns(); j++)
        if(pivot(j) != no_pivot)
            result.push_back(pivot(j));
    return result;
}
pivot_map_t SparseMatrix::pivot_columns() {
    pivot_map_t result(n_rows, no_pivot);
    for(uint j = 0; j < columns(); j++) {
        index_t i = pivot(j);
        if(i != no_pivot && result[i] == no_pivot)
            result.set(i, j);
    }
    return result;
}
std::vector<size_t> SparseMatrix::zero_column_indices() {
    std::vector<size_t> result;
    for(size_t j = 0; j < columns(); j++)
        if(pivot(j) == no_pivot)
            result.push_back(j);
    return result;
}
std::vector<size_t> SparseMatrix::non_zero_column_indices() {
    std::vector<size_t> result;
    for(size_t j = 0; j < columns(); j++)
        if(pivot(j) != no_pivot)
            result.push_back(j);
    return result;
}
SparseMatrix SparseMatrix::non_zero_columns()& {
    std::vector<ColumnType> result;
    for(uint j = 0; j < columns(); j++)
        if(pivot(j) != no_pivot)
            result.push_back(data[j]);
    return SparseMatrix(n_rows, result);
}
SparseMatrix SparseMatrix::non_zero_columns() && {
    std::vector<ColumnType> result;
    for(uint j = 0; j < columns(); j++)
        if(pivot(j) != no_pivot)
            result.push_back(std::move(data[j]));
    return SparseMatrix(n_rows, result);
}
bool SparseMatrix::is_zero() {
    consolidate();
    for(uint j = 0; j < columns(); j++)
        if(pivot(j) != no_pivot)
            return false;
    return true;
}
SparseMatrix SparseMatrix::get_columns(const std::vector<size_t> &indices) const& {
    return SparseMatrix(n_rows, get_elements(data, indices));
}
SparseMatrix SparseMatrix::get_columns(const std::vector<size_t> &indices) && {
    return SparseMatrix(n_rows, get_elements(std::move(data), indices));
}
void SparseMatrix::reindex_columns(const std::vector<size_t>& indices){
    std::vector<ColumnType> new_data(columns());
    for(uint j = 0; j < columns(); j++)
        new_data[indices[j]] = std::move(data[j]);
    data = std::move(new_data);
}
SparseMatrix SparseMatrix::get_rows(const std::vector<size_t>& indices) const {
    assert(indices.empty() || *std::max_element(indices.begin(), indices.end()) < n_rows);
    std::vector<index_t> lut(n_rows, no_pivot);
    for(uint j = 0; j < indices.size(); j++)
        lut[indices[j]] = j;
    SparseMatrix result(indices.size(), columns());
    #pragma omp parallel for
    for(uint j = 0; j < columns(); ++j)
        data[j].get_rows(lut, result[j]);
    return result;
}
SparseMatrix SparseMatrix::get_rows_consolidate(const std::vector<size_t>& indices) && {
    assert(indices.empty() || *std::max_element(indices.begin(), indices.end()) < n_rows);
    std::vector<index_t> lut(n_rows, no_pivot);
    for(uint j = 0; j < indices.size(); j++)
        lut[indices[j]] = j;
    SparseMatrix result(indices.size(), columns());
    #pragma omp parallel for
    for(uint j = 0; j < columns(); ++j)
        std::move(data[j]).get_rows_consolidate(lut, result[j]);
    return result;
}
void SparseMatrix::reindex_rows(const std::vector<size_t>& indices){
    #pragma omp parallel for
    for(auto &column : data)
        column.reindex_rows(indices);
}
SparseMatrix& SparseMatrix::h_extend(SparseMatrix other) {
    assert(n_rows == other.n_rows);
    std::move(other.data.begin(), other.data.end(), std::back_inserter(data));
    return *this;
}
SparseMatrix& SparseMatrix::h_extend(ColumnType column) {
    assert(rows() == column.rows());
    data.push_back(std::move(column));
    return *this;
}
SparseMatrix& SparseMatrix::v_extend(uint top, uint bottom) {
    n_rows += top + bottom;
    for(auto& column : data)
        column.v_extend(top, bottom);
    return *this;
}
SparseMatrix& SparseMatrix::v_extend(const SparseMatrix& other) {
    assert(columns() == other.columns());
    n_rows += other.n_rows;
    for(uint j = 0; j < columns(); j++) {
        data[j].v_extend(other.data[j]);
    }
    return *this;
}
void SparseMatrix::column_operation(uint from_column, uint to_column) {
    assert(from_column != to_column);
    data[to_column] += data[from_column];
}
void SparseMatrix::column_operation(const SparseMatrix& other, uint from_column, uint to_column) {
    assert((this != &other && n_rows == other.n_rows) || from_column != to_column);
    data[to_column] += other.data[from_column];
}
bool SparseMatrix::is_well_defined() const {
    assert(std::all_of(data.begin(), data.end(), [&](const ColumnType& column){return column.rows() == rows();}));
    assert(std::all_of(data.begin(), data.end(), [](const ColumnType& column){return column.is_well_defined();}));
    return true;
}
std::string SparseMatrix::stats() const {
    std::string str = std::to_string(rows()) + "x" + std::to_string(columns()) + "-matrix. Densities:";
    std::map<uint, uint, std::greater<uint>> sizes;
    for(auto &column : data)
        sizes[column.data.size()]++;
    uint c = 0;
    for(auto& [size, n] : sizes){
        str += std::to_string(n) + "x" + std::to_string(size) + " ";
        if(++c > 10)
            break;
    }
    return str;
}

template<class T>
GenericGradedMatrix<T>::GenericGradedMatrix(std::vector<grade> _row_grades, std::vector<grade> _column_grades, T _data):
    row_grades(std::move(_row_grades)), column_grades(std::move(_column_grades)), data(std::move(_data)) {
    assert(rows() == data.rows() && columns() == data.columns());
}

template<class T>
size_t GenericGradedMatrix<T>::rows() const {
    return row_grades.size();
}

template<class T>
size_t GenericGradedMatrix<T>::columns() const {
    return column_grades.size();
}

template<>
bool GenericGradedMatrix<SparseMatrix>::is_well_defined(bool transpose){
    assert(rows() == data.rows() && columns() == data.columns());
    assert(data.is_well_defined());
    data.consolidate();
    for(uint j = 0; j < data.columns(); j++)
        for(index_t i : data[j].data)
            if((!transpose && !(row_grades[i] <= column_grades[j])) || (transpose && !(row_grades[i] >= column_grades[j]))){
                std::cerr << "Graded matrix not well-defined: non-zero entry (" << i << ", " << j << ") "
                    << "with row grade (" << row_grades[i].x << ", " << row_grades[i].y << ") "
                    << "and column grade (" << column_grades[j].x << ", " << column_grades[j].y << ")." << std::endl;
                return false;
            }
    return true;

}

template<>
bool GenericGradedMatrix<BlockColumnMatrix>::is_well_defined(bool transpose){
    assert(rows() == data.rows() && columns() == data.columns());
    auto l = row_grades.begin();
    for(uint i = 0; i < data.matrices.size(); ++i){
        auto &m = data.matrices[i];
        std::vector<grade> g;
        g.insert(g.end(), l, l + m.rows());
        GenericGradedMatrix<SparseMatrix> block(std::move(g), column_grades, m);
        if(!block.is_well_defined(transpose))
            return false;
        l += m.rows();
    }
    return true;
}

template<class T>
bool GenericGradedMatrix<T>::is_minimal(){
    for(uint j = 0; j < columns(); ++j){
        if(data.pivot(j) != no_pivot && row_grades[data.pivot(j)] == column_grades[j])
            return false;
    }
    return true;
}

template class GenericGradedMatrix<SparseMatrix>;
template class GenericGradedMatrix<BlockColumnMatrix>;

GradedMatrix GradedMatrix::ones(std::vector<grade> row_grades, std::vector<grade> column_grades) {
    uint r = row_grades.size(), c = column_grades.size();
    return GradedMatrix(std::move(row_grades), std::move(column_grades), SparseMatrix::ones(r, c));
}
GradedMatrix::GradedMatrix():
    GradedMatrix({}, {})
{};
GradedMatrix::GradedMatrix(std::vector<grade> row_grades, std::vector<grade> column_grades):
    GradedMatrix(row_grades, column_grades, SparseMatrix(row_grades.size(), column_grades.size()))
{};
GradedMatrix::GradedMatrix(std::vector<grade> row_grades, std::vector<grade> column_grades, SparseMatrix data):
    GenericGradedMatrix<SparseMatrix>(std::move(row_grades), std::move(column_grades), std::move(data))
{}
GradedMatrix GradedMatrix::D() const {
    std::vector<grade> row_grades_reversed, column_grades_reversed;
    std::transform(row_grades.rbegin(), row_grades.rend(), std::back_inserter(row_grades_reversed), [](grade g){return -g;});
    std::transform(column_grades.rbegin(), column_grades.rend(), std::back_inserter(column_grades_reversed), [](grade g){return -g;});
    return GradedMatrix(std::move(column_grades_reversed), std::move(row_grades_reversed), data.anti_transpose());
}
GradedMatrix GradedMatrix::T() const {
    return GradedMatrix(column_grades, row_grades, data.transpose());
}
GradedMatrix& GradedMatrix::h_extend(GradedMatrix other) {
    assert(row_grades == other.row_grades);
    column_grades.insert(column_grades.end(), other.column_grades.begin(), other.column_grades.end());
    data.h_extend(std::move(other.data));
    return *this;
}
GradedMatrix& GradedMatrix::v_extend(const GradedMatrix &other) {
    assert(column_grades == other.column_grades);
    row_grades.insert(row_grades.end(), other.row_grades.begin(), other.row_grades.end());
    data.v_extend(other.data);
    return *this;
}
GradedMatrix& GradedMatrix::v_extend(GradedMatrix &&other) {
    assert(column_grades == other.column_grades);
    row_grades.insert(row_grades.end(), other.row_grades.begin(), other.row_grades.end());
    data.v_extend(std::move(other.data));
    return *this;
}
void GradedMatrix::reindex_rows(const std::vector<size_t>& indices){
    std::vector<grade> new_row_grades(rows(), {0,0});
    for(size_t i = 0; i < rows(); ++i) 
        new_row_grades[indices[i]] = row_grades[i];
    row_grades = std::move(new_row_grades);
    data.reindex_rows(indices);
}
void GradedMatrix::reindex_columns(const std::vector<size_t>& indices){
    std::vector<grade> new_column_grades(columns(), {0,0});
    for(size_t i = 0; i < columns(); ++i) 
        new_column_grades[indices[i]] = column_grades[i];
    column_grades = std::move(new_column_grades);
    data.reindex_columns(indices);
}
GradedMatrix GradedMatrix::get_rows(std::vector<size_t> indices) const {
    return GradedMatrix(get_elements(row_grades, indices), column_grades, data.get_rows(indices));
}
GradedMatrix GradedMatrix::get_columns(std::vector<size_t> indices) const & {
    return GradedMatrix(row_grades, get_elements(column_grades, indices), data.get_columns(indices));
}
GradedMatrix GradedMatrix::get_columns(std::vector<size_t> indices) && {
    return GradedMatrix(std::move(row_grades), get_elements(std::move(column_grades), indices), std::move(data).get_columns(indices));
}
GradedMatrix GradedMatrix::operator*(const GradedMatrix &other){
    assert(column_grades == other.row_grades);
    return GradedMatrix(row_grades, other.column_grades, data * other.data);
}
GradedMatrix& GradedMatrix::operator+=(const GradedMatrix &other){
    assert(column_grades == other.column_grades && row_grades == other.row_grades);
    data += other.data;
    return *this;
}
GradedMatrix GradedMatrix::operator+(const GradedMatrix &other){
    GradedMatrix result(*this);
    result += other;
    return result;
}

template<class C> std::vector<size_t> GradedMatrix::sort_columns(C comp) {
    std::vector<size_t> permutation = indirect_sort(column_grades, comp);
    column_grades = get_elements(std::move(column_grades), permutation);
    data.reindex_columns(inverse_permutation_inj(permutation));
    return permutation;
}

template<class C> std::vector<size_t> GradedMatrix::sort_rows(C comp){
    std::vector<size_t> permutation = indirect_sort(row_grades, comp);
    row_grades = get_elements(std::move(row_grades), permutation);
    data.reindex_rows(inverse_permutation_inj(permutation));
    return permutation;
}

template std::vector<size_t> GradedMatrix::sort_columns<>(grade::colex_less);
template std::vector<size_t> GradedMatrix::sort_columns<>(grade::colex_greater);
template std::vector<size_t> GradedMatrix::sort_columns<>(grade::lex_less);
template std::vector<size_t> GradedMatrix::sort_columns<>(grade::lex_greater);
template std::vector<size_t> GradedMatrix::sort_rows<>(grade::colex_less);
template std::vector<size_t> GradedMatrix::sort_rows<>(grade::colex_greater);
template std::vector<size_t> GradedMatrix::sort_rows<>(grade::lex_less);
template std::vector<size_t> GradedMatrix::sort_rows<>(grade::lex_greater);
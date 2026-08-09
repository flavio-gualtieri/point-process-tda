#ifndef POINTCLOUD_H
#define POINTCLOUD_H
#include <memory>
#include <optional>
#include <string>
#include <iosfwd>
#include <fstream>
#include "typedefs.hpp"
#include "matrices.hpp"

/// Represents a chain or cochain complex.
class Complex: public std::enable_shared_from_this<Complex> {
    public:
    virtual GradedMatrix next_matrix() = 0;
    virtual bool is_cochain() = 0;
    // For convenience, allows to replace things like `complex = std::make_shared<TransposeComplex>(complex)` by `complex = complex->apply<TransposeComplex>()`.
    template<class W, typename... Args> 
    std::shared_ptr<Complex> apply(Args&&... args){
        return std::make_shared<W>(shared_from_this(), std::forward<Args>(args)...);
    }virtual ~Complex() {};
};

/// Transposes every matrix of a (co)chain complex, effectively turning one into the other.
class TransposeComplex: public Complex {
    public:
    TransposeComplex(std::shared_ptr<Complex> complex);
    GradedMatrix next_matrix();
    bool is_cochain();

    private:
    std::shared_ptr<Complex> complex;
};

/// Exchanges the first and second coordinate of all grades.
class FlipGrades: public Complex {
    public:
    FlipGrades(std::shared_ptr<Complex>);
    GradedMatrix next_matrix();
    bool is_cochain();

    private:
    std::shared_ptr<Complex> complex;
};

/**
 * Reads `GradedMatrix`es from `stream`.
 *
 * Binary format expected from `stream`:
 * Each matrix is represented as a sequence of bytes with the following meaning:
 * uint         ignored
 * uint         number of rows R
 * uint         number of columns C
 * R*[double, double] row grades
 * C*[double, double] column grades
 * C*[
 *  uint        number N of entries in the column
 *  N*[uint]    entries
 * ]
 */
class MatricesFromFile: public Complex {
    public:
    MatricesFromFile(std::string filename);
    GradedMatrix next_matrix();
    bool is_cochain(){
        return false;
    }
    private:
    std::ifstream stream;
    uint d = 0;
    std::optional<GradedMatrix> last_matrix;
};

class MatricesFromSccFile: public Complex {
public:
    MatricesFromSccFile(std::ifstream& file);
    virtual ~MatricesFromSccFile(){};
    GradedMatrix next_matrix();
    bool is_cochain(){return true;}

private:
    std::vector<GradedMatrix> matrices;
};

class AugmentComplex: public Complex {
    public:
    AugmentComplex(std::shared_ptr<Complex> complex);
    bool is_cochain();
    GradedMatrix next_matrix();

    private:
    std::shared_ptr<Complex> complex;
    uint dim=0;
};

class CheckIsComplex: public Complex {
    public:
    CheckIsComplex(std::shared_ptr<Complex> complex);
    GradedMatrix next_matrix();
    bool is_cochain();

    private:
    std::shared_ptr<Complex> complex;
    std::optional<GradedMatrix> last_matrix;
};

/** Save a chain complex in scc format.
 * 
 * Only works for chain complexes. See https://bitbucket.org/mkerber/chain_complex_format/src/master 
 * for a specification of the format. */
void save_complex_scc(std::ostream &str, std::shared_ptr<Complex> complex, uint maxdim);
#endif

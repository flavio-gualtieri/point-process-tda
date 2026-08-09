# 2pac (_**2-pa**rameter persistent **c**ohomology_)
Authors: Fabian Lenzen

This software package accompanies the article 

> Ulrich Bauer, Fabian Lenzen and Michael Lesnick: 
> “[Efficient Two-Parameter Persistence Computation via Cohomology](https://drops.dagstuhl.de/opus/volltexte/2023/17865)”,
> 39th International Symposium on Computational Geometry (SoCG 2023),
> Leibniz International Proceedings in Informatics (LIPIcs) 258,
> Dagstuhl, Germany, 2023.

An extended version is available on [arXiv.org](https://arxiv.org/abs/2303.11193).

# Summary
The program `2pac` computes a minimal free resolution of 2-parameter persistent cohomology and homology of the function-Vietoris-Rips complex on a finite metric space.
If the (co)homology of the complex is zero outside a finite rectangle in the parameter space $\mathbf{Z}^2$, 
resolutions of cohomology and homology determine each other uniquely.
Computing 2-parameter persistent cohomology is a novel approach explained in detail in the paper cited above.
The implementation of 2-parameter persistent homology follows the approach of [Fugacci, Kerber and Rolle](https://bitbucket.org/mkerber/mpfree/src/master).

# Building
## Requirements
The following tools and libraries are needed for building and running `2pac`.

 * A C++-compiler that supports C++20. The program was successfully built with Homebrew clang++ 17 on MacOS and g++ 11 on Ubuntu Linux.
 * [gnu make](https://www.gnu.org/software/make)
 * [Boost](https://www.boost.org)
 * OpenMP
 * Optional: pybind11 for building python bindings

**MacOS**: All these requirements can be installed through Homebrew:

    brew install llvm libomp boost

## Downloading and compiling
To retrieve and build the software, run

    git clone https://gitlab.com/flenzen/2-parameter-persistent-cohomology.git 2pac
    mkdir build && cd build
    cmake ..
    make

in your terminal.
This should download the sourcecode and produce a file `2pac` that can be run via `./2pac --help`.

**MacOS:** If `cmake` uses Apple's `clang++` instead of the Homebrew `clang++`, the above may produce a lot of warnings.
This is because Apple's `clang++` does not support OpenMP.
Consequently, 2pac is built without OpenMP support, which likely affects the performance.
To tell `cmake` to use Homebrew's `clang++`, first clean `cmake`'s cache by running `rm -rf *` inside `build`, 
and then call `cmake -D"$(brew --prefix llvm)/bin/clang++" ..`.

**Debug symbols, matrix type, documentation:**
To build with debug symbols, run `cmake -DCMAKE_BUILD_TYPE=Debug ..` instead (or in a different folder).
In this case, the binary will be called `2pac_debug`.
To build with list-of-vector sparse matrices instead of list-of-heaps, use `cmake -DVECTORS=1`. For details on the matrix types, see [below](#matrix-representations).
If you want to work on the source code, run `doxygen` to build the documentation.

## Python bindings
We also provide experimental code to build a python module exposing 2pac's functionality to python.
This is automatically enabled if pybind11 is installed.
In this case, after running `cmake` as described above, one can run `make 2pacpy`,
which will produce a file `twopac.(...).so` or `twopac.(...).dll`, where `(...)` depends on your platform.
(the spelling `twopac` is because importing python modules with a digit in their name is inconvenient).
If this file has been build successfully, run 

    python3 <<- EOF
    	import twopac
    	print(twopac.__doc__)
    EOF

to test if python can load it.
If you built with debug symbols, the module is called `twopac_debug` instead.
See [below](#python) for details on how to use this.

# Usage
## First steps
If the build was successful, run

    ./2pac --help
    
to make sure the program starts, and to get an overview of the command line arguments.
To compute something interesting, run

    ./2pac -f ../samples/1-sphere.bin -s50

This will compute minimal free resolutions of the persistent cohomology and homology of the function-Rips complex of sample provided in the file `samples/1-sphere.bin`; see `samples/1-sphere.pdf` for a plot of the sample.
Here, the argument `-s50` subsamples the input.
Running

    ./2pac -f ../samples/1-sphere.bin -s50 --cone

ensures that the input complex is eventually acyclic, i.e., has non-zero (co)homology only in a bounded region of $\mathbf{Z}^2$.
In this case, both computations yield the same graded Betti numbers, as you can verify from the output.
Actually, both methods compute the same resolution in this case (see the paper for details).

The notebook `EXAMPLE.ipynb` shows how to gerate a sample interactively, write it to a file, apply `2pac` to it, and plot the computed graded Betti numbers.
The notebook uses the [python module](#python-bindings).
Further samples can be generated using `Generate Samples.ipynb`.

## General workflow
<img align="right" src="flowchart.svg">
In general, the assumed workflow when using `2pac` is that from the user input (which may be a filtered finite metric space, a sequence of matrices, of a bifiltered graph), `2pac` generates a cochain complex, which basically amounts to a sequence of coboundary matrices.
These matrices undergo a series of preprocessings, before they are passed to the homology or cohomology resolution computation algorithm:


The steps that appear in the diagram are as follows:
* **Cochain complex**: `2pac` supports different kinds of input files (see [below](#input-file-formats) for details on file formats): 
  * take filtered finite metric spaces (just specify `-f [filename]` on command line) and compute **function-Rips complex**
  * a sequence of coboundary matrices (specify `-f [filename] --matrix-input` or `-f [filename] --scc-input`, depending on file format),
  * a $\mathbf{Z}^2$-filtered graph (specify `-f [filename] --clique-input`) and build **bifiltered clique complex**
* **Strong filtration domination preprocessing**: For clique complexes (and thus also for function-Rips complexes), [Alonso, Kerber and Pritam](https://doi.org/10.1137/1.9781611977561.ch3) proposed a method to reduce the size of bifiltered graphs without changing the homology type of the associated clique complex. 
  Specify `--sfd` to run this preprocessing on the complex.

* Chunk preprocessing ([Fugacci, Kerber](https://doi.org/10.4230/LIPIcs.SoCG.2019.37)) reduces the size of the input chain complex by removing by removing trivial summands before further processing.
  Chunk* preprocessing does the same on cochain complexes.
  Specify `-C`$n$ to split off all trivial summands from the input chain cocomplex up to dimension $n$, and `-DC`$n$ to do the same on chain complexes.
* The cohomology algorithm relies on the assumption that the filtration complexes $K_{x,y}$ become acyclic independently for sufficiently large $x$ and $y$,
  To establish this property, we provide two strategies:
  * by building algebraic cones (specify `--cone=1` and `--Cone=1`): replace $C^\bullet(K_{x,y})$ by the algebraic cone on $C^\bullet(K_{x,y})$ for $x = \infty$ (in the case of `--cone`) or $y = \infty$ (for `--Cone=1`).
  * by computing a 1-parameter homology basis: we compute a homology basis of the 1-parameter filtered complex $x \mapsto K_{x,\infty}$, and adjoin generators to $K$ for all non-ephemeral homology classes (`--cone=2`).
  For `--Cone=2`, the roles of $x$ and $y$ are switched.

## Command line arguments
We only mention the command line arguments that are not already sufficiently described in `./2pac --help`.

* `-f` or `--input filename` Input binary file describing a filtered metric space in the file format [described below](#function-rips-complexes).
* `-M` or `--MatrixFile` The input file specified `-f` is a sequence of coboundary matrices describing a cochain complex in the file format [described below](#matrices).
* `--chunkdim d` Makes `2pac` apply the [chunk algorithm](https://dx.doi.org/10.4230/LIPIcs.SoCG.2019.37) to the coboundary matrices up to dimension `d`. This algorithm computes a cochain complex homotopy equivalent to the input such that up to dimension `d`, the output cochain complex contains no homological balls as direct summands, i.e. summands of the form $0 \to M \xrightarrow{\cong} M \to 0$.
* `-D` Applies the chunk algorithm to boundary matrices and not coboundary matrices.
* `--cone` The theory behind the cohomology algorithm requires that the input complex has homology with bounded support. To ensure this, specifying `--cone` will make `2pac` make the input an acyclic complex for infinite values of the density filtration parameter (i.e., the second parameter when specifying explicit matrices).
The following strategies to achieve this are provided:
  * computing a persistence basis of the one-parameter complex that one obtains by forgetting the other (= Rips) filtration parameter,
    and adjoining cells to the complex for each homology class. To do this, specify `--cone 2` or just `--cone`.
  * taking the algebraic cone at infinity, i.e., the [mapping cone](https://en.wikipedia.org/wiki/Mapping_cone_(homological_algebra)) of the identity. To do this, specify `--cone 1`.
* `--Cone` behaves as `--cone`, but acts on the first (= Rips) filtration parameter.
Specifying `--Cone` is necessary for the output of the cohomology algorithm to work correctly if the Rips filtration is cut off (by specifying `-t`).
* `--sfd` If the input is a bifiltered clique complex (e.g., a function Rips complex), then specifying `--sfd` makes `pac` apply the [strong filtration domination preprocessing](https://doi.org/10.1137/1.9781611977561.ch3) before computing the (co)homology resolution. This may save a lot of time.

# Python bindings
You can use the python bindings to invoke `2pac` directly from python.
At the moment, the following classes are exposed:

* `FunctionRipsComplex`. Inherits from [`Complex`](#complex). Methods:
  * `__init__(vertex_filtrtation: [double], dist_matrix: [[double]])`: of the distance matrix, only the lower triangular part is used. Entries on the diagonal and above are not used (and not checked).
  * `sfd(edge_ordering = rcolex) -> FunctionRipsComplex`: apply [strong filtration domination preprocessing](https://doi.org/10.1137/1.9781611977561.ch3).
    Edge orderings supported are (r)(co)lex.
  * Methods inherited from `Complex`: see [below](#complex)
* `Complex`<a name="#complex"></a>. Methods:
  * `is_cochain() -> bool`
  * `T() -> Complex` turns chain complexes into cochain complexes (and vice versa) by transposing the (co)boundary matrices.
  * `Cone(p: int) -> Complex` Make the complex at infinite x- or y-coordinate, depending on $p$.
  * `HBasisCone() -> Complex` Like Cone, but via the computation of a persistence basis.
  * `Cohomology(d: int) -> generator` Gives a generator of cohomology resolutions, up to dimension $d$.
  * `Homology(d: int) -> generator` Gives a generator of homology resolutions, up to dimension $d$.

Example usage:

    from bindings_debug import *
    import numpy as np
    N = 25
    points = np.random.rand(N, 3)
    grades = np.random.rand(N)
    dists  = np.linalg.norm(points[:,None,:] - points[None,:,:], axis=-1)
    
    # For homology with strong filtration domination and chunk* preprocessing:
    cpx = FunctionRipsComplex(grades, dists)
    for bettis in cpx.sfd().Chunk(3).Homology(2):
        print(bettis)
    
    # For homology with strong filtration domination and coning off:
    cpx = FunctionRipsComplex(grades, dists)
    for bettis in cpx.sfd().Cone(1).Cohomology(2):
        print(bettis)

# Input file formats
The program can be given the following input data:
* **Filtered point cloud data** specify `./2pac -f filename` to interpret `filename` (see [below](#function-rips-complexes) for file format) as a filtered finite metric space, and build function-Rips complex of it.
* **Clique complexes** specify `./2pac -f filename --clique` to interpret `filename` (see [below](#clique-complexes) for the format) as a 
* **Chain complex by matrices** specify `./2pac -f filename --matrix-input` (binary format, see [below](#matrices)) or `./2pac -f filename --scc-input` (scc2020 format, see [here](https://bitbucket.org/mkerber/chain_complex_format/src/master/) to interpret `filename` as a sequence of matrices specifying a complex.

## Function-Rips complexes
Specify as binary file format with the following structure:

 * a 4-byte uint with value 0. This might be another number in the future.
 * a 4-byte uint giving the number $n$ of points in the file.
 * $n$ 8-byte doubles, giving the function values of the points.
 * $n(n+1)$ 8-byte non-negative doubles, giving the pairwise distances $d(0,0), d(0,1), \dotsc, d(0,n), d(1,1), \dotsc, d(1,n), \dotsc$.

Additional data is ignored.

## Clique-complexes
Specify as binary file with the following structure:

 * a 4-byte integer giving the number $n$ of vertices
 * $n$ pairs (8-byte double, 8-byte double), giving the vertex grades $(x_1, y_1), (x_2, y_2), \dotsc$.
 * a 4-byte integer giving the number $m$ of edges
 * $m$ many tuples (4-byte integer, 4-byte integer, 8-byte double, 8-byte double) that define the indices of the two endpoints of each edge, and the edge's grade.

## Matrices
Calling `./2pac -Mf matrices.bin` makes `2pac` read the input `matrices.bin` as a sequence of graded matrices that form a cochain complex.
The file format for the binary files is as follows:

  * there is no header. For each coboundary matrix, the file has to contain:
  * a 4-byte integer with value 0
  * a 4-byte integer $m$ giving the number of rows
  * a 4-byte integer $n$ giving the number of columns
  * $2m$ many 8-byte doubles with the grades $(x, y) \in \mathbf{R}^2$ of the rows
  * $2n$ many 8-byte doubles with the grades $(x, y) \in \mathbf{R}^2$ of the columns
  * for each column index $j = 1,\dotsc,n$ the following information:
     * a 4-byte integer $s$
     * $s$ many 4-byte integers $i \leq m$ of row indices of the non-zero entries in the $j$th column

The matrix $M$ has to satisfy $M_{ij} = 0$ if the $i$th row grade is not less or equal (in the partial order on \mathbf{R}^2) than the $j$th column grade.
The matrices have to form a cochain complex.

# Notes on the implementation
If you want to work on the code, consider building the documentation by running `doxygen`.

## Matrix representations
`2pac` supports to ways to represent the columns of a matrix.
The default is [heaps](#heaps); to use [vectors](#vectors), build the code with `cmake -DVECTORS=1`.

#### Heaps
In this case, a matrix column is represented as a binary heap (using `boost::priority_queue`), such that the pivot is the head of the heap.  A column may contain more than one entry per row index.  The process of merging all these we call *consolidation*.  The pivot is consolidated at any time.

### Vectors
In this case, a matrix column is represented as a descendingly sorted `std::vector` without repetition (i.e., the pivot is the first entry, and to every row index, there is at most one entry).  Actually, because minimization requires the operation `pop_pivot` for a column, this column representation uses a wrapper `skip_vector` around `std::vector` that implements the operation `pop_front` by just skipping the first elements.



<!-- #### Indirection
In the variants *\*.vectors* using vectors, the following indirection scheme can be switched on by a command line flag.  Given as input a matrix `M`, we represent a column of the product `MV` (for a reduction matrix `V`) by a heap `p` as follows:

 * the entries of `p` are pairs `(q, i)` such that `q` is a pointer to a column of `M`, represented as a [vector](#vectors);
 * the heap is ordered by the relation `(q1, i1) < (q2, i2)` if `q[i] < q'[i]`.

The interpretation of the entries is that the column of `MV` is a linear combination of the columns of `M` ocurring as a `q` in `p`, and in each of these columns, all entries with index ≤ `i` sum up to zero with entries of other columns in `p`.

In minimization and reduction, a column may be operated on, and later be added to some other column (in minimization, only the local columns can be added to others). In the *semi-indirect* approach, when the reduction of a column (using the indirect approach described above) finishes, it is written back to the matrix explicitly. In the *fully indirect* approach, the heap for that column is stored, and merged upon addition to another column.

At the moment, indirection is only partly implemented, and no significant performance gains have been observed from it. -->

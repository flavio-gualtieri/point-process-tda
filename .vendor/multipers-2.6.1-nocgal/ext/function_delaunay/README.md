# function_delaunay #

Copyright 2023 TU Graz

### Author ###

Michael Kerber

### Summary ###

This software computes the sublevel Delaunay-Cech bifiltration as
defined in the paper

Angel Alonso, Michael Kerber, Tung Lam, Michael Lesnick: Delaunay Bifiltrations of Functions on Point Clouds. arxiv, 2023

The software accepts a text file in which a point cloud is specified,
with one point per line. The coordinates of the point are given as decimal
numbers separated by blank space. The number of coordinates per line
must be the same, but can be arbitrary.

The output is a chain complex, given in [scc2020 format](https://bitbucket.org/mkerber/chain_complex_format/src/master/) that specifies the simplices
of the bifiltration and their grades.

### Requirements ###

The library is written in C++ and requires a compiler that realizes the C++11-standard. Also *cmake* is required in version 3.9 or higher.

function_delaunay depends on numerous libraries: 

* first and foremost,
both [CGAL](https://www.cgal.org), [Gudhi](https://gudhi.inria.fr/index.html) and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) are required. In the CMakeLists.txt file, the paths to these libraries can
be specified using the variables CGAL_DIR, GUDHI_INCLUDE_DIR, and EIGEN_INCLUDE_DIR
The program has been tested with CGAL 5.5.1 and Gudhi 3.8 and Eigen 3.3.9.

* Furthermore, function_delaunay requires the libraries [phat](https://bitbucket.org/phat-code/phat/src/master/), scc,
and mpp_utils. For convenience, these libraries are included in the
repository.

* The main program further includes options that require the libraries
[mpfree](https://bitbucket.org/mkerber/mpfree/src/master/) and [multi-chunk](https://bitbucket.org/mkerber/multi_chunk/src/master/). Again, these libraries
are included in the project. They are not needed when function_delaunay
is included in another project.

* The timer library of *Boost* is used to measure the performance of the substeps of the algorithm. If *Boost* is not found, the timer are simply disabled. The code was tested using *Boost* 1.74

## Usage

A CMakeLists.txt file is included. The executable main is created using

	cmake .
	make

Upon compilation, we can test the program via

	./main ./sample_input.txt output.txt

and compare the result with the file `sample_output.txt` in the main directory.

We can also compute the sublevel Delaunay-Cech bifiltration
of 1000 randomly generated points in R^3 by

	./main --random 1000 3 output.txt

Try `main --help` for various other options.

### Interface ###

To use the implementation within other projects, use the function given in the file include/function_delaunay/function_delaunay_with_meb.h:

	template<typename GrMat>
	long function_delaunay_with_meb(std::vector<Point_with_density> &input_points,
				    std::vector<GrMat> &graded_matrices,
				    bool only_return_complex_size=false)

The vector input_points contains a set of points with density,
as defined in the file `include/function_delaunay/Point_with_density.h`.
`graded_matrices` is the result of the computation, containing the
matrices defining the maps in the (graded) chain complex that is computed.
Here, the template argument `GrMat` must be matrix type that represents
graded matrices. A possible instantiation is

	mpp_utils::Graded_matrix<>

The resulting matrices can be written in an output stream via the command

	mpp_utils::print_in_scc_format(graded_matrices,ofstr);


### License ###

The software is published under the GNU General Public License (GPL).

### Contact ###

Michael Kerber (kerber@tugraz.at)

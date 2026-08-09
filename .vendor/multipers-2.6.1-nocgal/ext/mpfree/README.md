# mpfree #

Copyright 2020 TU Graz

### Authors ###

Michael Kerber and Alexander Rolle

### Summary ###

This software computes the **m**inimal **p**resentation of a **free** implicit
representation. It is based on the paper

Michael Kerber, Alexander Rolle: Fast Minimal Presentations of Bi-graded Persistence Modules. ALENEX 2021

which itself is an improvement of the result described in

Michael Lesnick and Matthew Wright. Computing minimal presentations and bigraded betti numbers of 2-parameter persistent homology. [arXiv:1902.05708](https://arxiv.org/abs/1902.05708)

The software accepts a text file in scc2020 (explained [here](https://bitbucket.org/mkerber/chain_complex_format/src/master/)) or firep format (explained [here](https://rivet.readthedocs.io/en/latest/inputdata.html#firep-algebraic-input))
and computes a (usually much smaller) text file that is also in scc2020 format and represents the minimal presentation of the input.

### Requirements ###

The library is written in C++ and requires a compiler that realizes the C++11-standard. Also *cmake* is required in version 3.9 or higher.

Further dependencies are optional:

* mpfree uses the *phat* library for representing matrices, but it comes with its own (slightly modified) version of *phat*. If desired, the phat version can be changed by setting the PHAT_INCLUDE_DIR variable of cmake to the appropriate include-path of phat. At least version 1.6 of *phat* is required.
* The timer library of *Boost* is used to measure the performance of the substeps of the algorithm. If *Boost* is not found, the timer are simply disabled. The code was tested using *Boost* 1.71
* *OpenMP* is used to parallelize the execution (see below)

## Usage

A CMakeLists.txt file is included. The executables are created using

	cmake .
	make

The programs mpfree and mpfree_sequential require both an input file and an output file and computes the minimal presentation. The former is only compiled if *OpenMP* is installed, and uses by default the maximal number of cores in the computation, while mpfree_sequential does not use *OpenMP*. For both executables, the -v option shows the progress of the computation, and timing information if *Boost* is enabled. There are several more options for the algorithm which can be listed with the -h flag. See the paper above for details.


### Interface ###

To use the implementation within other projects, use the function given in the file include/mpfree.h:

	template< typename ColumnType=phat::vector_vector,
	          typename CoordinateTraits=Coordinate_traits_with_map<double> >
	void compute_minimal_presentation(const std::string& infile,
	                                  const std::string& outfile,
	                                  bool use_chunk=true,
	                                  bool use_clearing=false)

In the simplest form, it takes two strings representing the input and the output file. The template arguments have default values, but can be modified to adapt to other contexts. Their meanings are:

* ColumnType: The data structure to store the columns of matrices. All column types of the phat library are supported.
* CoordinateTraits: Defines how grade coordinates are handled. Specifically, the template parameter class should contain a type `Coordinate` which is used to store coordinates, functions `from_string` and `to_string` to convert coordinates from and to string, and a functor class `Compare` to check which of two coordinates is smaller. The default choice `Coordinate_traits_with_map<double>` interprets theinput coordinates as doubles, and writes precisely the same input string also into the output file. It is required that two distinct strings are converted into distinct double coordinates.

### Testing ###

To test that mpfree and mpfree_sequential are working as expected, run the script test.py in the tests directory (requires at least Python 3.5). This simply checks that the outputs of mpfree and mpfree_sequential agree with precomputed minimal presentations of the sample inputs in the sample_files directory.

### License ###

The software is published under the GNU Lesser General Public License (LGPL).

### Contact ###

Michael Kerber (kerber@tugraz.at)

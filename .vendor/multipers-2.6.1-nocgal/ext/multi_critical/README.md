# multi-critical #

Copyright 2025 TU Graz

### Author ###

Michael Kerber

### Summary ###

This software contains algorithms to process multi-criticial bifiltered
chain complexes. In particular, it contains algorithm to compute
a free implicit representation in a fixed homology dimension,
implementing an algorithm by Chacholski, Scolamiero, and Vaccarino.
Moreover, it contains two algorithms to compute a free resolution
of the input chain complex, that is, a 1-critical chain complex
that is quasi-isomorphic. All algorithm are described in the paper

Ulrich Bauer, Tamal Dey, Michael Kerber, Florian Russold, Matthias S&ouml;ls: Fast free resolutions of bifiltered chain complexes. [arXiv:2512.08652](https://arxiv.org/abs/2512.08652)

### Requirements ###

The library is written in C++ and requires a compiler that with the C++14-standard. Also *cmake* is required in version 3.9 or higher.
Optionally, the timer library of *Boost* is used to measure the performance of the substeps of the algorithm. If *Boost* is not found, the timer are simply disabled. The code was tested using *Boost* 1.74.
The library also uses the libraries *phat*, *scc*, *mpp_utils*, *multi_chunk*, and *mpfree* for various tasks. Current versions of these libraries are included, but a different version can be used adapting the CMakeLists.txt file.

## Usage

A CMakeLists.txt file is included. The executable called *multi_critical* is created using

```
    cmake .
    make
```

That program expects an input file in "extended" scc2020 format representing a multi-critical 2-parameter chain complex. The format is the same as for scc2020 (explained [here](https://bitbucket.org/mkerber/chain_complex_format/src/master/)), except that instead of two grades, an arbitrary even number of grades can be specified, defining the upsets in which the basis element is present. Moreover, the semicolon as separator between grades and boundary is mandatory.
The folder sample_files contains some examples.

The output file is either a free implicit representation, a minimal presentation, or a 1-critical chain complex in "proper" scc2020 format.

There are several more options for the algorithm which can be listed with the -h flag. See the paper above for details.


### Interface ###

The csv algorithm is found in the file include/multi_critical/firep.h

```
    template<typename ParserType, typename GradedMatrix>
    void firep_via_csv(ParserType& parser,
                       int dim,
                       GradedMatrix& GM1,
                       GradedMatrix& GM2,
                       bool interpret_dim_as_hom_dim=true)
```

where the free implicit representation of the input (given as parser object) is computed and stored in the matrices GM1 and GM2. Such a parser object is easily created via

```
    std::ifstream ifstr(infile);
    scc::Scc<> parser(ifstr);
```

The other main routine is found in include/multi_critical/free_resolution.h

```
    template<typename ParserType, typename GradedMatrix>
    void free_resolution(ParserType& parser,
                         std::vector<GradedMatrix>& result,
                         bool use_logpath=true)
```

where the result is stored as a sequence of matrices in the result vector.



### License ###

The software is published under the GNU Lesser General Public License (LGPL).

### Contact ###

Michael Kerber (kerber@tugraz.at)

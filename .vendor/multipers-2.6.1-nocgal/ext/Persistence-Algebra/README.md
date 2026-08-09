# Persistence-Algebra

A C++17 header-only library for computational multiparameter persistent homology. It provides flexible, efficient data structures and algorithms for working with **presentations of persistence modules** — graded linear maps between free modules over arbitrary posets — with a focus on the two-parameter case over ℝ².

This library forms the algebraic backbone of the author's PhD research and is used directly by three companion projects:

- **[AIDA](https://github.com/JanJend/AIDA)** — Approximation and Interval Decomposition Algorithm
- **[Skyscraper-Invariant](https://github.com/JanJend/Skyscraper-Invariant)** — computation of the skyscraper invariant for 2-parameter persistence modules
- **[Stable-Decomposition](https://github.com/JanJend/Stable-Decomposition)** — stable decomposition of multiparameter persistence modules

---

## Table of Contents

- [Background](#background)
- [Citing This Work](#citing-this-work)
- [Related Projects](#related-projects)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation and Build](#installation-and-build)
- [File Formats](#file-formats)
- [Library Structure](#library-structure)
- [Command-Line Tools](#command-line-tools)
- [Usage Examples](#usage-examples)
- [License](#license)

---

## Background

A **persistence module** over a poset *P* is a functor from *P* to vector spaces over a field (here 𝔽₂ for now). We store them via **presentations**: graded linear maps

```
d : F₁ → F₀
```

between free graded modules, whose cokernel is the persistence module of interest. This is because in practice these matrices are much smaller than storing all of the necessary structure maps. This library provides the data structures and algorithms needed to construct, manipulate, minimise, and analyse such presentations, as well as to compute derived algebraic objects such as kernels, resolutions, and homomorphism spaces.

---

## Citing This Work

If you use this library in your research, please cite it via its DOI. The full citation metadata is in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{Jendrysiak_Persistence_Algebra,
  author       = {Jendrysiak, Jan},
  title        = {Persistence-Algebra},
  version      = {0.2},
  year         = {2024},
  license      = {LGPL-3.0-or-later},
  doi          = {10.4230/artifacts.23283},
  orcid        = {https://orcid.org/0000-0002-3761-3463}
}
```

If you use this library through or alongside **AIDA**, please also cite the accompanying paper:

```bibtex
@InProceedings{djk25,
  author    = {Dey, Tamal K. and Jendrysiak, Jan and Kerber, Michael},
  title     = {{Decomposing Multiparameter Persistence Modules}},
  booktitle = {41st International Symposium on Computational Geometry (SoCG 2025)},
  pages     = {41:1--41:19},
  series    = {Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN      = {978-3-95977-370-6},
  ISSN      = {1868-8969},
  year      = {2025},
  volume    = {332},
  editor    = {Aichholzer, Oswin and Wang, Haitao},
  publisher = {Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address   = {Dagstuhl, Germany},
  doi       = {10.4230/LIPIcs.SoCG.2025.41}
}
```

A preprint is also available on arXiv: [arXiv:2504.08119](https://arxiv.org/abs/2504.08119).

The kernel computation for ℝ²-graded matrices is adapted from the **MPfree** algorithm by Michael Kerber and Alexander Rolle; please cite the relevant MPfree paper if that component is central to your use.

@inProceedings{mpfree,
author = {Michael Kerber and Alexander Rolle},
title = {Fast Minimal Presentations of Bi-graded Persistence Modules},
booktitle = {Algorithm Engineering and Experiments (ALENEX)},
year = {2021},
doi = {10.1137/1.9781611976472.16},
}

---

## Related Projects

This library is used as a dependency in the following projects, all part of the author's PhD thesis at TU Graz:

- **[AIDA](https://github.com/JanJend/AIDA)** — Computing indecomposable decompositions of multiparameter persistence modules; published at [SoCG 2025](https://doi.org/10.4230/LIPIcs.SoCG.2025.41) (Dey, Jendrysiak, Kerber)
- **[Skyscraper-Invariant](https://github.com/JanJend/Skyscraper-Invariant)** — efficient computation of the skyscraper invariant (Fersztand, Jendrysiak)
- **[Stable-Decomposition](https://github.com/JanJend/Stable-Decomposition)** — algorithm to compute the Pruning (Bjerkevik '24) to stabilise decomposition of multiparameter modules
(Bjerkevik, Jendrysiak, Lenzen)
---

## Features

- **Flexible matrix types** over 𝔽₂, with columns stored as sorted index vectors, sets, or dense bitsets
- **Graded sparse matrices** parametrised by an arbitrary degree type `D`, with existing specialisations for ℝ¹, ℝ², and ℝ³
- **Column reduction algorithms**: standard, triangular, graded, with and without tracking of performed operations
- **Kernel computation** for ℝ²-graded matrices, adapted from the MPfree algorithm (Kerber, Rolle - Implementation by Michael Kerber, TU Graz)
- **Minimisation** of graded presentations
- **Free resolutions** up to second syzygies
- **Hom-space computation** between graded modules
- **Snap-to-grid** operations for discretising presentations
- **Quiver representation** extraction from presentations of the corresponding representations of quivers
- Reading and writing of `.scc` and `.firep` file formats (compatible with RIVET, mpfree, and related tools)
- OpenMP parallelisation for matrix transformations
- Comprehensive template design: column type, index type, and degree type are all template parameters

---

## Prerequisites

- A C++17-compatible compiler (GCC ≥ 9, Clang ≥ 9, MSVC ≥ 19.14)
- [CMake](https://cmake.org/) ≥ 3.15
- [Boost](https://www.boost.org/) ≥ 1.70 (components: `timer`, `system`; `dynamic_bitset` is header-only)

On Ubuntu/Debian:

```bash
sudo apt install cmake libboost-all-dev
```

On macOS with Homebrew:

```bash
brew install cmake boost
```

---

## Installation and Build

Clone the repository and build with CMake:

```bash
git clone https://github.com/JanJend/Persistence-Algebra.git
cd Persistence-Algebra
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

For a debug build (produces executables with a `_debug` suffix):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

To install the command-line tools system-wide:

```bash
sudo make install
```

### Using the library as a header-only dependency

Since the core functionality lives entirely in the `include/grlina/` headers, you can integrate this into your own CMake project by adding the include path:

```cmake
target_include_directories(your_target PRIVATE path/to/Persistence-Algebra/include)
find_package(Boost REQUIRED COMPONENTS timer system)
target_link_libraries(your_target Boost::timer Boost::system)
```

---

## File Formats

The library reads and writes two standard formats used in the multiparameter persistence community.

### `.scc` (Sparse Column-Compressed)

```
scc2020
2
<num_cols> <num_rows> 0
<x1> <y1> ; <entry1> <entry2> ...
...
<x1> <y1> ;
...
```

The first block (num_cols lines) gives the relations with their degrees and non-zero row indices. The second block (num_rows lines) gives the generator degrees with no entries.

### `.firep` (Free-to-Free Presentation)

Similar structure but with a different header and an additional line describing the ambient chain complex level.

Both formats are compatible with [RIVET](https://github.com/rivetTDA/rivet), [mpfree](https://bitbucket.org/mkerber/mpfree), and [multipers](https://github.com/DavidLapous/multipers).

---

## Library Structure

```
include/grlina/
├── column_types.hpp       # Column representations (vec, set, bitset) and Column_traits
├── matrix_base.hpp        # MatrixUtil CRTP base: reduction, kernel, cokernel, solve
├── sparse_matrix.hpp      # SparseMatrix<index>: column-sparse 𝔽₂ matrix
├── dense_matrix.hpp       # DenseMatrix: bitset-based dense 𝔽₂ matrix
├── graded_matrix.hpp      # GradedSparseMatrix<D,index,DERIVED>: generic graded matrix
├── r2graded_matrix.hpp    # R2GradedSparseMatrix<index>: ℝ²-graded, kernel via MPfree
├── r3graded_matrix.hpp    # R3GradedSparseMatrix<index>: ℝ³-graded matrices
├── orders_and_graphs.hpp  # Degree_traits, partial orders, Hasse diagrams, sorting
├── graded_linalg.hpp      # Umbrella include
├── homomorphisms.hpp      # Hom-space computation
├── modules.hpp            # Higher-level module operations
├── to_quiver.hpp          # Quiver representation extraction
├── grid_scheduler.hpp     # Grid traversal scheduler for kernel computation
├── draw_hf.hpp            # Hilbert function visualisation helpers
└── bitset_algebra.hpp     # Bitset arithmetic utilities
```

### Key types at a glance

| Type | Description |
|---|---|
| `SparseMatrix<index>` | Ungraded 𝔽₂ matrix, columns as sorted `vec<index>` |
| `GradedSparseMatrix<D, index, DERIVED>` | Matrix graded by degree type `D`; CRTP base for concrete degree specialisations |
| `R2GradedSparseMatrix<index>` | ℝ²-graded matrix; primary working type for 2-parameter persistence |
| `R3GradedSparseMatrix<index>` | ℝ³-graded matrix |
| `R2Sequence<index>` | A two-term chain complex of ℝ²-graded matrices |
| `R2Resolution<index>` | A free resolution `F₂ → F₁ → F₀` |
| `Degree_traits<D>` | Policy struct defining the partial order, lexicographic sort, join/meet, I/O for degree type `D` |
| `Column_traits<COLUMN, index>` | Policy struct for column arithmetic (add, pivot, scalar product, etc.) |

---

## Command-Line Tools

After building, the following executables are available in `build/`:

| Executable | Description |
|---|---|
| `minimize` | Compute a minimal presentation from an input `.scc`/`.firep` file |
| `resolution` | Compute a free resolution (up to second syzygies) |
| `hom` | Compute the homomorphism space between two presentations |
| `shift_endo` | Compute the shift-endomorphism of a presentation |
| `snap_grid` | Snap all degrees to a coarser grid |
| `submodule_at` | Extract the submodule generated at a given degree |
| `pres_to_quiver` | Convert a presentation to a quiver representation |
| `analyse_ind` | Analyse indecomposable summands |
| `thickness` | Compute the thickness invariant |
| `size` | Print the size (number of generators and relations) of a presentation |
| `cut_module` | Cut a module by removing degrees above a threshold |

Each tool reads from a file path passed as the first argument and writes to stdout or an output file. Run any tool with no arguments for usage information.

---

## Usage Examples

### Reading and minimising a presentation

```cpp
#include <grlina/graded_linalg.hpp>
using namespace graded_linalg;

int main() {
    // Read a presentation from an scc file
    R2GradedSparseMatrix<int> M("path/to/presentation.scc");

    std::cout << "Input: " << M.get_num_cols() << " relations, "
              << M.get_num_rows() << " generators\n";

    // Sort columns and rows lexicographically (required before minimisation)
    M.sort_columns_lexicographically();
    M.sort_rows_lexicographically();

    // Minimise in place
    M.minimize();

    std::cout << "Minimal: " << M.get_num_cols() << " relations, "
              << M.get_num_rows() << " generators\n";

    // Write the result
    std::ofstream out("minimal.scc");
    M.to_stream_r2(out);
}
```

### Computing the kernel (graded syzygy module)

```cpp
#include <grlina/graded_linalg.hpp>
using namespace graded_linalg;

int main() {
    R2GradedSparseMatrix<int> M("presentation.scc");
    M.sort_columns_lexicographically();
    M.sort_rows_lexicographically();

    // Kernel gives a presentation of the first syzygy module
    R2GradedSparseMatrix<int> K = M.graded_kernel();

    K.sort_columns_lexicographically();
    K.minimize();

    std::ofstream out("syzygies.scc");
    K.to_stream_r2(out);
}
```

### Building a free resolution

```cpp
#include <grlina/r2graded_matrix.hpp>
using namespace graded_linalg;

int main() {
    R2GradedSparseMatrix<int> d1("presentation.scc");
    d1.sort_columns_lexicographically();
    d1.sort_rows_lexicographically();

    // Construct the resolution F₂ -d2-> F₁ -d1-> F₀
    R2Resolution<int> res(d1);

    // Write the resolution to a file (scc2020 format with 3 levels)
    std::ofstream out("resolution.scc");
    res.to_stream(out);

    // Query the dimension of the presented module at a point
    r2degree alpha = {1.5, 2.0};
    std::cout << "dim at (1.5, 2.0) = " << res.dim_at(alpha) << "\n";
}
```

### Snapping to a coarser grid

```cpp
#include <grlina/r2graded_matrix.hpp>
using namespace graded_linalg;

int main() {
    R2GradedSparseMatrix<int> M("presentation.scc");

    // Snap all degrees to a uniform 10x10 grid within the bounding box
    M.snap_to_equidistant_grid(10);

    std::ofstream out("snapped.scc");
    M.to_stream_r2(out);
}
```

### Extending the library to a new degree type

Persistence-Algebra is designed to be extended. To work with a new poset, specialise `Degree_traits`:

```cpp
#include <grlina/orders_and_graphs.hpp>
using namespace graded_linalg;

struct MyDegree { int x, y, z; };

template<>
struct Degree_traits<MyDegree> {
    static bool smaller_equal(const MyDegree& a, const MyDegree& b) {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }
    static bool equals(const MyDegree& a, const MyDegree& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
    // ... implement smaller, greater, join, meet, lex_lambda, from_stream, position, etc.
};

// Then use GradedSparseMatrix<MyDegree, int, MyDerivedMatrix>
```

---

## License

This library is released under the **GNU Lesser General Public License v3.0** (LGPL-3.0). See [`LICENSE`](LICENSE) for the full text.

Copyright © 2025 Jan Jendrysiak, TU Graz.

You are free to use, modify, and distribute this library, including in proprietary software, provided that any modifications to the library itself are shared under the same licence.
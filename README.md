## License and Attribution

`ccdcpp` is released into the public domain under the Unlicense.

This project is a C++17 translation of the USGS LCMAP PyCCD software.
The original PyCCD implementation was released into the public domain.

The purpose of this project is to:

- preserve PyCCD algorithm behavior
- reproduce PyCCD outputs
- provide a high-performance C++ implementation
- enable Python interoperability through pybind11

During translation, minor changes were required to maintain compatibility
with newer NumPy versions and to adapt Python implementations into C++.

## Independent Development

`ccdcpp` is an independent software project developed separately from the
U.S. Geological Survey (USGS).

This project was created independently by the author for the purposes of
research, software development, performance exploration, and improving the
availability of a native C++ implementation of the CCD algorithm.

The author is not affiliated with, employed by, contracted by, or acting on
behalf of the USGS in the development of this project. The USGS did not
commission, fund, review, endorse, or provide technical support for this
implementation.

The relationship to PyCCD is limited to algorithmic translation and comparison
against the publicly available PyCCD implementation. Any modifications,
optimizations, design decisions, and implementation choices in `ccdcpp` are
the independent work of this project.
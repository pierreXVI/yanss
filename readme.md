# YANNS

<!-- YANNS is a software designed to solve Fluid Dynamics simulations -->

## Dependencies

This software use the following external libraries:
  - An MPI implementation, such as [MPICH](https://www.mpich.org)
  - [PETSc](https://www.mcs.anl.gov/petsc), a suite of data structures and routines for the scalable solution of scientific applications modeled by partial differential equations,
  - [LibYAML](https://github.com/yaml/libyaml), a C library for parsing YAML.

The C compiler must be c99 compliant. This is usually ensured with the compiler option `-std=c99`.

## Installation

To install the software, run the usual `./configure; make`:

    $ cd yanns
    $ ./configure --help  # To view configure options
    $ ./configure
    $ make

To specify external libraries locations, use the configure option `--with-library-dir /path/to/library` to locate the root directory of the library installation. The root directory contains a 'include' folder with the necessary headers, and a 'lib' folder with the libraries. It is usually the directory specified with `--prefix=/root/directory` on the library configuration.

## Using the software

To use the software, run:

    $ ./build/bin/yanns input_file.yaml

where `input_file.yaml` is the YAML input file. An example is given in `data/example.yaml`.

The mesh is specified with the option `-mesh path/to/mesh`.

Any other option can be given in the command line (`yanss -option option_value`) or in the YAML input file:

    Options:
      - -option option_value

Note that the first dash is from the YAML list item notation, and the second is the option prefix.

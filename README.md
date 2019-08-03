SyCLDslash MiniApp 
========================

This git repository contains the SyCLDslash
package. SyCLDslash depends for its testing reference
implementatins on QDP++. To build this code it is
recommended to perform a recursive checkout of the 
repository:

[`github.com;bjoo/SyCLDslashWorkspace.git`][https://github.com/bjoo/SyCLDslashWorkspace,git]

which contains QDP++ and its dependencies

In addition `SyCLDslash` depends on `googletest`
and includes it as a sub-module.

Checking out this repository
============================

It is highly recommended that this library be checked out using
the `--recursive` option to Git so that the `googletest` submodule
is checked out. Alternatively consider checking out [`SyCLDslashWorkspace`][https://github.com/bjoo/SyCLDslashWorkspace,git].



Building 
========

This code builds with CMake. 

Useful CMake options are:
* `-DQDPXX_DIR=<qdp-install-location>/share` - points to the `share` directory of an installation of QDP++ which contains a `FindQDPXX.cmake` file.

* `-DMG_FORTRANLIKE_COMPLEX=ON` enables (RIRIRIRI) storage for complex numbers in 
vectors. Turning it to 'OFF' 

*  `-DMG_USE_NEIGHBOR_TABLE=OFF` (if set to 'ON` would use a neighbour table taht is actualy a lookup 
 table. Currently this is set to OFF as that option is not currently supported, neighbor indices are
explicitly computed.

* `-DMG_USE_LAYOUT_LEFT=OFF` ( if set to `ON` selects left index fastest indexing for View objects, if 
  set to `OFF` uses right fastest indexing, following Kokkos. )


Running The Mini-App:
=====================

Once building is complete the executables will be in `build/build_sycl_dslash/tests`
Two executables are of primary interest: `test_dslash_sycl` and `test_dslash_sycl_vperf`.

The `test_dslash_sycl` app performs some unit testing of the dslash operator for a selection
of vector lenghts.

The performance test is `test_dslash_sycl_vperf` which will first apply a dslash for the 
purposes of JIT-compiling all the components, and then it will apply dslash again to time
a single application. It will use this timing to choose a number of iterations commensurate
to 5 seconds of runtime or minimally 1 iteration if it takes longer than 5 seconds. It will
then perform 5 timing loops. 

Known issues
=============
 - The code has been tested only with the Intel LLVM/SyCL compiler which is in development,
on a system running an OpenCL runtime. It occasionally crashes OpenCL with the error message:

```OpenCL API failed. OpenCL API returns: -34 (CL_INVALID_CONTEXT)```

This issue needs to be chased down still.

To Do Items
===========

* Need to figure out how to run this code with other SyCL Compilers e.g. CodePlay etc

Licensing and copyright
=======================

The Jefferson Lab License under which this workspace is distributed is in the file `LICENSE`
Licenses for dependencies are included in their source code and/or in the `LICENSES` directory.
 



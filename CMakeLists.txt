cmake_minimum_required(VERSION 3.13)
project(SyCLDslash VERSION 0.1
                  DESCRIPTION "SyCL Dslash"
                  LANGUAGES CXX)


# Basic Includes
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)


# Deal with default loglevel
# If code is built in release mode it will be INFO
# Otherwise  it will be build set to DEBUG
# Users can set it of course.
set( MG_DEFAULT_LOGLEVEL "DEBUG" CACHE STRING "Default LogLevel can be INFO, DEBUG, DEBUG2 DEBUG3")
set( MG_VECLEN_SP "4" CACHE STRING "Single Precision Vector Length (default is 4 for AVX2")
option(MG_FORTRANLIKE_COMPLEX "Fortran like complex ordering (r,i,r,i,...r,i) for vectors" OFF)
option(MG_DEBUG_INCLUDES "Print info about complex/vnode file inclusion. Default is OFF" OFF)
option(MG_USE_LAYOUT_LEFT "Use left fastest indexing. Default is ON" ON)
option(MG_USE_NEIGHBOR_TABLE "Use memory based table for neighbor indexing (as opposed to computing on the fly). Default is OFF" OFF)
option(MG_USE_COMPUTE_CPP "Use ComputeCPP" OFF)
option(MG_USE_FIXED_ITERS "Use fixed iteration count in timing loop in test_dslash_sycl_vperf" ON)
if ( MG_USE_COMPUTE_CPP ) 
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
  include(${PROJECT_SOURCE_DIR}/cmake/FindComputeCpp.cmake)
  find_package(ComputeCpp)
endif()

# Default Loglevel 
if ( NOT MG_DEFAULT_LOGLEVEL )
if ( CMAKE_BUILD_TYPE )
  if ( ${CMAKE_BUILD_TYPE} STREQUAL "Release" )
     set( MG_DEFAULT_LOGLEVEL "INFO")
  else()
     set( MG_DEFAULT_LOGLEVEL "DEBUG")
   endif()
else()
  message( WARNING "CMAKE_BUILD_TYPE not set. Setting Default Loglevel to DEBUG" )
  set(MG_DEFAULT_LOGLEVEL, "DEBUG")
endif()

endif(NOT MG_DEFAULT_LOGLEVEL)

# Find OpenMP for backward compatibiliy
#find_package(OpenMP REQUIRED)

# Find QDP++  for tests
list(APPEND CMAKE_MODULE_PATH "${QDPXX_DIR}")
find_package(QDPXX)
if( QDPXX_FOUND ) 
  set( HAVE_QDPXX "1")
  if( QDPXX_PARALLEL_ARCH STREQUAL "parscalar" )
     set(MG_PARSCALAR_ARCH TRUE)
  endif()
endif() 

#IMPORT OpenCL as an imported target
# Look for it in /usr/lib64
# 
#find_library(OpenCLLib NAME "OpenCL" PATHS "/usr/lib64")
#if( OpenCLLib STREQUAL "OpenCLLib-NOTFOUND") 
#  message(FATAL_ERROR "Couldnt find OpenCL in /usr/lib64")
#else()
 set( HAVE_OPENCL "1")
#  add_library(OpenCL UNKNOWN IMPORTED)
# set_target_properties(OpenCL PROPERTIES IMPORTED_LOCATION ${OpenCLLib})
#endif()


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/sycl_dslash_config.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/sycl_dslash_config.h)
  
add_subdirectory(lib)

#Make tests here
add_subdirectory(test)


# Install the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install the generated headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/sycl_dslash_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

# Set some variables to export to the config file if not already set
set(SyCLDslash_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)

# Make the config file
configure_package_config_file(SyCLDslashConfig.cmake.in SyCLDslashConfig.cmake INSTALL_DESTINATION lib/cmake/SyCLDslash PATH_VARS SyCLDslash_INCLUDE_DIR)

# Make the version file
write_basic_package_version_file(
    SyCLDslashVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SyCLDslashVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/SyCLDslashConfig.cmake DESTINATION lib/cmake/SyCLDslash)


#install(EXPORT SyCLDslashTargets NAMESPACE SyCLDslash:: DESTINATION lib/cmake/SyCLDslash )

# Export stuff
# This remains a mystery
export(PACKAGE SyCLDslash)

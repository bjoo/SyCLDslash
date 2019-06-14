cmake_minimum_required(VERSION 3.13)
project(SyCLDslash VERSION 0.1
                  DESCRIPTION "SyCL Dslash"
                  LANGUAGES CXX)


# Basic Includes
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

#Make libs here 
list(APPEND CMAKE_MODULE_PATH "${QDPXX_DIR}")

find_package(QDPXX)

add_subdirectory(lib)

#Make tests here
add_subdirectory(test)


# Install the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install the generated headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/myproj_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

# Set some variables to export to the config file if not already set
set(SyCLDslash_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
message(STATUS "Setting SyCLDslash_INCLUDE_DIR: ${SyCLDslash_INCLUDE_DIR}")

# Make the config file
configure_package_config_file(SyCLDslashConfig.cmake.in SyCLDslashConfig.cmake INSTALL_DESTINATION lib/cmake/SyCLDslash PATH_VARS SyCLDslash_INCLUDE_DIR)

# Make the version file
write_basic_package_version_file(
    SyCLDslashVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SyCLDslashVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/SyCLDslashConfig.cmake DESTINATION lib/cmake/SyCLDslash)


install(EXPORT SyCLDslashTargets NAMESPACE SyCLDslash:: DESTINATION lib/cmake/SyCLDslash )

# Export stuff
# This remains a mystery
export(PACKAGE SyCLDslash)

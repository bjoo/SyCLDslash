
# Boilerplate from Introduction to modern CMake

# Add the Googletest directory and mark its targets etc as advanced
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

# Do this to Keep IDE's happy according to introduction to modern cmake
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)


add_library( testutils qdpxx_utils.h qdpxx_latticeinit.h qdpxx_latticeinit.cpp
    	reunit.cpp dslashm_w.cpp )
    	
target_link_libraries( testutils libmg qdp gtest ) 

add_library( test_env_main test_env.h test_env.cpp )
target_link_libraries( test_env_main  libmg gtest )

# This macro takes the testname and atts an executable from the argumnets
# (ARGN is every argument after the listed one) which allows you to pass a list
# of sources after the testname
macro(package_add_test TESTNAME)
    # Make the executable
    add_executable(${TESTNAME} ${ARGN} )
    if( MG_USE_COMPUTE_CPP )
      target_include_directories(${TESTNAME} PUBLIC ${COMPUTECPP_INCLUDE_DIRECTORY})
      add_sycl_to_target(TARGET ${TESTNAME} SOURCES ${ARGN})
    endif()
    # Add on the gtest libs
    target_link_libraries(${TESTNAME} libmg testutils qdp -lOpenCL test_env_main)

    if( NOT MG_USE_COMPUTE_CPP ) 
    	target_link_libraries(${TESTNAME} -lsycl)
    endif()
    
    # Add the test to CTest
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})

    # I am not 100% sure what tis is for but it is a property
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

# Here TESTNAME is test1
# and ARGN is test1.cpp
package_add_test(test1 test1.cpp)
package_add_test(test_vectype_omp test_vectype_omp.cpp)
package_add_test(test_vectype_sycl  test_vectype_sycl.cpp)
package_add_test(test_vnode_sycl  test_vnode_sycl.cpp)
package_add_test(test_view_sycl  test_view_sycl.cpp)
package_add_test(test_vneighbor_table_sycl  test_vneighbor_table_sycl.cpp)
package_add_test(test_qdp_utils_sycl  test_qdp_utils_sycl.cpp)
package_add_test(test_vspinproj_sycl  test_vspinproj_sycl.cpp)
package_add_test(test_dslash_sycl  test_dslash_sycl.cpp)
package_add_test(test_dslash_sycl_vperf  test_dslash_sycl_vperf.cpp)
#package_add_test(test_vneighbor_table_sycl_broken  test_vneighbor_table_sycl_broken.cpp)

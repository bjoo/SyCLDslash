set( LIB_SRCS ./utils/print_utils.cpp
                     ../include/utils/print_utils.h 
                     ./utils/initialize.cpp 
                     ../include/utils/initialize.h
                     ../include/lattice/lattice_info.h
                     ./lattice/lattice_info.cpp 
                     ../include/lattice/nodeinfo.h
                     ./lattice/nodeinfo.cpp )
                     
if( MG_ARCH_PARSCALAR_ARCH )
   list(APPEND LIB_SRCS ./lattice/nodeinfo_qmp.cpp)
else()
   list(APPEND LIB_SRCS ./lattice/nodeinfo_single.cpp)
endif()
 

add_library( libmg ${LIB_SRCS} ) 
target_link_libraries( libmg  qdp)
target_link_libraries( libmg )

# Add include/ onto the include path
target_include_directories( libmg PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	$<INSTALL_INTERFACE:include>)

target_include_directories( libmg PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<INSTALL_INTERFACE:include>)

set_target_properties( libmg PROPERTIES CXX_STANDARD 11 )
set_target_properties( libmg PROPERTIES CXX_EXTENSIONS OFF )


# Link any dependent libraries
# target_link_libraries( libmg dependent_lib )

# Install the library and the targets 
# into the export for the targets
install(TARGETS libmg EXPORT MyProjectTargets
  ARCHIVE DESTINATION lib
  LIBRARY  DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)



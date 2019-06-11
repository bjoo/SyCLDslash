set( LIB_SRCS ./libsrc.cpp )
add_library( mylib ${LIB_SRCS} ) 

# Add include/ onto the include path
target_include_directories( mylib PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	$<INSTALL_INTERFACE:include>)

target_include_directories( mylib PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<INSTALL_INTERFACE:include>)

set_target_properties( mylib PROPERTIES CXX_STANDARD 11 )
set_target_properties( mylib PROPERTIES CXX_EXTENSIONS OFF )


# Link any dependent libraries
# target_link_libraries( mylib dependent_lib )

# Install the library and the targets 
# into the export for the targets
install(TARGETS mylib EXPORT MyProjectTargets
  ARCHIVE DESTINATION lib
  LIBRARY  DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)



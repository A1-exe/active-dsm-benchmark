#------------------------------------------------------------------------------
# Include Header Directories
#------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# Build Database (DB) Library
#------------------------------------------------------------------------------
# add_library(database_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/database_lib.cc)
# target_link_libraries(database_lib
#         ${Boost_FILESYSTEM_LIBRARY}
#         ${Boost_SYSTEM_LIBRARY})

#------------------------------------------------------------------------------
# Build Benchmark
#------------------------------------------------------------------------------
add_executable(benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc)
target_link_libraries(benchmark ${HermesShm_LIBRARIES})
# add_dependencies(grocery_db database_lib)
# target_link_libraries(grocery_db database_lib)

#------------------------------------------------------------------------------
# Add libraries + executables to CMake install
#------------------------------------------------------------------------------
# install(
#         TARGETS
#         database_lib
#         grocery_db
#         movies_db
#         LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#         RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#-----------------------------------------------------------------------------
# Add header file(s) to CMake Install
#-----------------------------------------------------------------------------
# install(
#         FILES
#         ${CMAKE_SOURCE_DIR}/include/database_lib.h
#         DESTINATION
#         ${CMAKE_INSTALL_PREFIX}/include
#         COMPONENT
#         headers)
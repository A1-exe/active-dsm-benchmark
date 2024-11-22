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
add_executable(benchmark_int ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_int.cc)
target_link_libraries(benchmark_int ${HermesShm_LIBRARIES})

add_executable(benchmark_char ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_char.cc)
target_link_libraries(benchmark_char ${HermesShm_LIBRARIES})

add_executable(benchmark_double ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_double.cc)
target_link_libraries(benchmark_double ${HermesShm_LIBRARIES})

add_executable(benchmark_size_t ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_size_t.cc)
target_link_libraries(benchmark_size_t ${HermesShm_LIBRARIES})

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
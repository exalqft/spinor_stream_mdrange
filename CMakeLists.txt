cmake_minimum_required(VERSION 3.21.0)
project(spinor-stream-mdrange VERSION 1.0.0 LANGUAGES CXX)

# Honor already set CMAKE_CXX_STANDARD.
# Otherwise default to C++17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

#
# Kokkos
#
# General Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

add_executable(spinor-stream-mdrange-SC-array-external spinor-stream-mdrange-SC-array-external.cpp)
target_link_libraries(spinor-stream-mdrange-SC-array-external Kokkos::kokkos)

add_executable(spinor-stream-mdrange-SC-array-internal spinor-stream-mdrange-SC-array-internal.cpp)
target_link_libraries(spinor-stream-mdrange-SC-array-internal Kokkos::kokkos)

add_executable(spinor-stream-mdrange-SC-static spinor-stream-mdrange-SC-static.cpp)
target_link_libraries(spinor-stream-mdrange-SC-static Kokkos::kokkos)

add_executable(spinor-stream-mdrange-SC-static-internal spinor-stream-mdrange-SC-static-internal.cpp)
target_link_libraries(spinor-stream-mdrange-SC-static-internal Kokkos::kokkos)


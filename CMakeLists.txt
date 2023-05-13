cmake_minimum_required(VERSION 3.0)
project(sgpt VERSION 0.0.1 LANGUAGES C)

enable_testing()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)

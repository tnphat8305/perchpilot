cmake_minimum_required(VERSION 3.16)
project(HoldPad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-msgs10 REQUIRED)

add_library(HoldPad SHARED HoldPad.cc)
target_link_libraries(HoldPad PRIVATE gz-sim8::gz-sim8 gz-plugin2::register gz-msgs10::gz-msgs10)

add_executable(hold_margin_test test/hold_margin_test.cc)
add_executable(footprint_test test/footprint_test.cc)
add_executable(load_cell_test test/load_cell_test.cc)
add_executable(slip_sensor_test test/slip_sensor_test.cc)
target_include_directories(slip_sensor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(load_cell_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(footprint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hold_margin_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

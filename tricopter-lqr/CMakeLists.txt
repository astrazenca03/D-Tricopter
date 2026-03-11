cmake_minimum_required(VERSION 3.14)
project(tricopter_lqr VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies via FetchContent ---
include(FetchContent)

# Eigen3 — header-only linear algebra library
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
# Eigen's CMakeLists can be slow; just fetch, don't configure
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
)

# yaml-cpp — YAML parser
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# --- Core library (shared between main and tests) ---
add_library(tricopter_core STATIC
    src/config.cpp
    src/dynamics.cpp
    src/control_allocation.cpp
    src/lqr.cpp
    src/integrator.cpp
    src/simulation.cpp
)
target_include_directories(tricopter_core PUBLIC include)
target_link_libraries(tricopter_core PUBLIC Eigen3::Eigen yaml-cpp::yaml-cpp)

# --- Main simulation executable ---
add_executable(tricopter_sim src/main.cpp)
target_link_libraries(tricopter_sim PRIVATE tricopter_core)

# --- Unit tests executable ---
add_executable(tricopter_tests tests/test_main.cpp)
target_link_libraries(tricopter_tests PRIVATE tricopter_core)

# Copy config file to build directory for convenience
configure_file(
    ${CMAKE_SOURCE_DIR}/config/tricopter_default.yaml
    ${CMAKE_BINARY_DIR}/config/tricopter_default.yaml
    COPYONLY
)

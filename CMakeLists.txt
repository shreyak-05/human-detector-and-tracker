cmake_minimum_required(VERSION 3.14)
project(HumanDetectorTracker VERSION 1.0)

#
# GoogleTest Setup
# ref: https://google.github.io/googletest/quickstart-cmake.html
#

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV dependency
find_package(OpenCV 4 REQUIRED)

# Set up ONNX Runtime
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so")

# Check if ONNX Runtime exists
if(EXISTS ${ONNXRUNTIME_LIB})
  message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
  message(STATUS "ONNX Runtime includes: ${ONNXRUNTIME_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIB}

Please install ONNX Runtime:
1. Download: wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
2. Extract: tar -xzf onnxruntime-linux-x64-1.16.3.tgz
3. Rename: mv onnxruntime-linux-x64-1.16.3 onnxruntime

See README.md for detailed installation instructions.")
endif()

# Make the include directory and library available
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# declares a dependency on GoogleTest which is downloaded from GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enables testing for this directory and below
enable_testing()
include(GoogleTest)

#
# Doxygen Support check if Doxygen is installed
# ref: https://cmake.org/cmake/help/latest/module/FindDoxygen.html
#
find_package(Doxygen)
if ( DOXYGEN_FOUND )
  set( DOXYGEN_OUTPUT_DIRECTORY     ${PROJECT_SOURCE_DIR}/docs )
  set( DOXYGEN_COLLABORATION_GRAPH  YES )
  set( DOXYGEN_EXTRACT_ALL          YES )
  set( DOXYGEN_CLASS_DIAGRAMS       YES )
  set( DOXYGEN_HIDE_UNDOC_RELATIONS NO )
  set( DOXYGEN_HAVE_DOT             YES )
  set( DOXYGEN_CLASS_GRAPH          YES )
  set( DOXYGEN_CALL_GRAPH           YES )
  set( DOXYGEN_CALLER_GRAPH         YES )
  set( DOXYGEN_COLLABORATION_GRAPH  YES )
  set( DOXYGEN_BUILTIN_STL_SUPPORT  YES )
  set( DOXYGEN_EXTRACT_PRIVATE      YES )
  set( DOXYGEN_EXTRACT_PACKAGE      YES )
  set( DOXYGEN_EXTRACT_STATIC       YES )
  set( DOXYGEN_EXTRACT_LOCALMETHODS YES )
  set( DOXYGEN_UML_LOOK             YES )
  set( DOXYGEN_UML_LIMIT_NUM_FIELDS 50 )
  set( DOXYGEN_TEMPLATE_RELATIONS   YES )
  set( DOXYGEN_DOT_GRAPH_MAX_NODES  100 )
  set( DOXYGEN_MAX_DOT_GRAPH_DEPTH  0 )
  set( DOXYGEN_DOT_TRANSPARENT      YES )
else()
  message( FATAL_ERROR "Doxygen needs to be installed to generate the doxygen documentation" )
endif()

#
# Create the compilation database for clangd
#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Enable compile_commands.json


#
# Set up code coverage generation.
# Must compile with debug, e.g,
#   cmake -S ./ -B build/ -D CMAKE_BUILD_TYPE=Debug
#
option(WANT_COVERAGE "this option enable coverage" ON)
if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
  set(WANT_COVERAGE OFF)
endif()

if(WANT_COVERAGE)
  message("Enabling coverage")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
  include(CodeCoverage)
  # The following line come from cmake-modules/CodeCoverage.cmake
  append_coverage_compiler_flags() # will add "--coverage" compilation flag
  # Exclude certain directories from coverage analysis
  set(COVERAGE_EXCLUDES
    ${PROJECT_BINARY_DIR}
    )
endif()

#
# c++ Boilerplate Modification Starts Here
# ref: https://iamsorush.com/posts/cpp-cmake-essential/
# ref: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
#
add_subdirectory(libs)
add_subdirectory(app)
add_subdirectory(test)

# Ensure results directory exists
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

# Create a target to build documentation - runs automatically
if(DOXYGEN_FOUND)
  doxygen_add_docs(docs           # target name
    # List of files or directories
    ${PROJECT_SOURCE_DIR}/app
    ${PROJECT_SOURCE_DIR}/libs
    ${PROJECT_SOURCE_DIR}/include
    COMMENT "Generating Doxygen documentation"
  )
  # Make docs depend on detector_lib being built first
  add_dependencies(docs detector_lib)
else()
  add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E echo "Doxygen not available - skipping documentation"
  )
endif()

# Add cppcheck custom target - runs automatically, doesn't fail build
find_program(CPPCHECK_EXECUTABLE cppcheck)
if(CPPCHECK_EXECUTABLE)
  add_custom_target(cpp-check
    COMMAND ${CPPCHECK_EXECUTABLE}
      --enable=all
      --inline-suppr
      --error-exitcode=0
      --suppress=missingIncludeSystem
      --suppress=preprocessorErrorDirective
      --suppress=unusedFunction
      -I ${PROJECT_SOURCE_DIR}/include
      -I ${PROJECT_SOURCE_DIR}/libs
      --std=c++17
      --language=c++
      --project=${CMAKE_BINARY_DIR}/compile_commands.json
      -i ${CMAKE_BINARY_DIR}/_deps
      -i ${CMAKE_BINARY_DIR}
      > ${CMAKE_SOURCE_DIR}/results/cppcheck.txt 2>&1 || true
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running cppcheck static analysis (output saved to results/cppcheck.txt)"
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
  )
else()
  add_custom_target(cpp-check
    COMMAND ${CMAKE_COMMAND} -E echo "cppcheck not available - skipping static analysis"
  )
endif()

# Create meta-target that runs both docs and cppcheck automatically
add_custom_target(auto-docs-check
  DEPENDS docs cpp-check
  COMMENT "Running documentation and static analysis"
)
add_dependencies(auto-docs-check detector_lib)

# Make auto-docs-check run automatically with every build
add_custom_target(all-docs-check ALL
  DEPENDS auto-docs-check
)

# create targets for building code coverage reports
if(WANT_COVERAGE)
  # unit test coverage:
  setup_target_for_coverage_lcov ( # use lcov to generate coverage report
    NAME test_coverage          # target name
    EXECUTABLE run_tests || true    # what to run insdie the build directory?
                                 #   '|| true' --> continue coverage even if ctest failed
    # EXECUTABLE ctest     # what to run insdie the build directory?
    EXCLUDE
      "*gtest*"          # Don't analyze googleTest code
      "/usr/include/*"   # Don't analyze system headers
    )

  # code coverage for the app (shell-app):
  # Use gcovr insetad of lcov if you want to see branch coverage as well
  setup_target_for_coverage_gcovr_html ( # use gcovr to generate coverage report
    NAME app_coverage          # target name
    EXECUTABLE demo       # what to run insdie the build directory?
    EXCLUDE
    "test/test.cpp"    # App does not run unit test, so don't analyze it
    "test/main.cpp"    # App does not run unit test, so don't analyze it
    )

endif()

# Sanity check:
# can also do "cmake -S ./ -B build/ -LAH" to print all variables
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "WANT_COVERAGE    = ${WANT_COVERAGE}")

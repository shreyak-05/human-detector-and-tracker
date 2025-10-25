# Create the test executable
add_executable(run_tests
  test_depth.cpp
  test_detector_tracker.cpp
  test_preprocessor.cpp
  test_transformer.cpp
)

# Link libraries
target_link_libraries(run_tests PRIVATE
  detector_lib
  gtest_main
  ${OpenCV_LIBS}
)

# Enable CMake's test runner to discover the tests
gtest_discover_tests(run_tests)


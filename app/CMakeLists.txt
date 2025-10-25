# Create the demo executable
add_executable(demo main.cpp)

# Link libraries
target_link_libraries(demo PRIVATE
  detector_lib
  ${OpenCV_LIBS}
)

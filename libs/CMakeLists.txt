# Create the main detector library
add_library(detector_lib)

# Add all source files to the library
target_sources(detector_lib PRIVATE
  detector_tracker/detector_tracker.cpp
  ml_depth_estimator/ml_depth_estimator.cpp
  preprocessor/preprocessor.cpp
  transformer/transformer.cpp
  onnx_network/onnx_network.cpp
)

target_link_libraries(detector_lib PUBLIC
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so.1.16.3
)
target_include_directories(detector_lib PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIRS}
)


# Optional: enable GPU inference
# add_definitions(-DUSE_CUDA)
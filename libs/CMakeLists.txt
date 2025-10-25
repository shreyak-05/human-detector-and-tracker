# Create the main detector library
add_library(detector_lib)

# Add all source files to the library
target_sources(detector_lib PRIVATE
    detector_tracker/detector_tracker.cpp
    ml_depth_estimator/ml_depth_estimator.cpp
    preprocessor/preprocessor.cpp
    transformer/transformer.cpp
)

# Set include directories
target_include_directories(detector_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Link OpenCV
target_link_libraries(detector_lib PUBLIC ${OpenCV_LIBS})



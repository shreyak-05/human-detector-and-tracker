find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# ---------- Preprocessor ----------
add_library(preprocessor preprocessor/preprocessor.cpp)
target_include_directories(preprocessor PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(preprocessor PUBLIC ${OpenCV_LIBS})

# ---------- Depth Estimator ----------
add_library(ml_depth_estimator ml_depth_estimator/ml_depth_estimator.cpp)
target_include_directories(ml_depth_estimator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ml_depth_estimator PUBLIC ${OpenCV_LIBS})

# ---------- Transformer ----------
add_library(transformer transformer/transformer.cpp)
target_include_directories(transformer PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(transformer PUBLIC ${OpenCV_LIBS})

# ---------- Detector Tracker ----------
add_library(detector_tracker detector_tracker/detector_tracker.cpp)
target_include_directories(detector_tracker PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(detector_tracker PUBLIC preprocessor ${OpenCV_LIBS})


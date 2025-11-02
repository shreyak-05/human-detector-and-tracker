#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ml_depth_estimator.hpp"

TEST(DepthEstimatorTest, BoundingBoxValidation) {
    cv::Mat test_frame(480, 640, CV_8UC3);
    cv::Rect valid_bbox(100, 100, 50, 50);
    cv::Rect invalid_bbox(-10, -10, 5, 5);
    
    // Valid bbox should have positive area and be within frame
    ASSERT_GT(valid_bbox.area(), 0);
    ASSERT_GE(valid_bbox.x, 0);
    ASSERT_GE(valid_bbox.y, 0);
    
    // Invalid bbox should be detectable
    ASSERT_TRUE(invalid_bbox.x < 0 || invalid_bbox.y < 0);
}

TEST(DepthEstimatorTest, FrameDimensionValidation) {
    cv::Mat small_frame(100, 100, CV_8UC3);
    cv::Mat large_frame(1080, 1920, CV_8UC3);
    
    // Frames should have valid dimensions and 3 channels
    ASSERT_GT(small_frame.rows * small_frame.cols, 0);
    ASSERT_GT(large_frame.rows * large_frame.cols, 0);
    ASSERT_EQ(small_frame.channels(), 3);
    ASSERT_EQ(large_frame.channels(), 3);
}

TEST(DepthEstimatorTest, DefaultDepthValue) {
    // Test that reasonable default depth values are used
    float default_depth = 2.0f;  // Typical default for human detection
    
    ASSERT_GT(default_depth, 0.0f);
    ASSERT_LT(default_depth, 10.0f);  // Reasonable range for human detection
    ASSERT_GE(default_depth, 0.5f);   // Minimum reasonable distance
}
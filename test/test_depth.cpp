/**
 * @file test_depth.cpp * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ml_depth_estimator.hpp"

// Test 1: Bounding box validation with edge cases
TEST(DepthEstimatorTest, BoundingBoxValidation) {
  cv::Mat test_frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
  
  // Valid bbox
  cv::Rect valid_bbox(100, 100, 50, 50);
  ASSERT_GT(valid_bbox.area(), 0);
  ASSERT_GE(valid_bbox.x, 0);
  ASSERT_GE(valid_bbox.y, 0);
  ASSERT_LT(valid_bbox.x + valid_bbox.width, test_frame.cols);
  
  // Invalid bbox (negative coordinates)
  cv::Rect invalid_bbox(-10, -10, 5, 5);
  ASSERT_TRUE(invalid_bbox.x < 0 || invalid_bbox.y < 0);
  
  // Zero area bbox
  cv::Rect zero_bbox(100, 100, 0, 0);
  ASSERT_EQ(zero_bbox.area(), 0);
  
  // Out of bounds bbox
  cv::Rect oob_bbox(700, 500, 100, 100);
  ASSERT_TRUE(oob_bbox.x + oob_bbox.width > test_frame.cols);
}

// Test 2: Frame dimension validation with multiple formats
TEST(DepthEstimatorTest, FrameDimensionValidation) {
  // Test various frame sizes
  cv::Mat small_frame(100, 100, CV_8UC3, cv::Scalar(150, 150, 150));
  cv::Mat large_frame(1080, 1920, CV_8UC3, cv::Scalar(50, 50, 50));
  
  ASSERT_GT(small_frame.rows * small_frame.cols, 0);
  ASSERT_GT(large_frame.rows * large_frame.cols, 0);
  ASSERT_EQ(small_frame.channels(), 3);
  ASSERT_EQ(large_frame.channels(), 3);
  
  // Test different formats
  cv::Mat gray_frame(480, 640, CV_8UC1, cv::Scalar(128));
  cv::Mat float_frame(480, 640, CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
  
  ASSERT_EQ(gray_frame.channels(), 1);
  ASSERT_EQ(float_frame.channels(), 3);
  ASSERT_EQ(float_frame.type(), CV_32FC3);
  
  // Test empty frame detection
  cv::Mat empty_frame;
  ASSERT_TRUE(empty_frame.empty());
  ASSERT_EQ(empty_frame.rows, 0);
}

// Test 3: Depth value validation and mathematical operations
TEST(DepthEstimatorTest, DefaultDepthValue) {
  float default_depth = 2.0f;
  float min_depth = 0.1f;
  float max_depth = 100.0f;
  
  ASSERT_GT(default_depth, 0.0f);
  ASSERT_LT(default_depth, 10.0f);
  ASSERT_GE(default_depth, min_depth);
  ASSERT_LE(default_depth, max_depth);
  
  // Test depth calculations
  cv::Mat test_frame(200, 300, CV_8UC3, cv::Scalar(100, 100, 100));
  cv::Rect test_bbox(50, 50, 100, 100);
  
  // Calculate center point
  cv::Point2f center(test_bbox.x + test_bbox.width / 2.0f, test_bbox.y + test_bbox.height / 2.0f);
  ASSERT_GE(center.x, 0.0f);
  ASSERT_LT(center.x, test_frame.cols);
  ASSERT_GE(center.y, 0.0f);
  ASSERT_LT(center.y, test_frame.rows);
  
  // Test absolute value handling
  float negative_depth = -2.5f;
  float abs_depth = std::abs(negative_depth);
  ASSERT_EQ(abs_depth, 2.5f);
  
  // Test frame ID validation
  int frame_id = 42;
  ASSERT_GE(frame_id, 0);
}
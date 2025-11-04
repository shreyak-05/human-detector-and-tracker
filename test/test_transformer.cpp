/**
 * @file test_transformer.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "transformer.hpp"

class TransformerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
        500.0, 0, 320.0, 
        0, 500.0, 240.0, 
        0, 0, 1.0);
    transformer_ = std::make_unique<perception::Transformer3D>(camera_matrix);
  }

  std::unique_ptr<perception::Transformer3D> transformer_;
};

// Test 1: Basic projection math and center pixel handling
TEST_F(TransformerTest, BasicProjection) {
  // Test center pixel (should give ~0,0,depth)
  cv::Point2f center(320.0f, 240.0f);
  cv::Point3f result = transformer_->project_to_3d(center, 2.0f);
  EXPECT_NEAR(result.x, 0.0f, 0.01f);
  EXPECT_NEAR(result.y, 0.0f, 0.01f);
  EXPECT_NEAR(result.z, 2.0f, 0.01f);
  
  // Test off-center pixel
  cv::Point2f offset(420.0f, 340.0f);
  cv::Point3f offset_result = transformer_->project_to_3d(offset, 1.0f);
  EXPECT_GT(offset_result.x, 0.0f);
  EXPECT_GT(offset_result.y, 0.0f);
  EXPECT_NEAR(offset_result.z, 1.0f, 0.01f);
}

// Test 2: Depth scaling verification
TEST_F(TransformerTest, CenterProjection) {
  cv::Point2f pixel(400.0f, 300.0f);
  cv::Point3f result1 = transformer_->project_to_3d(pixel, 1.0f);
  cv::Point3f result2 = transformer_->project_to_3d(pixel, 3.0f);
  
  // Coordinates should scale proportionally with depth
  EXPECT_NEAR(result2.x, result1.x * 3.0f, 0.01f);
  EXPECT_NEAR(result2.y, result1.y * 3.0f, 0.01f);
  EXPECT_NEAR(result2.z, result1.z * 3.0f, 0.01f);
}

// Test 3: Edge cases and different camera parameters
TEST_F(TransformerTest, DepthScaling) {
  // Test zero depth
  cv::Point3f zero_result = transformer_->project_to_3d(cv::Point2f(100, 100), 0.0f);
  EXPECT_NEAR(zero_result.z, 0.0f, 0.001f);
  
  // Test negative coordinates (top-left corner)
  cv::Point3f neg_result = transformer_->project_to_3d(cv::Point2f(0, 0), 1.0f);
  EXPECT_LT(neg_result.x, 0.0f);
  EXPECT_LT(neg_result.y, 0.0f);
  
  // Test different camera matrix constructor path
  cv::Mat diff_matrix = (cv::Mat_<double>(3, 3) << 800.0, 0, 640.0, 0, 800.0, 480.0, 0, 0, 1.0);
  auto transformer2 = std::make_unique<perception::Transformer3D>(diff_matrix);
  cv::Point3f diff_result = transformer2->project_to_3d(cv::Point2f(320, 240), 1.0f);
  EXPECT_NE(diff_result.x, zero_result.x);  // Different camera should give different results
}
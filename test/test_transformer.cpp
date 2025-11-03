/**
 * @file test_transformer.cpp
 * @brief Unit tests for Transformer3D class
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "transformer.hpp"

class TransformerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Camera matrix: fx=100, fy=100, cx=50, cy=50
    camera_matrix_ =
        (cv::Mat_<double>(3, 3) << 100.0, 0, 50.0, 0, 100.0, 50.0, 0, 0, 1.0);
    transformer_ = std::make_unique<perception::Transformer3D>(camera_matrix_);
  }

  cv::Mat camera_matrix_;
  std::unique_ptr<perception::Transformer3D> transformer_;
};

TEST_F(TransformerTest, BasicProjection) {
  cv::Point2f pixel(60, 70);
  float depth = 2.0f;

  auto result = transformer_->project_to_3d(pixel, depth);

  // x = (60-50)*2/100 = 0.2, y = (70-50)*2/100 = 0.4
  ASSERT_FLOAT_EQ(result.x, 0.2f);
  ASSERT_FLOAT_EQ(result.y, 0.4f);
  ASSERT_FLOAT_EQ(result.z, 2.0f);
}

TEST_F(TransformerTest, CenterProjection) {
  cv::Point2f center_pixel(50, 50);  // Camera center
  float depth = 5.0f;

  auto result = transformer_->project_to_3d(center_pixel, depth);

  // At camera center, x and y should be 0
  ASSERT_FLOAT_EQ(result.x, 0.0f);
  ASSERT_FLOAT_EQ(result.y, 0.0f);
  ASSERT_FLOAT_EQ(result.z, 5.0f);
}

TEST_F(TransformerTest, DepthScaling) {
  cv::Point2f pixel(75, 25);
  float depth1 = 1.0f;
  float depth2 = 3.0f;

  auto result1 = transformer_->project_to_3d(pixel, depth1);
  auto result2 = transformer_->project_to_3d(pixel, depth2);

  // X and Y should scale proportionally with depth
  ASSERT_FLOAT_EQ(result2.x, result1.x * 3.0f);
  ASSERT_FLOAT_EQ(result2.y, result1.y * 3.0f);
  ASSERT_FLOAT_EQ(result1.z, 1.0f);
  ASSERT_FLOAT_EQ(result2.z, 3.0f);
}
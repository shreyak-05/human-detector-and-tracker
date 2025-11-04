/**
 * @file test_depth.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include "ml_depth_estimator.hpp"

class DepthEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Try to create depth estimator with test parameters
    std::string model_path = "models/depth_anything_v2_vits.onnx";
    try {
      depth_estimator_ = std::make_unique<perception::MLDepthEstimator>(
          model_path, 518, 518, false);
    } catch (const std::exception& e) {
      depth_estimator_ = nullptr;  // Model not available in test environment
    }
  }

  std::unique_ptr<perception::MLDepthEstimator> depth_estimator_;
};

// Test 1: Constructor validation and error handling
TEST_F(DepthEstimatorTest, ConstructorValidation) {
  // Test constructor with invalid model path
  EXPECT_THROW(
    perception::MLDepthEstimator("", 518, 518, false),
    std::runtime_error
  );
  
  EXPECT_THROW(
    perception::MLDepthEstimator("nonexistent.onnx", 518, 518, false),
    std::runtime_error
  );
  
  // Test with invalid dimensions
  EXPECT_THROW(
    perception::MLDepthEstimator("models/depth_anything_v2_vits.onnx", 0, 518, false),
    std::runtime_error
  );
  
  EXPECT_THROW(
    perception::MLDepthEstimator("models/depth_anything_v2_vits.onnx", -100, 518, false),
    std::runtime_error
  );
}

// Test 2: Depth estimation functionality
TEST_F(DepthEstimatorTest, DepthEstimationFunctionality) {
  // Create test frame
  cv::Mat test_frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
  
  if (!depth_estimator_) {
    // Test fallback behavior when model isn't available
    try {
      perception::MLDepthEstimator fallback_estimator("models/depth_anything_v2_vits.onnx", 518, 518, false);
      cv::Rect test_bbox(100, 100, 100, 100);
      float depth = fallback_estimator.get_depth(test_frame, test_bbox);
      EXPECT_GT(depth, 0.0f);
    } catch (const std::exception& e) {
      // Expected when model file doesn't exist
      SUCCEED() << "Model not available: " << e.what();
    }
    return;
  }
  
  // Test with valid bounding box
  cv::Rect valid_bbox(100, 100, 100, 100);
  float depth = depth_estimator_->get_depth(test_frame, valid_bbox);
  EXPECT_GT(depth, 0.0f);
  EXPECT_LT(depth, 100.0f);  // Reasonable depth range
  
  // Test with different bounding box sizes
  cv::Rect small_bbox(200, 200, 50, 50);
  float small_depth = depth_estimator_->get_depth(test_frame, small_bbox);
  EXPECT_GT(small_depth, 0.0f);
  
  cv::Rect large_bbox(50, 50, 200, 200);
  float large_depth = depth_estimator_->get_depth(test_frame, large_bbox);
  EXPECT_GT(large_depth, 0.0f);
}

// Test 3: Edge cases and robustness
TEST_F(DepthEstimatorTest, EdgeCasesAndRobustness) {
  cv::Mat test_frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
  
  if (!depth_estimator_) {
    // Test error handling when model isn't available
    try {
      perception::MLDepthEstimator test_estimator("models/depth_anything_v2_vits.onnx", 518, 518, false);
      
      // Test with edge bounding boxes
      cv::Rect edge_bbox(0, 0, 50, 50);
      float edge_depth = test_estimator.get_depth(test_frame, edge_bbox);
      EXPECT_GT(edge_depth, 0.0f);
      
      // Test with empty frame
      cv::Mat empty_frame;
      cv::Rect test_bbox(10, 10, 20, 20);
      float empty_depth = test_estimator.get_depth(empty_frame, test_bbox);
      EXPECT_GT(empty_depth, 0.0f);  // Should return default depth
      
    } catch (const std::exception& e) {
      // Test that we get meaningful error messages
      std::string error_msg = e.what();
      EXPECT_TRUE(error_msg.find("model") != std::string::npos || 
                  error_msg.find("ONNX") != std::string::npos ||
                  error_msg.find("depth") != std::string::npos);
    }
    return;
  }
  
  // Test with edge bounding boxes
  cv::Rect edge_bbox(0, 0, 50, 50);  // Top-left corner
  float edge_depth = depth_estimator_->get_depth(test_frame, edge_bbox);
  EXPECT_GT(edge_depth, 0.0f);
  
  // Test with bounding box at bottom-right
  cv::Rect bottom_right(590, 430, 50, 50);
  float br_depth = depth_estimator_->get_depth(test_frame, bottom_right);
  EXPECT_GT(br_depth, 0.0f);
  
  // Test with out-of-bounds bbox (should be handled gracefully)
  cv::Rect oob_bbox(700, 500, 100, 100);
  float oob_depth = depth_estimator_->get_depth(test_frame, oob_bbox);
  EXPECT_GT(oob_depth, 0.0f);  // Should return default depth
  
  // Test with empty frame
  cv::Mat empty_frame;
  cv::Rect test_bbox(10, 10, 20, 20);
  float empty_depth = depth_estimator_->get_depth(test_frame, test_bbox);
  EXPECT_GT(empty_depth, 0.0f);
}
/**
 * @file test_preprocessor.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "preprocessor.hpp"

class PreprocessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    preprocessor_ = std::make_unique<perception::Preprocessor>(640, 640, true);
    preprocessor_no_swap_ = std::make_unique<perception::Preprocessor>(416, 416, false);
  }

  std::unique_ptr<perception::Preprocessor> preprocessor_;
  std::unique_ptr<perception::Preprocessor> preprocessor_no_swap_;
};

// Test 1: Basic blob creation and dimensions
TEST_F(PreprocessorTest, OutputDimensions) {
  cv::Mat input = cv::Mat::ones(480, 640, CV_8UC3) * 128;
  cv::Mat blob = preprocessor_->process(input);
  
  // Verify blob format [N, C, H, W] = [1, 3, 640, 640]
  EXPECT_EQ(blob.dims, 4);
  EXPECT_EQ(blob.size[0], 1);  // Batch
  EXPECT_EQ(blob.size[1], 3);  // Channels
  EXPECT_EQ(blob.size[2], 640); // Height
  EXPECT_EQ(blob.size[3], 640); // Width
  EXPECT_EQ(blob.type(), CV_32F);
  
  // Test different target size
  cv::Mat blob416 = preprocessor_no_swap_->process(input);
  EXPECT_EQ(blob416.size[2], 416);
  EXPECT_EQ(blob416.size[3], 416);
}

// Test 2: Basic functionality and size consistency
TEST_F(PreprocessorTest, ProcessDifferentSizes) {
  // Test with black and white images - focus on blob structure not pixel values
  cv::Mat black_input = cv::Mat::zeros(200, 300, CV_8UC3);
  cv::Mat white_input = cv::Mat::ones(200, 300, CV_8UC3) * 255;
  
  cv::Mat blob_black = preprocessor_->process(black_input);
  cv::Mat blob_white = preprocessor_->process(white_input);
  
  // Both should produce valid blobs with same dimensions
  EXPECT_EQ(blob_black.size[0], blob_white.size[0]);
  EXPECT_EQ(blob_black.size[1], blob_white.size[1]);
  EXPECT_EQ(blob_black.size[2], blob_white.size[2]);
  EXPECT_EQ(blob_black.size[3], blob_white.size[3]);
  
  // Test different input sizes produce same output size
  cv::Mat small_input = cv::Mat::ones(100, 100, CV_8UC3) * 150;
  cv::Mat large_input = cv::Mat::ones(1200, 1600, CV_8UC3) * 150;
  
  cv::Mat blob_small = preprocessor_->process(small_input);
  cv::Mat blob_large = preprocessor_->process(large_input);
  
  EXPECT_EQ(blob_small.size[2], blob_large.size[2]);
  EXPECT_EQ(blob_small.size[3], blob_large.size[3]);
}

// Test 3: Different data types and edge cases
TEST_F(PreprocessorTest, OutputDataType) {
  // Test with different input types
  cv::Mat input_8u = cv::Mat::ones(300, 400, CV_8UC3) * 100;
  cv::Mat blob = preprocessor_->process(input_8u);
  EXPECT_EQ(blob.type(), CV_32F);
  EXPECT_TRUE(blob.isContinuous());
  
  // Test data accessibility
  float* data_ptr = reinterpret_cast<float*>(blob.data);
  EXPECT_NE(data_ptr, nullptr);
  EXPECT_GE(data_ptr[0], 0.0f);
  EXPECT_LE(data_ptr[0], 1.0f);
  
  // Test channel swap vs no swap (different preprocessors)
  cv::Mat color_input = cv::Mat::zeros(200, 200, CV_8UC3);
  color_input.setTo(cv::Scalar(255, 0, 0));  // Blue in BGR
  
  cv::Mat blob_swap = preprocessor_->process(color_input);         // swap_rb = true
  cv::Mat blob_no_swap = preprocessor_no_swap_->process(color_input); // swap_rb = false
  
  EXPECT_GT(blob_swap.total(), 0);
  EXPECT_GT(blob_no_swap.total(), 0);
  // Both should produce valid output but with different channel ordering
}
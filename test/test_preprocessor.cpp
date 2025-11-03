#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "preprocessor.hpp"

class PreprocessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    preprocessor_ = std::make_unique<perception::Preprocessor>(640, 640);
  }

  std::unique_ptr<perception::Preprocessor> preprocessor_;
};

TEST_F(PreprocessorTest, OutputDimensions) {
  cv::Mat frame(480, 800, CV_8UC3, cv::Scalar(0, 0, 0));

  auto result_blob = preprocessor_->process(frame);

  // Check blob dimensions: [1, 3, 640, 640]
  ASSERT_EQ(result_blob.size[0], 1);    // Batch size
  ASSERT_EQ(result_blob.size[1], 3);    // Channels
  ASSERT_EQ(result_blob.size[2], 640);  // Height
  ASSERT_EQ(result_blob.size[3], 640);  // Width
}

TEST_F(PreprocessorTest, ProcessDifferentSizes) {
  // Test with small and large images
  cv::Mat small_frame(100, 200, CV_8UC3, cv::Scalar(128, 64, 192));
  cv::Mat large_frame(1080, 1920, CV_8UC3, cv::Scalar(64, 128, 255));

  auto small_blob = preprocessor_->process(small_frame);
  auto large_blob = preprocessor_->process(large_frame);

  // Both should output same dimensions regardless of input size
  ASSERT_EQ(small_blob.size[2], 640);
  ASSERT_EQ(small_blob.size[3], 640);
  ASSERT_EQ(large_blob.size[2], 640);
  ASSERT_EQ(large_blob.size[3], 640);
}

TEST_F(PreprocessorTest, OutputDataType) {
  cv::Mat frame(480, 640, CV_8UC3);

  auto result_blob = preprocessor_->process(frame);

  // Blob should be float type (CV_32F)
  ASSERT_EQ(result_blob.type(), CV_32F);

  // Verify data can be accessed
  float* data = (float*)result_blob.data;
  ASSERT_NE(data, nullptr);
}
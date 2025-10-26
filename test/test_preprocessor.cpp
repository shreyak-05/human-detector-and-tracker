#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "preprocessor.hpp"

class PreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Assume your model needs a 640x640 input
        preprocessor_ = std::make_unique<perception::Preprocessor>(640, 640);
    }
    
    std::unique_ptr<perception::Preprocessor> preprocessor_;
};

TEST_F(PreprocessorTest, ResizeAndPadImage) {
    // Create a non-square, non-model-size frame
    cv::Mat frame(480, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    
    auto result_blob = preprocessor_->process(frame);
    
    // cv::dnn::blobFromImage creates a 4D blob: (1, 3, height, width)
    ASSERT_EQ(result_blob.size[0], 1);   // Batch size
    ASSERT_EQ(result_blob.size[1], 3);    // Channels
    ASSERT_EQ(result_blob.size[2], 640);  // Height
    ASSERT_EQ(result_blob.size[3], 640);  // Width
}
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "transformer.hpp"

class TransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a dummy 3x3 cv::Mat camera_matrix_ with CV_64F (double) type
        // fx=100.0, fy=100.0, cx=50.0, cy=50.0
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 100.0, 0, 50.0, 0, 100.0, 50.0, 0, 0, 1.0);
        
        // Initialize transformer_ = std::make_unique<Transformation>(camera_matrix_);
        transformer_ = std::make_unique<perception::Transformer3D>(camera_matrix_);
    }
    
    cv::Mat camera_matrix_;
    std::unique_ptr<perception::Transformer3D> transformer_;
};

TEST_F(TransformerTest, ProjectPixelTo3D) {
    cv::Point2f pixel(60, 70);
    float depth = 2.0f;
    
    auto result = transformer_->project_to_3d(pixel, depth);
    
    // Based on pinhole math: x = (60-50)*2/100 = 0.2
    // y = (70-50)*2/100 = 0.4
    ASSERT_FLOAT_EQ(result.x, 0.2f);
    ASSERT_FLOAT_EQ(result.y, 0.4f);
    ASSERT_FLOAT_EQ(result.z, 2.0f);
}
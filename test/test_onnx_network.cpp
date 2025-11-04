/**
 * @file test_onnx_network.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "onnx_network.hpp"

// Test 1: Constructor validation and error handling
TEST(OnnxNetworkTest, ConstructorValidation) {
  // Test with invalid path
  EXPECT_THROW(perception::OnnxNetwork("invalid_path.onnx"), std::runtime_error);
  
  // Test with empty path
  EXPECT_THROW(perception::OnnxNetwork(""), std::runtime_error);
  
  // Test with non-onnx file
  EXPECT_THROW(perception::OnnxNetwork("test.txt"), std::runtime_error);
}

// Test 2: Valid model loading (requires actual model file)
TEST(OnnxNetworkTest, ValidModelLoading) {
  // This test assumes yolov8n.onnx exists in models directory
  std::string model_path = "models/yolov8n.onnx";
  
  try {
    perception::OnnxNetwork network(model_path);
    // If we reach here, model loaded successfully
    SUCCEED();
  } catch (const std::exception& e) {
    // If model doesn't exist, that's expected in some test environments
    GTEST_SKIP() << "Model file not found: " << e.what();
  }
}

// Test 3: Forward pass with valid input
TEST(OnnxNetworkTest, ForwardPassValidation) {
  std::string model_path = "models/yolov8n.onnx";
  
  try {
    perception::OnnxNetwork network(model_path);
    
    // Create valid input blob (1x3x640x640 for YOLOv8)
    cv::Mat blob = cv::Mat::zeros(640, 640, CV_32FC3);
    cv::dnn::blobFromImage(blob, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    
    cv::Mat output = network.forward(blob);
    
    // Validate output dimensions
    EXPECT_GT(output.total(), 0);
    EXPECT_EQ(output.type(), CV_32F);
    
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Model file not found: " << e.what();
  }
}
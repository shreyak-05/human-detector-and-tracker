/**
 * @file test_onnx_network.cpp
 * @author Shreya Kalyanaraman  
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <string>
#include "onnx_network/onnx_network.hpp"

TEST(OnnxNetworkTest, ConstructorErrorHandling) {
  EXPECT_THROW(perception::OnnxNetwork(""), std::runtime_error);
  EXPECT_THROW(perception::OnnxNetwork("nonexistent.onnx"), std::runtime_error);
  EXPECT_THROW(perception::OnnxNetwork("/invalid/path/model.onnx"), std::runtime_error);
}

TEST(OnnxNetworkTest, ValidConstructionAndForward) {
  const char* env = std::getenv("YOLONNX");
  std::string model_path = env ? std::string(env) : "models/yolov8n.onnx";
  if (!std::ifstream(model_path).good()) GTEST_SKIP() << "Model not found: " << model_path;

  perception::OnnxNetwork net(model_path);

  cv::Mat img(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, {640,640}, cv::Scalar(), true, false);

  cv::Mat out = net.forward(blob);
  EXPECT_FALSE(out.empty());
}

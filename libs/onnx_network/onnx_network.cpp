/**
 * @file onnx_network.cpp
 * @brief Implementation of ONNX neural network
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include "onnx_network.hpp"

#include <opencv2/dnn.hpp>
#include <stdexcept>

using namespace cv;
using namespace perception;

OnnxNetwork::OnnxNetwork(const std::string& model_path) {
  try {
    net_ = dnn::readNetFromONNX(model_path);
    if (net_.empty()) {
      throw std::runtime_error("Failed to load ONNX model: " + model_path);
    }
  } catch (const cv::Exception& e) {
    throw std::runtime_error("Error loading ONNX model: " + std::string(e.what()));
  }
}

Mat OnnxNetwork::forward(const Mat& blob) {
  net_.setInput(blob);
  return net_.forward();
}


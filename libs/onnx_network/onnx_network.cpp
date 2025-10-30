/*
 * Copyright 2025 Shreya Kalyanaraman and Tirth Sadaria
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /**
 * @file onnx_network.cpp
 * @brief Implementation of ONNX neural network
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include "onnx_network.hpp"

#include <opencv2/dnn.hpp>
#include <stdexcept>
#include <iostream>

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
  std::cout << "OnnxNetwork::forward() - Input blob shape: ";
  for (int i = 0; i < blob.dims; i++) {
    std::cout << blob.size[i] << " ";
  }
  std::cout << std::endl;
  
  net_.setInput(blob);
  Mat output = net_.forward();
  
  std::cout << "OnnxNetwork::forward() - Output shape: ";
  for (int i = 0; i < output.dims; i++) {
    std::cout << output.size[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "OnnxNetwork::forward() - Output type: " << output.type() << " depth: " << output.depth() << std::endl;
  
  return output;
}


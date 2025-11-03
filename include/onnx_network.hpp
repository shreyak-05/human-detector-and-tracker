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
 * @file onnx_network.hpp
 * @brief ONNX neural network implementation
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <opencv2/dnn.hpp>
#include <string>

#include "inetwork.hpp"

namespace perception {

/**
 * @brief ONNX-based neural network implementation.
 *
 * This class implements the INetwork interface using OpenCV's DNN module
 * to load and run ONNX models.
 */
class OnnxNetwork : public INetwork {
 public:
  /**
   * @brief Constructor for ONNX network.
   * @param model_path Path to the ONNX model file
   */
  explicit OnnxNetwork(const std::string& model_path);

  /**
   * @brief Forward pass through the network.
   *
   * Sets the input blob and runs inference, returning the network output.
   *
   * @param blob Input 4D blob tensor [N,C,H,W]
   * @return Network output tensor
   */
  cv::Mat forward(const cv::Mat& blob) override;

 private:
  cv::dnn::Net net_;  ///< OpenCV DNN network instance
};

}  // namespace perception

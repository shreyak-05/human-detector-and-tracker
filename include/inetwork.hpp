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
 * @file inetwork.hpp
 * @brief Neural network interface for abstraction
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Interface for neural network inference.
 * 
 * This abstract interface defines the contract for neural network operations,
 * allowing for dependency injection and mocking in unit tests.
 */
class INetwork {
 public:
  /**
   * @brief Virtual destructor for polymorphic destruction.
   */
  virtual ~INetwork() = default;
  
  /**
   * @brief Forward pass through the neural network.
   * 
   * This pure virtual method must be implemented by concrete network classes.
   * It takes a preprocessed blob and returns the network's output.
   * 
   * @param blob Preprocessed input blob [N,C,H,W] where typically N=1
   * @return Network output tensor
   */
  virtual cv::Mat forward(const cv::Mat& blob) = 0;
};

}  // namespace perception


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
 * @file preprocessor.hpp
 * @brief Image preprocessing utilities for DNN models
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Interface for image preprocessing.
 * 
 * This abstract interface defines the contract for image preprocessing operations.
 * Implementations of this interface are responsible for converting raw image frames
 * into 4D blob tensors suitable for input to neural networks.
 * 
 * @note This interface supports dependency injection, allowing mock implementations
 *       for unit testing and flexible runtime preprocessor selection.
 */
class IPreprocessor {
 public:
  /**
   * @brief Virtual destructor for polymorphic destruction.
   */
  virtual ~IPreprocessor() = default;
  
  /**
   * @brief Process image frame into a blob for DNN.
   * 
   * This pure virtual method must be implemented by concrete preprocessor classes.
   * It takes a raw image frame and returns a 4D blob tensor suitable for neural network input.
   * 
   * @param frame Input image frame (typically BGR format from video/webcam)
   * @return Processed 4D blob tensor [N,C,H,W] where N=1, C=3
   */
  virtual cv::Mat process(const cv::Mat& frame) = 0;
};

/**
 * @brief Image preprocessing for DNN models.
 */
class Preprocessor : public IPreprocessor {
 public:
  /**
   * @brief Constructor for preprocessor.
   * @param input_w Target width for input images (default: 640)
   * @param input_h Target height for input images (default: 640)
   * @param swap_rb Swap R and B channels for RGB/BGR conversion (default: true)
   */
  Preprocessor(int input_w = 640, int input_h = 640, bool swap_rb = true);

  /**
   * @brief Process image frame into a 4D blob for DNN.
   * 
   * Resizes the image to the target size, normalizes pixel values (0-255 to 0-1),
   * swaps RGB/BGR channels if configured, and creates a 4D blob tensor.
   * 
   * @param frame Input image frame (BGR format)
   * @return 4D blob tensor [N,C,H,W] where N=1, C=3, H=input_h, W=input_w
   */
  cv::Mat process(const cv::Mat& frame) override;

  /**
   * @brief Create normalized blob for DNN.
   * @param img Input BGR image
   * @return 4D blob tensor [N,C,H,W]
   */
  cv::Mat makeBlob(const cv::Mat& img) const;

 private:
  int input_w_;   ///< Target width
  int input_h_;   ///< Target height
  bool swap_rb_;  ///< Swap R/B channels
};

}  // namespace perception

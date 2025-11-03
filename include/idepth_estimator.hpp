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
 * @file idepth_estimator.hpp
 * @brief Depth estimation interface for abstraction
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Interface for depth estimation.
 *
 * This abstract interface defines the contract for depth estimation operations,
 * allowing for dependency injection and mocking in unit tests. Concrete
 * implementations may use various approaches such as monocular depth estimation
 * networks (MiDaS, Depth Anything) or stereo vision techniques.
 *
 * @note This interface enables polymorphic depth estimation and facilitates
 * testing by allowing mock implementations for unit testing.
 */
class IDepthEstimator {
 public:
  /**
   * @brief Virtual destructor for polymorphic destruction.
   */
  virtual ~IDepthEstimator() = default;

  /**
   * @brief Estimate depth for a bounding box region.
   *
   * This pure virtual method must be implemented by concrete depth estimator
   * classes. It estimates the depth value (in meters) for a specific region in
   * the image, typically by analyzing the content within the bounding box and
   * applying depth estimation algorithms.
   *
   * @param frame Input image frame (BGR format) containing the full frame
   * @param bbox Bounding box region (x, y, width, height) to estimate depth
   * for. The depth is typically estimated at the center or averaged across the
   * region.
   * @return Estimated depth value in meters (typically positive, with 0.0
   * indicating error)
   */
  virtual float get_depth(const cv::Mat& frame, cv::Rect bbox) = 0;
};

}  // namespace perception

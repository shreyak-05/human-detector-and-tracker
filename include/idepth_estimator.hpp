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
 * allowing for dependency injection and mocking in unit tests. Concrete implementations
 * may use various approaches such as monocular depth estimation networks (MiDaS, 
 * Depth Anything) or stereo vision techniques.
 * 
 * @note This interface enables polymorphic depth estimation and facilitates testing
 *       by allowing mock implementations for unit testing.
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
   * This pure virtual method must be implemented by concrete depth estimator classes.
   * It estimates the depth value (in meters) for a specific region in the image, typically
   * by analyzing the content within the bounding box and applying depth estimation algorithms.
   * 
   * @param frame Input image frame (BGR format) containing the full frame
   * @param bbox Bounding box region (x, y, width, height) to estimate depth for.
   *              The depth is typically estimated at the center or averaged across the region.
   * @return Estimated depth value in meters (typically positive, with 0.0 indicating error)
   */
  virtual float get_depth(const cv::Mat& frame, cv::Rect bbox) = 0;
};

}  // namespace perception


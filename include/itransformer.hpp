/**
 * @file itransformer.hpp
 * @brief 3D transformation interface for abstraction
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Interface for 3D transformation.
 * 
 * This abstract interface defines the contract for 3D coordinate transformation operations,
 * allowing for dependency injection and mocking in unit tests. Concrete implementations
 * use camera intrinsic parameters (focal length, principal point) to convert 2D pixel
 * coordinates with depth into 3D camera coordinates using the pinhole camera model.
 * 
 * @note This interface enables polymorphic coordinate transformation and facilitates testing
 *       by allowing mock implementations for unit testing.
 */
class ITransformer {
 public:
  /**
   * @brief Virtual destructor for polymorphic destruction.
   */
  virtual ~ITransformer() = default;
  
  /**
   * @brief Project 2D pixel and depth to 3D coordinates.
   * 
   * This pure virtual method must be implemented by concrete transformer classes.
   * It converts a 2D pixel coordinate with depth into a 3D point using camera intrinsics
   * and the pinhole camera model. The transformation uses the formula:
   * x = (u - cx) * depth / fx, y = (v - cy) * depth / fy, z = depth
   * 
   * @param pixel 2D pixel coordinates (u, v) in pixel space
   * @param depth Depth value in meters at the pixel location
   * @return 3D point (x, y, z) in camera frame, where x/y are in meters and z is depth
   */
  virtual cv::Point3f project_to_3d(cv::Point2f pixel, float depth) = 0;
};

}  // namespace perception


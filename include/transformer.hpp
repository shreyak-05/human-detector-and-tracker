/**
 * @file transformer.hpp
 * @brief 3D transformation utilities for camera coordinate conversion
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
*/

#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

#include "perception_types.hpp"

namespace perception {

/**
 * @brief Projects 2D pixels to 3D robot coordinates.
 */
class Transformer3D {
 public:
  /**
   * @brief Camera intrinsic parameters.
   */
  struct Intrinsics {
    float fx;  ///< Focal length x (pixels)
    float fy;  ///< Focal length y (pixels)
    float cx;  ///< Principal point x (pixels)
    float cy;  ///< Principal point y (pixels)
  };

  /**
   * @brief Constructor for 3D transformer.
   * @param K Camera intrinsics
   * @param T_cam_to_robot 4x4 transformation matrix
   */
  Transformer3D(const Intrinsics& K, const cv::Matx44f& T_cam_to_robot);

  /**
   * @brief Constructor for 3D transformer with camera matrix.
   * @param camera_matrix 3x3 camera intrinsic matrix containing fx, fy, cx, cy
   */
  Transformer3D(const cv::Mat& camera_matrix);

  /**
   * @brief Project 2D pixel to 3D point using depth.
   * @param pixel 2D pixel coordinates (u, v)
   * @param depth Depth value in meters
   * @return 3D point in camera frame (x, y, z)
   */
  cv::Point3f project_to_3d(cv::Point2f pixel, float depth);

  /**
   * @brief Convert pixel to robot coordinates.
   * @param u Pixel x coordinate
   * @param v Pixel y coordinate
   * @param depth Depth value (meters)
   * @return 3D point in robot frame
   */
  GroundPoint pixelToRobot(float u, float v, float depth) const;

  /**
   * @brief Convert bounding box to robot coordinates.
   * @param box Bounding box
   * @param depth Depth map
   * @return 3D ground point in robot frame
   */
  GroundPoint boxToRobot(const cv::Rect& box, const cv::Mat& depth) const;

 private:
  Intrinsics K_;                ///< Camera intrinsics
  cv::Matx44f T_cam_to_robot_;  ///< Transformation matrix
  cv::Mat camera_matrix_;       ///< Camera intrinsic matrix
};

}  // namespace perception

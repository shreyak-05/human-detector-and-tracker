#pragma once
#include <opencv2/opencv.hpp>

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
};

}  // namespace perception

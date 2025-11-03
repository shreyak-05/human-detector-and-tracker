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
 * @file transformer.hpp
 * @brief 3D transformation utilities for camera coordinate conversion
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

#include "itransformer.hpp"
#include "perception_types.hpp"

namespace perception {

/**
 * @brief Projects 2D pixels to 3D camera/robot coordinates using pinhole camera model.
 *
 * Implements 3D coordinate transformation using camera intrinsic parameters.
 * Converts 2D pixel coordinates (u, v) with depth to 3D points (x, y, z) in the
 * camera frame. Uses the standard pinhole camera model transformation:
 * x = (u - cx) * depth / fx, y = (v - cy) * depth / fy, z = depth
 */
class Transformer3D : public ITransformer {
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
  explicit Transformer3D(const cv::Mat& camera_matrix);

  /**
   * @brief Project 2D pixel to 3D point using depth.
   * @param pixel 2D pixel coordinates (u, v)
   * @param depth Depth value in meters
   * @return 3D point in camera frame (x, y, z)
   */
  cv::Point3f project_to_3d(cv::Point2f pixel, float depth) override;

  /**
   * @brief Project pixel coordinates to 3D space using depth
   * @param pixel Pixel coordinates (u,v)
   * @param depth Depth value at pixel
   * @return 3D point in robot frame
   */
  cv::Point3f project_to_3d(const cv::Point2f& pixel, float depth) const;

 private:
  Intrinsics K_;                ///< Camera intrinsics
  cv::Matx44f T_cam_to_robot_;  ///< Transformation matrix
  cv::Mat camera_matrix_;       ///< Camera intrinsic matrix
};

}  // namespace perception

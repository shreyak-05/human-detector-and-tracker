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

#pragma once
#include "idepth_estimator.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace perception {

/**
 * @brief Monocular depth estimator using ONNX models.
 */
class MLDepthEstimator : public IDepthEstimator {
 public:
  /**
   * @brief Constructor for depth estimator.
   * @param model_path Path to ONNX model file
   * @param input_w Model input width (default: 256)
   * @param input_h Model input height (default: 256)
   * @param use_gpu Use GPU acceleration (default: false)
   */
  MLDepthEstimator(const std::string& model_path, int input_w = 256,
                   int input_h = 256, bool use_gpu = false);

  /**
   * @brief Predict depth map from image.
   * @param bgr Input BGR image
   * @return Depth map (CV_32F)
   */
  cv::Mat infer(const cv::Mat& bgr);

  /**
   * @brief Normalize depth for display.
   * @param depth Input depth map
   * @return Normalized depth (CV_8U, 0-255)
   */
  static cv::Mat normalizeDepth(const cv::Mat& depth);

  /**
   * @brief Get depth at bounding box center.
   * 
   * Runs inference and extracts depth value at bbox center.
   * 
   * @param frame Input image frame
   * @param bbox Bounding box region
   * @return Depth value in meters
   */
  float get_depth(const cv::Mat& frame, cv::Rect bbox) override;

  /**
   * @brief Set current frame ID for caching.
   * @param frame_id Current frame number
   */
  void set_frame_id(int frame_id);

  /**
   * @brief Get cached depth map.
   * @return Cached depth map
   */
  cv::Mat get_cached_depth_map() const;

 private:
  Ort::Env env_;                   ///< ONNX Runtime environment
  Ort::SessionOptions session_options_;  ///< Session options
  std::unique_ptr<Ort::Session> session_; ///< ONNX Runtime session
  Ort::MemoryInfo memory_info_;    ///< Memory info
  int input_w_;       ///< Input width
  int input_h_;       ///< Input height
  
  // Caching for performance
  cv::Mat cached_depth_map_;       ///< Cached depth map
  int cached_depth_frame_id_ = -1; ///< Frame ID of cached depth map
  int current_frame_id_ = 0;       ///< Current frame ID
};

}  // namespace perception

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
 * @file detector_tracker.hpp
 * @brief YOLO-based human detection and tracking
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "idepth_estimator.hpp"
#include "inetwork.hpp"
#include "itransformer.hpp"
#include "perception_types.hpp"
#include "preprocessor.hpp"

namespace perception {

/**
 * @brief YOLO-based human detector with IoU tracking.
 *
 * Integrates YOLO detection with depth estimation and 3D transformation.
 * Uses dependency injection for modular architecture and testability.
 */
class DetectorTracker {
  friend class DetectorTrackerTest_PostProcessYoloOutput_Test;

 public:
  /**
   * @brief Constructor with dependency injection.
   *
   * Injects all dependencies for clean separation of concerns and testing.
   *
   * @param preprocessor Image preprocessor for network input
   * @param network Neural network for detection inference
   * @param depth_estimator Depth estimator for 3D positioning
   * @param transformer 3D coordinate transformer
   * @param confidence_threshold Detection confidence threshold (default: 0.5)
   */
  explicit DetectorTracker(std::shared_ptr<IPreprocessor> preprocessor,
                           std::shared_ptr<INetwork> network,
                           std::shared_ptr<IDepthEstimator> depth_estimator,
                           std::shared_ptr<ITransformer> transformer,
                           float confidence_threshold = 0.5f);

  /**
   * @brief Detect humans in a frame.
   *
   * Runs the full detection pipeline: preprocess → inference → post-process.
   *
   * @param frame Input video frame (BGR format)
   * @return Vector of filtered detections
   */
  std::vector<Detection> detect(const cv::Mat& frame);

  /**
   * @brief Get 3D positions of detected humans.
   *
   * Extends detect() with depth estimation and 3D transformation.
   *
   * @param frame Input video frame (BGR format)
   * @return Vector of 3D detections with positions
   */
  std::vector<Detection3D> get_3d_positions(const cv::Mat& frame);

  /**
   * @brief Post-process YOLO output to extract detections.
   *
   * Parses YOLO tensor, filters by confidence, and applies NMS.
   *
   * @param output YOLO network output tensor
   * @param img_width Original image width
   * @param img_height Original image height
   * @param conf_thresh Confidence threshold
   * @return Vector of filtered detections
   */
  std::vector<Detection> post_process(const cv::Mat& output, int img_width,
                                      int img_height, float conf_thresh) const;

  /**
   * @brief Calculate Intersection over Union between two rectangles.
   * @param a First bounding box
   * @param b Second bounding box
   * @return IoU score [0.0, 1.0]
   */
  float iou(const cv::Rect& a, const cv::Rect& b) const;

 private:
  std::shared_ptr<IPreprocessor>
      preprocessor_;                   ///< Image preprocessor for network input
  std::shared_ptr<INetwork> network_;  ///< Neural network for inference
  std::shared_ptr<IDepthEstimator> depth_estimator_;  ///< Depth estimator
  std::shared_ptr<ITransformer> transformer_;  ///< 3D coordinate transformer
  float confidence_threshold_;  ///< Confidence threshold for filtering
  int next_id_{0};              ///< Next available track ID
  std::vector<Track> tracks_;   ///< Currently active tracks
  int frame_count_{0};          ///< Frame counter for optimized logging
};

}  // namespace perception

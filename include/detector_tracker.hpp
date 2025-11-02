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
 * @brief YOLO-based person detector and IoU tracker.
 *
 * Combines object detection using YOLO models with IoU-based tracking
 * to maintain persistent track IDs across video frames.
 */
class DetectorTracker {
  friend class DetectorTrackerTest_PostProcessYoloOutput_Test;
 public:
  /**
   * @brief Constructor for detector-tracker with dependency injection.
   * 
   * This constructor uses dependency injection for all components, providing a clean
   * separation of concerns and enabling flexible testing with mock objects. All dependencies
   * are passed as shared pointers to their respective interfaces, enabling polymorphic
   * behavior and isolated unit testing without requiring actual ONNX models or camera hardware.
   * 
   * This architecture allows the DetectorTracker to be a pure orchestration class that
   * coordinates the workflow but delegates all concrete operations to injected components.
   * 
   * @param preprocessor Shared pointer to the image preprocessor implementing IPreprocessor.
   *                     Converts raw frames to neural network input blobs.
   * @param network Shared pointer to the neural network implementing INetwork.
   *                Runs YOLO-style detection inference.
   * @param depth_estimator Shared pointer to the depth estimator implementing IDepthEstimator.
   *                        Estimates depth values for detected regions.
   * @param transformer Shared pointer to the transformer implementing ITransformer.
   *                   Converts 2D pixel coordinates with depth to 3D positions.
   * @param confidence_threshold Confidence threshold for filtering detections (default: 0.5).
   *                              Only detections with confidence >= threshold are returned.
   */
  explicit DetectorTracker(std::shared_ptr<IPreprocessor> preprocessor,
                           std::shared_ptr<INetwork> network,
                           std::shared_ptr<IDepthEstimator> depth_estimator,
                           std::shared_ptr<ITransformer> transformer,
                           float confidence_threshold = 0.5f);

  /**
   * @brief Run detection and update active tracks.
   * @param frame Input video frame
   * @param conf_thresh Confidence threshold for detections
   * @param nms_thresh Non-maximum suppression threshold
   * @param person_class_id Class ID for person detections
   * @return Vector of active tracks with updated positions
   */
  std::vector<Track> step(const cv::Mat& frame, float conf_thresh = 0.4f,
                          float nms_thresh = 0.45f, int person_class_id = 0);

  /**
   * @brief Detect humans in a frame.
   * 
   * This method orchestrates the full detection pipeline:
   * 1. Preprocess the frame using the injected preprocessor
   * 2. Run neural network inference on the preprocessed blob
   * 3. Post-process the network output to extract bounding boxes and confidences
   * 
   * @param frame Input video frame (BGR format)
   * @return Vector of filtered detections with bounding boxes and confidence scores
   */
  std::vector<Detection> detect(const cv::Mat& frame);

  /**
   * @brief Get 3D positions of detected humans.
   * 
   * This method orchestrates the full 3D position estimation pipeline:
   * 1. Detect humans in 2D using the detect() method to get bounding boxes
   * 2. Estimate depth for each detection using the depth estimator
   * 3. Extract the center pixel of each bounding box
   * 4. Transform 2D coordinates to 3D using the transformer with the estimated depth
   * 
   * The resulting 3D positions are in camera/robot coordinate frame, suitable for
   * navigation and obstacle avoidance systems.
   * 
   * @param frame Input video frame (BGR format) containing the scene to analyze
   * @return Vector of 3D positions with detection IDs. Each Detection3D contains:
   *         - detection_id: Unique identifier for the detection
   *         - position: 3D point (x, y, z) in camera/robot frame (meters)
   */
  std::vector<Detection3D> get_3d_positions(const cv::Mat& frame);

  /**
   * @brief Draw active tracks.
   * @param img Image to draw on (modified in-place)
   * @param tracks Vector of tracks to visualize
   */
  static void drawTracks(cv::Mat& img, const std::vector<Track>& tracks);

  /**
   * @brief Post-process YOLO network output to extract detections.
   * 
   * This method parses YOLO raw output tensor, filters detections by confidence threshold,
   * converts center-based bounding boxes to top-left format, and applies NMS.
   * Public for testing purposes (via FRIEND_TEST).
   * 
   * @param output Raw YOLO network output tensor [5, N_detections] where each column
   *               contains [cx, cy, w, h, conf] for a detection
   * @param conf_thresh Confidence threshold for filtering (e.g., 0.5)
   * @return Vector of filtered and processed detections
   */
  std::vector<Detection> post_process(const cv::Mat& output, 
                                       int img_width, int img_height,
                                       float conf_thresh) const;

 private:

  /**
   * @brief Parse YOLO network output into detections.
   * @param out Raw network output tensor
   * @param img_size Original image size for coordinate scaling
   * @param conf_thresh Confidence threshold for filtering
   * @param class_id Target class ID to filter (0 = person)
   * @return Vector of filtered detections
   */
  std::vector<Detection> parse(const cv::Mat& out, const cv::Size& img_size,
                               float conf_thresh, int class_id) const;

  /**
   * @brief Associate new detections with existing tracks.
   * @param dets Vector of current frame detections
   * @return Updated vector of tracks
   */
  std::vector<Track> associate(const std::vector<Detection>& dets);

  /**
   * @brief Calculate Intersection over Union between two rectangles.
   * @param a First bounding box
   * @param b Second bounding box
   * @return IoU score [0.0, 1.0]
   */
  float iou(const cv::Rect& a, const cv::Rect& b) const;

  std::shared_ptr<IPreprocessor> preprocessor_;     ///< Image preprocessor for network input
  std::shared_ptr<INetwork> network_;               ///< Neural network for inference
  std::shared_ptr<IDepthEstimator> depth_estimator_; ///< Depth estimator
  std::shared_ptr<ITransformer> transformer_;       ///< 3D coordinate transformer
  float confidence_threshold_;                     ///< Confidence threshold for filtering
  int next_id_{0};                                  ///< Next available track ID
  std::vector<Track> tracks_;                       ///< Currently active tracks
  int frame_count_{0};                              ///< Frame counter for optimized logging
};

}  // namespace perception

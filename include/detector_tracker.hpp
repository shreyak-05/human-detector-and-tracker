/**
 * @file detector_tracker.hpp
 * @brief YOLO-based human detection and tracking
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <gtest/gtest.h>
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "inetwork.hpp"
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
  FRIEND_TEST(DetectorTest, PostProcessYoloOutput);
 public:
  /**
   * @brief Constructor for detector-tracker with dependency injection.
   * 
   * This constructor uses dependency injection for both the preprocessor and network,
   * allowing for flexible testing with mock objects and runtime component selection.
   * Both dependencies are passed as shared pointers to their respective interfaces,
   * enabling polymorphic behavior.
   * 
   * @param preprocessor Shared pointer to the image preprocessor implementing IPreprocessor.
   *                     Ownership is transferred to this DetectorTracker instance.
   * @param network Shared pointer to the neural network implementing INetwork.
   *                Ownership is transferred to this DetectorTracker instance.
   * @param confidence_threshold Confidence threshold for filtering detections (default: 0.5)
   * 
   * @throws cv::Exception if the network fails to initialize
   */
  explicit DetectorTracker(std::shared_ptr<IPreprocessor> preprocessor,
                           std::shared_ptr<INetwork> network,
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
  std::vector<Detection> post_process(const cv::Mat& output, float conf_thresh) const;

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

  std::shared_ptr<IPreprocessor> preprocessor_;   ///< Image preprocessor for network input
  std::shared_ptr<INetwork> network_;             ///< Neural network for inference
  float confidence_threshold_;                    ///< Confidence threshold for filtering
  int next_id_{0};                                ///< Next available track ID
  std::vector<Track> tracks_;                     ///< Currently active tracks
};

}  // namespace perception

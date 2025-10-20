#pragma once
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

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
 public:
  /**
   * @brief Constructor for detector-tracker.
   * @param model_path Path to YOLO ONNX model file
   * @param input_w Model input width (default: 640)
   * @param input_h Model input height (default: 640)
   * @param use_gpu Whether to use GPU acceleration (default: false)
   */
  explicit DetectorTracker(const std::string& model_path, int input_w = 640,
                           int input_h = 640, bool use_gpu = false);

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
   * @brief Draw active tracks.
   * @param img Image to draw on (modified in-place)
   * @param tracks Vector of tracks to visualize
   */
  static void drawTracks(cv::Mat& img, const std::vector<Track>& tracks);

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

  cv::dnn::Net net_;           ///< YOLO neural network
  Preprocessor pre_;           ///< Image preprocessor for network input
  int next_id_{0};             ///< Next available track ID
  std::vector<Track> tracks_;  ///< Currently active tracks
};

}  // namespace perception

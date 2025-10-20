#include "detector_tracker.hpp"
#include <algorithm>
using namespace cv;
using namespace perception;

DetectorTracker::DetectorTracker(const std::string& model,
                                 int w, int h, bool use_gpu)
    : pre_(w, h, true) {
  net_ = dnn::readNetFromONNX(model);
  if (use_gpu) {
#if CV_VERSION_MAJOR >= 4
    net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
#endif
  } else {
    net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
  }
}

float DetectorTracker::iou(const Rect& a, const Rect& b) const {
  // TODO: Calculate intersection over union
  return 0.0f;
}

std::vector<Detection> DetectorTracker::parse(const cv::Mat& out,
                                              const cv::Size& img_size,
                                              float conf_thresh,
                                              int cls_id) const {
    // TODO: Parse YOLO output to detections
    std::vector<Detection> detections;
    return detections;
}

std::vector<Track> DetectorTracker::associate(const std::vector<Detection>& dets) {
  // TODO: Associate detections with tracks
  return tracks_;
}

std::vector<Track> DetectorTracker::step(const Mat& frame,
                                         float conf, float nms,
                                         int cls) {
  // TODO: Run detection and tracking pipeline
  std::vector<Track> tracks;
  return tracks;
}

void DetectorTracker::drawTracks(Mat& img, const std::vector<Track>& tracks) {
  // TODO: Draw bounding boxes and track IDs
}

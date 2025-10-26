/**
 * @file detector_tracker.cpp
 * @brief Implementation of YOLO-based human detection and tracking
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include "detector_tracker.hpp"

#include <algorithm>
using namespace cv;
using namespace perception;

DetectorTracker::DetectorTracker(std::shared_ptr<IPreprocessor> preprocessor,
                                 const std::string& model,
                                 bool use_gpu)
    : preprocessor_(std::move(preprocessor)) {
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

std::vector<Detection> DetectorTracker::detect(const cv::Mat& frame) {
  cv::Mat blob = preprocessor_->process(frame);
  
  // We will add DNN logic here in the next step
  
  return {}; // Return empty detections for now
}

std::vector<Detection> DetectorTracker::post_process(const cv::Mat& output, float conf_thresh) const {
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  
  // Iterate columns (detections)
  for (int i = 0; i < output.cols; ++i) {
    float conf = output.at<float>(4, i);
    if (conf < conf_thresh) continue;
    
    float cx = output.at<float>(0, i);
    float cy = output.at<float>(1, i);
    float w = output.at<float>(2, i);
    float h = output.at<float>(3, i);
    
    int x = static_cast<int>(cx - w / 2);
    int y = static_cast<int>(cy - h / 2);
    
    boxes.emplace_back(x, y, static_cast<int>(w), static_cast<int>(h));
    confidences.push_back(conf);
  }
  
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, 0.4f, indices);
  
  std::vector<Detection> detections;
  for (int idx : indices) {
    detections.push_back({boxes[idx], confidences[idx], 0});
  }
  
  return detections;
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

std::vector<Track> DetectorTracker::associate(
    const std::vector<Detection>& dets) {
  // TODO: Associate detections with tracks
  return tracks_;
}

std::vector<Track> DetectorTracker::step(const Mat& frame, float conf,
                                         float nms, int cls) {
  // TODO: Run detection and tracking pipeline
  std::vector<Track> tracks;
  return tracks;
}

void DetectorTracker::drawTracks(Mat& img, const std::vector<Track>& tracks) {
  // TODO: Draw bounding boxes and track IDs
}

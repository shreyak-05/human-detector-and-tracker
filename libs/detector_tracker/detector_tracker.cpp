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
                                 std::shared_ptr<INetwork> network,
                                 std::shared_ptr<IDepthEstimator> depth_estimator,
                                 std::shared_ptr<ITransformer> transformer,
                                 float confidence_threshold)
    : preprocessor_(std::move(preprocessor)),
      network_(std::move(network)),
      depth_estimator_(std::move(depth_estimator)),
      transformer_(std::move(transformer)),
      confidence_threshold_(confidence_threshold) {
}

std::vector<Detection> DetectorTracker::detect(const cv::Mat& frame) {
  cv::Mat blob = preprocessor_->process(frame);
  cv::Mat output = network_->forward(blob);
  return post_process(output, confidence_threshold_);
}

std::vector<Detection3D> DetectorTracker::get_3d_positions(const cv::Mat& frame) {
  // First, get the 2D detections
  std::vector<Detection> detections = detect(frame);
  
  std::vector<Detection3D> positions;
  
  // Now, iterate and get 3D info
  for (const auto& det : detections) {
    // Get depth for the detection
    float depth = depth_estimator_->get_depth(frame, det.box);
    
    // Get the pixel at the center of the bounding box
    cv::Point2f center_pixel(det.box.x + det.box.width / 2.0f, 
                             det.box.y + det.box.height / 2.0f);
    
    // Convert to 3D
    cv::Point3f pos = transformer_->project_to_3d(center_pixel, depth);
    
    positions.push_back({0, det.box, pos}); // TODO: Add real tracking ID
  }
  
  return positions;
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

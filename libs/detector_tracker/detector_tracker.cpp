/*
 * Copyright 2025 Shreya Kalyanaraman anstd::vector<Dstd::vector<Detection> DetectorTracker::detect(const cv::Mat& frame) {
  frame_count_++;
  cv::Mat blob = preprocessor_->process(fra  // Print structured summary (only very occasionally to reduce spam)
  static int call_count = 0;
  call_count++;
  if (call_count % 50 == 1 && max_person_conf > 0.0f) {
    std::cout << "Detection: " << boxes.size() << " candidates, " 
              << indices.size() << " after NMS (max confidence: " 
              << std::fixed << std::setprecision(3) << max_person_conf << ")" << std::endl;
  }v::Mat output = network_->forward(blob);
  
  auto detections = post_process(output, frame.cols, frame.rows, confidence_threshold_);
  
  // Only print summary every 50 frames to reduce console spam
  if (frame_count_ % 50 == 1) {
    std::cout << "Found " << detections.size() << " human(s)" << std::endl;
  }
  
  return detections;
}ctorTracker::detect(const cv::Mat& frame) {
  frame_count_++;
  cv::Mat blob = preprocessor_->process(frame);
  cv::Mat output = network_->forward(blob);
  
  auto detections = post_process(output, frame.cols, frame.rows, confidence_threshold_);
  
  // Only print summary every 10 frames to reduce console spam
  if (frame_count_ % 10 == 1) {
    std::cout << "Found " << detections.size() << " human(s)" << std::endl;
  }
  
  return detections;
}a
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
#include <iostream>
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
      confidence_threshold_(confidence_threshold),
      frame_count_(0) {
}

std::vector<Detection> DetectorTracker::detect(const cv::Mat& frame) {
  frame_count_++;
  cv::Mat blob = preprocessor_->process(frame);
  cv::Mat output = network_->forward(blob);
  
  auto detections = post_process(output, frame.cols, frame.rows, confidence_threshold_);
  
  // Only print summary every 50 frames to reduce console spam further
  if (frame_count_ % 50 == 1) {
    std::cout << "Found " << detections.size() << " human(s)" << std::endl;
  }
  
  return detections;
}

std::vector<Detection3D> DetectorTracker::get_3d_positions(const cv::Mat& frame) {
  // First, get the 2D detections
  std::vector<Detection> detections = detect(frame);
  
  std::vector<Detection3D> positions;
  
  // Process each detection for 3D positioning
  for (size_t i = 0; i < detections.size(); i++) {
    const auto& det = detections[i];
    
    float depth = depth_estimator_->get_depth(frame, det.box);
    cv::Point2f center_pixel(det.box.x + det.box.width / 2.0f, 
                             det.box.y + det.box.height / 2.0f);
    cv::Point3f pos = transformer_->project_to_3d(center_pixel, depth);
    
    positions.push_back({static_cast<int>(i), det.box, pos});
  }
  
  return positions;
}

std::vector<Detection> DetectorTracker::post_process(const cv::Mat& output, 
                                                     int img_width, int img_height,
                                                     float conf_thresh) const {
  // Calculate scaling factors from YOLO input (640x640) to original image
  float x_scale = static_cast<float>(img_width) / 640.0f;
  float y_scale = static_cast<float>(img_height) / 640.0f;
  
  // YOLOv8 outputs [1, 84, 8400] - we need to reshape to [84, 8400]
  cv::Mat reshaped_output;
  if (output.dims == 3 && output.size[0] == 1) {
    int sizes[] = {output.size[1], output.size[2]};
    reshaped_output = output.reshape(0, 2, sizes);
  } else {
    reshaped_output = output;
  }
  
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  
  float max_person_conf = 0.0f;
  
  // Iterate columns (detections) - YOLOv8 format
  for (int i = 0; i < reshaped_output.cols; ++i) {
    // YOLOv8 format: [cx, cy, w, h, class0_prob, class1_prob, ..., class79_prob]
    float cx = reshaped_output.at<float>(0, i);
    float cy = reshaped_output.at<float>(1, i);
    float w = reshaped_output.at<float>(2, i);
    float h = reshaped_output.at<float>(3, i);
    
    // Find the highest class probability and its index
    float max_class_prob = 0.0f;
    int best_class_id = -1;
    for (int c = 0; c < 80; ++c) {  // 80 classes in COCO
      float class_prob = reshaped_output.at<float>(4 + c, i);
      if (class_prob > max_class_prob) {
        max_class_prob = class_prob;
        best_class_id = c;
      }
    }
    
    // Check specifically for person class (class 0)
    float person_class_prob = reshaped_output.at<float>(4, i);  // class 0 = person
    
    if (person_class_prob > max_person_conf) {
      max_person_conf = person_class_prob;
    }
    
    // Only keep "person" detections (class 0)
    if (best_class_id != 0) continue;
    
    // Final confidence = class_probability (no separate objectness in modern YOLOv8)
    float final_conf = max_class_prob;
    
    if (final_conf < conf_thresh) continue;
    
    // Scale coordinates from YOLO input size (640x640) to original image size
    float scaled_cx = cx * x_scale;
    float scaled_cy = cy * y_scale;
    float scaled_w = w * x_scale;
    float scaled_h = h * y_scale;
    
    // Convert from center-based to top-left corner format
    int x = static_cast<int>(scaled_cx - scaled_w / 2);
    int y = static_cast<int>(scaled_cy - scaled_h / 2);
    
    // Ensure coordinates are within image bounds
    x = std::max(0, std::min(x, img_width - 1));
    y = std::max(0, std::min(y, img_height - 1));
    int width = static_cast<int>(scaled_w);
    int height = static_cast<int>(scaled_h);
    
    // Ensure box doesn't exceed image boundaries
    width = std::min(width, img_width - x);
    height = std::min(height, img_height - y);
    
    boxes.emplace_back(x, y, width, height);
    confidences.push_back(final_conf);
  }
  
  // Apply Non-Maximum Suppression
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, 0.5f, indices);
  
  // Print structured summary (only occasionally to reduce spam)
  static int call_count = 0;
  call_count++;
  if (call_count % 10 == 1 && max_person_conf > 0.0f) {
    std::cout << "Detection: " << boxes.size() << " candidates, " 
              << indices.size() << " after NMS (max confidence: " 
              << std::fixed << std::setprecision(3) << max_person_conf << ")" << std::endl;
  }
  
  std::vector<Detection> detections;
  for (int idx : indices) {
    detections.push_back({boxes[idx], confidences[idx], 0});
  }
  
  return detections;
}

std::vector<Track> DetectorTracker::step(const Mat& frame, float conf,
                                         float nms, int cls) {
  // Simple implementation: convert detections to tracks
  auto detections = detect(frame);
  std::vector<Track> tracks;
  
  for (size_t i = 0; i < detections.size(); i++) {
    Track track;
    track.id = static_cast<int>(i);
    track.det = detections[i];
    track.age = 1;
    track.time_since_update = 0;
    tracks.push_back(track);
  }
  
  return tracks;
}

float DetectorTracker::iou(const Rect& a, const Rect& b) const {
  // Calculate intersection area
  int x1 = std::max(a.x, b.x);
  int y1 = std::max(a.y, b.y);
  int x2 = std::min(a.x + a.width, b.x + b.width);
  int y2 = std::min(a.y + a.height, b.y + b.height);
  
  if (x2 <= x1 || y2 <= y1) {
    return 0.0f;  // No intersection
  }
  
  int intersection_area = (x2 - x1) * (y2 - y1);
  int area_a = a.width * a.height;
  int area_b = b.width * b.height;
  int union_area = area_a + area_b - intersection_area;
  
  return static_cast<float>(intersection_area) / static_cast<float>(union_area);
}

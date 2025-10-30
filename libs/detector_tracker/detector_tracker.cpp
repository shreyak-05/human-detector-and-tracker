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
      confidence_threshold_(confidence_threshold) {
}

std::vector<Detection> DetectorTracker::detect(const cv::Mat& frame) {
  std::cout << "DetectorTracker::detect() - Frame size: " << frame.size() << std::endl;
  
  cv::Mat blob = preprocessor_->process(frame);
  std::cout << "DetectorTracker::detect() - Blob size: " << blob.size() << std::endl;
  
  cv::Mat output = network_->forward(blob);
  std::cout << "DetectorTracker::detect() - Network output size: " << output.size() << std::endl;
  
  auto detections = post_process(output, frame.cols, frame.rows, 0.1f);  // Lower threshold for testing
  std::cout << "DetectorTracker::detect() - Found " << detections.size() << " detections with confidence > " << confidence_threshold_ << std::endl;
  
  return detections;
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

std::vector<Detection> DetectorTracker::post_process(const cv::Mat& output, 
                                                     int img_width, int img_height,
                                                     float conf_thresh) const {
  std::cout << "post_process() - Original image size: " << img_width << "x" << img_height << std::endl;
  std::cout << "post_process() - Output dimensions: " << output.rows << "x" << output.cols << std::endl;
  std::cout << "post_process() - Output dims: " << output.dims << std::endl;
  for (int i = 0; i < output.dims; i++) {
    std::cout << "post_process() - Dim " << i << ": " << output.size[i] << std::endl;
  }
  std::cout << "post_process() - Confidence threshold: " << conf_thresh << std::endl;
  
  // Calculate scaling factors from YOLO input (640x640) to original image
  float x_scale = static_cast<float>(img_width) / 640.0f;
  float y_scale = static_cast<float>(img_height) / 640.0f;
  std::cout << "post_process() - Scale factors: x=" << x_scale << ", y=" << y_scale << std::endl;
  
  // YOLOv8 outputs [1, 84, 8400] - we need to reshape to [84, 8400]
  cv::Mat reshaped_output;
  if (output.dims == 3 && output.size[0] == 1) {
    // Reshape from [1, 84, 8400] to [84, 8400]
    int sizes[] = {output.size[1], output.size[2]};
    reshaped_output = output.reshape(0, 2, sizes);
    std::cout << "post_process() - Reshaped to: " << reshaped_output.rows << "x" << reshaped_output.cols << std::endl;
  } else {
    reshaped_output = output;
    std::cout << "post_process() - Using original shape" << std::endl;
  }
  
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  
  int valid_detections = 0;
  int person_detections = 0;
  float max_person_conf = 0.0f;
  
  // Track some stats for debugging
  std::vector<float> all_person_confidences;
  
  // Sample some raw output values to understand what we're getting
  std::cout << "post_process() - Sampling raw output values:" << std::endl;
  for (int sample = 0; sample < 5; ++sample) {
    int idx = sample * 1000;  // Sample every 1000th detection
    if (idx >= reshaped_output.cols) break;
    
    std::cout << "  Detection " << idx << ": ";
    std::cout << "cx=" << reshaped_output.at<float>(0, idx) << " ";
    std::cout << "cy=" << reshaped_output.at<float>(1, idx) << " ";
    std::cout << "w=" << reshaped_output.at<float>(2, idx) << " ";
    std::cout << "h=" << reshaped_output.at<float>(3, idx) << " ";
    std::cout << "cls0=" << reshaped_output.at<float>(4, idx) << " ";
    std::cout << "cls1=" << reshaped_output.at<float>(5, idx) << " ";
    std::cout << "cls2=" << reshaped_output.at<float>(6, idx) << std::endl;
  }
  
  // Iterate columns (detections) - YOLOv8 format
  for (int i = 0; i < reshaped_output.cols; ++i) {
    // YOLOv8 format: [cx, cy, w, h, class0_prob, class1_prob, ..., class79_prob]
    // Note: Modern YOLOv8 doesn't have a separate objectness score
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
    
    if (person_class_prob > 0.01f) {  // Collect stats for any non-trivial person confidence
      all_person_confidences.push_back(person_class_prob);
      if (person_class_prob > max_person_conf) {
        max_person_conf = person_class_prob;
      }
    }
    
    // Only keep "person" detections (class 0)
    if (best_class_id != 0) continue;
    
    person_detections++;
    
    // Final confidence = class_probability (no separate objectness in modern YOLOv8)
    float final_conf = max_class_prob;
    
    if (final_conf < conf_thresh) continue;
    
    valid_detections++;
    
    // Scale coordinates from YOLO input size (640x640) to original image size
    float scaled_cx = cx * x_scale;
    float scaled_cy = cy * y_scale;
    float scaled_w = w * x_scale;
    float scaled_h = h * y_scale;
    
    if (valid_detections <= 5) {  // Show first 5 detections only
      std::cout << "Valid detection " << valid_detections << ": conf=" << final_conf 
                << " (cls_prob=" << max_class_prob << ")"
                << " original_box=(" << cx << "," << cy << "," << w << "," << h << ")"
                << " scaled_box=(" << scaled_cx << "," << scaled_cy << "," << scaled_w << "," << scaled_h << ")" << std::endl;
    }
    
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
  
  std::cout << "post_process() - Total person detections (any confidence): " << person_detections << std::endl;
  std::cout << "post_process() - Found " << valid_detections << " detections above threshold" << std::endl;
  std::cout << "post_process() - Max person confidence found: " << max_person_conf << std::endl;
  
  // Show distribution of person confidences
  if (!all_person_confidences.empty()) {
    std::sort(all_person_confidences.begin(), all_person_confidences.end(), std::greater<float>());
    std::cout << "post_process() - Top 10 person confidences: ";
    for (size_t i = 0; i < std::min(size_t(10), all_person_confidences.size()); ++i) {
      std::cout << all_person_confidences[i] << " ";
    }
    std::cout << std::endl;
  }
  
  std::cout << "post_process() - Before NMS: " << boxes.size() << " boxes" << std::endl;
  
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, 0.5f, indices);  // Increased NMS threshold
  
  std::cout << "post_process() - After NMS: " << indices.size() << " boxes kept" << std::endl;
  
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

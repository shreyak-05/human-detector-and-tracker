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

#include "ml_depth_estimator.hpp"
using namespace cv;
using namespace perception;

MLDepthEstimator::MLDepthEstimator(const std::string& path, int w, int h,
                                   bool use_gpu)
    : input_w_(w), input_h_(h) {
  net_ = dnn::readNetFromONNX(path);
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

Mat MLDepthEstimator::infer(const Mat& bgr) {
  // TODO: Run depth estimation inference
  Mat depth = Mat::zeros(bgr.size(), CV_32F);
  return depth;
}

Mat MLDepthEstimator::normalizeDepth(const Mat& d) {
  // TODO: Normalize depth for visualization
  Mat normalized = Mat::zeros(d.size(), CV_8U);
  return normalized;
}

float MLDepthEstimator::get_depth(const Mat& frame, Rect bbox) {
  // Get the center of the bounding box
  cv::Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
  
  // Extract the ROI from the frame
  Rect roi(bbox.x, bbox.y, bbox.width, bbox.height);
  roi &= Rect(0, 0, frame.cols, frame.rows);  // Make sure ROI is within frame bounds
  
  if (roi.width <= 0 || roi.height <= 0) {
    return 0.0f;
  }
  
  Mat frame_roi = frame(roi);
  
  // Run inference to get depth map
  Mat depth_map = infer(frame_roi);
  
  if (depth_map.empty()) {
    return 0.0f;
  }
  
  // Get the depth value at the center (accounting for ROI offset)
  int local_center_x = static_cast<int>(center.x - bbox.x);
  int local_center_y = static_cast<int>(center.y - bbox.y);
  
  // Ensure coordinates are within bounds
  local_center_x = std::max(0, std::min(local_center_x, depth_map.cols - 1));
  local_center_y = std::max(0, std::min(local_center_y, depth_map.rows - 1));
  
  // Extract the depth value
  float depth_value = depth_map.at<float>(local_center_y, local_center_x);
  
  return depth_value;
}

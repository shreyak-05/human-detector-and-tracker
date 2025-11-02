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

 #include "transformer.hpp"
using namespace cv;
using namespace perception;

Transformer3D::Transformer3D(const Intrinsics& K, const Matx44f& T)
    : K_(K), T_cam_to_robot_(T) {}

Transformer3D::Transformer3D(const cv::Mat& camera_matrix)
    : camera_matrix_(camera_matrix.clone()) {}

cv::Point3f Transformer3D::project_to_3d(cv::Point2f pixel, float depth) {
  double fx = camera_matrix_.at<double>(0, 0);
  double fy = camera_matrix_.at<double>(1, 1);
  double cx = camera_matrix_.at<double>(0, 2);
  double cy = camera_matrix_.at<double>(1, 2);
  
  float x = (pixel.x - cx) * depth / fx;
  float y = (pixel.y - cy) * depth / fy;
  
  return cv::Point3f(x, y, depth);
}

GroundPoint Transformer3D::pixelToRobot(float u, float v, float d) const {
  double fx = camera_matrix_.at<double>(0, 0);
  double fy = camera_matrix_.at<double>(1, 1);
  double cx = camera_matrix_.at<double>(0, 2);
  double cy = camera_matrix_.at<double>(1, 2);
  
  float x = (u - cx) * d / fx;
  float y = (v - cy) * d / fy;
  
  return {x, y, d};
}

GroundPoint Transformer3D::boxToRobot(const Rect& box, const Mat& depth) const {
  float center_u = box.x + box.width / 2.0f;
  float center_v = box.y + box.height / 2.0f;
  
  int u = std::max(0, std::min(static_cast<int>(center_u), depth.cols - 1));
  int v = std::max(0, std::min(static_cast<int>(center_v), depth.rows - 1));
  
  float d = depth.at<float>(v, u);
  if (d <= 0.0f) d = 2.0f;  // Default fallback
  
  return pixelToRobot(center_u, center_v, d);
}

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
  
  auto x = (pixel.x - cx) * depth / fx;
  auto y = (pixel.y - cy) * depth / fy;
  
  return cv::Point3f(x, y, depth);
}

GroundPoint Transformer3D::pixelToRobot(float u, float v, float d) const {
  // TODO: Transform pixel to robot coordinates
  return {0.0f, 0.0f, 0.0f};
}

GroundPoint Transformer3D::boxToRobot(const Rect& box, const Mat& depth) const {
  // TODO: Transform bounding box to robot coordinates
  return {0.0f, 0.0f, 0.0f};
}

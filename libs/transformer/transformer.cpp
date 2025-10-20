#include "transformation.hpp"
using namespace cv;
using namespace perception;

Transformer3D::Transformer3D(const Intrinsics& K, const Matx44f& T)
    : K_(K), T_cam_to_robot_(T) {}

GroundPoint Transformer3D::pixelToRobot(float u, float v, float d) const {
  // TODO: Transform pixel to robot coordinates
  return {0.0f, 0.0f, 0.0f};
}

GroundPoint Transformer3D::boxToRobot(const Rect& box, const Mat& depth) const {
  // TODO: Transform bounding box to robot coordinates
  return {0.0f, 0.0f, 0.0f};
}

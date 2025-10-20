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

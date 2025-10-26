#pragma once
#include "idepth_estimator.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Monocular depth estimator using ONNX models.
 */
class MLDepthEstimator : public IDepthEstimator {
 public:
  /**
   * @brief Constructor for depth estimator.
   * @param model_path Path to ONNX model file
   * @param input_w Model input width (default: 256)
   * @param input_h Model input height (default: 256)
   * @param use_gpu Use GPU acceleration (default: false)
   */
  MLDepthEstimator(const std::string& model_path, int input_w = 256,
                   int input_h = 256, bool use_gpu = false);

  /**
   * @brief Predict depth map from image.
   * @param bgr Input BGR image
   * @return Depth map (CV_32F)
   */
  cv::Mat infer(const cv::Mat& bgr);

  /**
   * @brief Normalize depth for display.
   * @param depth Input depth map
   * @return Normalized depth (CV_8U, 0-255)
   */
  static cv::Mat normalizeDepth(const cv::Mat& depth);

 private:
  cv::dnn::Net net_;  ///< DNN network
  int input_w_;       ///< Input width
  int input_h_;       ///< Input height
};

}  // namespace perception

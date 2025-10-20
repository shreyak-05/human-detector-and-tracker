#pragma once
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Image preprocessing for DNN models.
 */
class Preprocessor {
 public:
  /**
   * @brief Constructor for preprocessor.
   * @param input_w Target width (default: 640)
   * @param input_h Target height (default: 640)
   * @param swap_rb Swap R and B channels (default: true)
   */
  Preprocessor(int input_w = 640, int input_h = 640, bool swap_rb = true);

  /**
   * @brief Create normalized blob for DNN.
   * @param img Input BGR image
   * @return 4D blob tensor [N,C,H,W]
   */
  cv::Mat makeBlob(const cv::Mat& img) const;

 private:
  int input_w_;   ///< Target width
  int input_h_;   ///< Target height
  bool swap_rb_;  ///< Swap R/B channels
};

}  // namespace perception

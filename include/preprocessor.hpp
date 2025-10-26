/**
 * @file preprocessor.hpp
 * @brief Image preprocessing utilities for DNN models
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

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
   * @param input_w Target width for input images (default: 640)
   * @param input_h Target height for input images (default: 640)
   * @param swap_rb Swap R and B channels for RGB/BGR conversion (default: true)
   */
  Preprocessor(int input_w = 640, int input_h = 640, bool swap_rb = true);

  /**
   * @brief Process image frame into a 4D blob for DNN.
   * 
   * Resizes the image to the target size, normalizes pixel values (0-255 to 0-1),
   * swaps RGB/BGR channels if configured, and creates a 4D blob tensor.
   * 
   * @param frame Input image frame (BGR format)
   * @return 4D blob tensor [N,C,H,W] where N=1, C=3, H=input_h, W=input_w
   */
  cv::Mat process(const cv::Mat& frame);

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

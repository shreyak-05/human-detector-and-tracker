/**
 * @file onnx_network.hpp
 * @brief ONNX neural network implementation
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include "inetwork.hpp"
#include <opencv2/dnn.hpp>
#include <string>

namespace perception {

/**
 * @brief ONNX-based neural network implementation.
 * 
 * This class implements the INetwork interface using OpenCV's DNN module
 * to load and run ONNX models.
 */
class OnnxNetwork : public INetwork {
 public:
  /**
   * @brief Constructor for ONNX network.
   * @param model_path Path to the ONNX model file
   */
  explicit OnnxNetwork(const std::string& model_path);
  
  /**
   * @brief Forward pass through the network.
   * 
   * Sets the input blob and runs inference, returning the network output.
   * 
   * @param blob Input 4D blob tensor [N,C,H,W]
   * @return Network output tensor
   */
  cv::Mat forward(const cv::Mat& blob) override;

 private:
  cv::dnn::Net net_;  ///< OpenCV DNN network instance
};

}  // namespace perception


/**
 * @file inetwork.hpp
 * @brief Neural network interface for abstraction
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

namespace perception {

/**
 * @brief Interface for neural network inference.
 * 
 * This abstract interface defines the contract for neural network operations,
 * allowing for dependency injection and mocking in unit tests.
 */
class INetwork {
 public:
  /**
   * @brief Virtual destructor for polymorphic destruction.
   */
  virtual ~INetwork() = default;
  
  /**
   * @brief Forward pass through the neural network.
   * 
   * This pure virtual method must be implemented by concrete network classes.
   * It takes a preprocessed blob and returns the network's output.
   * 
   * @param blob Preprocessed input blob [N,C,H,W] where typically N=1
   * @return Network output tensor
   */
  virtual cv::Mat forward(const cv::Mat& blob) = 0;
};

}  // namespace perception


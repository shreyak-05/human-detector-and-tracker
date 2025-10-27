/**
 * @file mocks.hpp
 * @brief Mock objects for unit testing
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <gmock/gmock.h>
#include "idepth_estimator.hpp"
#include "inetwork.hpp"
#include "itransformer.hpp"
#include "preprocessor.hpp"

namespace perception {

/**
 * @brief Mock preprocessor for unit testing.
 * 
 * This mock class implements the IPreprocessor interface using Google Mock (GMock).
 * It allows setting expectations on method calls and controlling return values
 * during unit tests, enabling isolated testing of components that depend on
 * IPreprocessor without requiring real image processing.
 * 
 * @note Use with ::testing::StrictMock<MockPreprocessor> to ensure all expected
 *       calls are made and unexpected calls fail the test.
 * 
 * @example
 * ```cpp
 * auto mock = std::make_shared<::testing::StrictMock<MockPreprocessor>>();
 * EXPECT_CALL(*mock, process(::testing::_)).WillOnce(Return(blob));
 * ```
 */
class MockPreprocessor : public IPreprocessor {
 public:
  /**
   * @brief Mock implementation of the process method.
   * 
   * This method uses GMock's MOCK_METHOD macro to enable setting expectations
   * and controlling return values in unit tests.
   * 
   * @param frame Input image frame (can be matched with ::testing::_ for any input)
   * @return Mocked return value (set via EXPECT_CALL)
   */
  MOCK_METHOD(cv::Mat, process, (const cv::Mat& frame), (override));
};

/**
 * @brief Mock network for unit testing.
 * 
 * This mock class implements the INetwork interface using Google Mock (GMock).
 * It allows setting expectations on neural network forward pass calls and controlling
 * return values during unit tests, enabling isolated testing of components that depend
 * on INetwork without requiring real model loading and inference.
 * 
 * @note Use with ::testing::StrictMock<MockNetwork> to ensure all expected
 *       calls are made and unexpected calls fail the test.
 * 
 * @example
 * ```cpp
 * auto mock = std::make_shared<::testing::StrictMock<MockNetwork>>();
 * EXPECT_CALL(*mock, forward(::testing::_)).WillOnce(Return(output));
 * ```
 */
class MockNetwork : public INetwork {
 public:
  /**
   * @brief Mock implementation of the forward method.
   * 
   * This method uses GMock's MOCK_METHOD macro to enable setting expectations
   * and controlling return values in unit tests.
   * 
   * @param blob Input blob tensor [N,C,H,W] (can be matched with ::testing::_ for any input)
   * @return Mocked return value (set via EXPECT_CALL)
   */
  MOCK_METHOD(cv::Mat, forward, (const cv::Mat& blob), (override));
};

/**
 * @brief Mock depth estimator for unit testing.
 * 
 * This mock class implements the IDepthEstimator interface using Google Mock.
 * It allows setting expectations on depth estimation calls during unit tests.
 */
class MockDepthEstimator : public IDepthEstimator {
 public:
  MOCK_METHOD(float, get_depth, (const cv::Mat& frame, cv::Rect bbox), (override));
};

/**
 * @brief Mock transformer for unit testing.
 * 
 * This mock class implements the ITransformer interface using Google Mock.
 * It allows setting expectations on coordinate transformation calls during unit tests.
 */
class MockTransformer : public ITransformer {
 public:
  MOCK_METHOD(cv::Point3f, project_to_3d, (cv::Point2f pixel, float depth), (override));
};

}  // namespace perception


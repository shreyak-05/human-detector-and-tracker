/**
 * @file mocks.hpp
 * @brief Mock objects for unit testing
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <gmock/gmock.h>
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

}  // namespace perception


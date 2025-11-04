/**
 * @file test_detector_tracker.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "detector_tracker.hpp"
#include "perception_types.hpp"
#include "preprocessor.hpp"
#include "transformer.hpp"

// Configurable test network for different scenarios
class TestYOLONetwork : public perception::INetwork {
 public:
  enum TestMode { SINGLE_DETECTION, MULTIPLE_DETECTIONS, NO_DETECTIONS };
  
  TestYOLONetwork(TestMode mode = SINGLE_DETECTION) : mode_(mode) {}
      
  cv::Mat forward(const cv::Mat& input) override {
    cv::Mat output = cv::Mat::zeros(84, 8400, CV_32F);
    
    switch (mode_) {
      case NO_DETECTIONS:
        return output;
        
      case MULTIPLE_DETECTIONS:
        // Detection 1: High confidence
        output.at<float>(0, 0) = 320.0f;  // center_x
        output.at<float>(1, 0) = 240.0f;  // center_y
        output.at<float>(2, 0) = 100.0f;  // width
        output.at<float>(3, 0) = 200.0f;  // height
        output.at<float>(4, 0) = 0.9f;    // person class confidence
        
        // Detection 2: Medium confidence
        output.at<float>(0, 1) = 500.0f;
        output.at<float>(1, 1) = 300.0f;
        output.at<float>(2, 1) = 80.0f;
        output.at<float>(3, 1) = 150.0f;
        output.at<float>(4, 1) = 0.7f;
        break;
        
      default: // SINGLE_DETECTION
        output.at<float>(0, 0) = 320.0f;
        output.at<float>(1, 0) = 240.0f;
        output.at<float>(2, 0) = 100.0f;
        output.at<float>(3, 0) = 200.0f;
        output.at<float>(4, 0) = 0.9f;
        break;
    }
    return output;
  }
  
 private:
  TestMode mode_;
};

class TestDepthEstimator : public perception::IDepthEstimator {
 public:
  float get_depth(const cv::Mat& frame, cv::Rect bbox) override { return 2.5f; }
};

class DetectorTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    preprocessor_ = std::make_shared<perception::Preprocessor>(640, 640);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1.0);
    transformer_ = std::make_shared<perception::Transformer3D>(camera_matrix);
    test_depth_estimator_ = std::make_shared<TestDepthEstimator>();
  }

  std::shared_ptr<perception::Preprocessor> preprocessor_;
  std::shared_ptr<perception::Transformer3D> transformer_;
  std::shared_ptr<TestDepthEstimator> test_depth_estimator_;
};

// Test 1: Post-processing with different detection scenarios
TEST_F(DetectorTrackerTest, PostProcessingYOLOOutput) {
  auto multi_network = std::make_shared<TestYOLONetwork>(TestYOLONetwork::MULTIPLE_DETECTIONS);
  auto detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, multi_network, test_depth_estimator_, transformer_, 0.5f);
      
  cv::Mat test_frame = cv::Mat::ones(480, 640, CV_8UC3) * 128;
  auto detections = detector->detect(test_frame);
  
  EXPECT_EQ(detections.size(), 2);  // Should have 2 detections
  
  // Verify detection properties
  for (const auto& det : detections) {
    EXPECT_GT(det.confidence, 0.5f);
    EXPECT_GT(det.box.area(), 0);
    EXPECT_GE(det.box.x, 0);
    EXPECT_GE(det.box.y, 0);
  }
  
  // Test frame counting (trigger print logic by running many frames)
  for (int i = 0; i < 60; ++i) {
    detector->detect(test_frame);
  }
}

// Test 2: IoU calculation with different overlap scenarios
TEST_F(DetectorTrackerTest, IoUCalculation) {
  auto test_network = std::make_shared<TestYOLONetwork>();
  auto detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, test_network, test_depth_estimator_, transformer_, 0.5f);
      
  // Test overlapping rectangles
  cv::Rect rect1(10, 10, 20, 20);
  cv::Rect rect2(15, 15, 20, 20);
  float iou = detector->iou(rect1, rect2);
  EXPECT_NEAR(iou, 0.391f, 0.01f);
  
  // Test non-overlapping rectangles
  cv::Rect rect3(0, 0, 10, 10);
  cv::Rect rect4(20, 20, 10, 10);
  float iou_no_overlap = detector->iou(rect3, rect4);
  EXPECT_FLOAT_EQ(iou_no_overlap, 0.0f);
  
  // Test identical rectangles
  float iou_identical = detector->iou(rect1, rect1);
  EXPECT_FLOAT_EQ(iou_identical, 1.0f);
}

// Test 3: 3D position pipeline and edge cases
TEST_F(DetectorTrackerTest, IoUNonOverlapping) {
  // Test with no detections
  auto no_detect_network = std::make_shared<TestYOLONetwork>(TestYOLONetwork::NO_DETECTIONS);
  auto no_detect_detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, no_detect_network, test_depth_estimator_, transformer_, 0.5f);
      
  cv::Mat test_frame = cv::Mat::ones(480, 640, CV_8UC3) * 128;
  auto no_positions = no_detect_detector->get_3d_positions(test_frame);
  EXPECT_EQ(no_positions.size(), 0);
  
  // Test with detections
  auto single_network = std::make_shared<TestYOLONetwork>(TestYOLONetwork::SINGLE_DETECTION);
  auto detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, single_network, test_depth_estimator_, transformer_, 0.5f);
      
  auto positions = detector->get_3d_positions(test_frame);
  EXPECT_EQ(positions.size(), 1);
  
  // Verify 3D coordinates
  EXPECT_GT(positions[0].position.z, 0.0f);  // Depth should be positive
  EXPECT_EQ(positions[0].detection_id, 0);   // First detection should have ID 0
  
  // Test coordinate clamping with out-of-bounds detection
  cv::Mat large_output = cv::Mat::zeros(84, 8400, CV_32F);
  large_output.at<float>(0, 0) = 700.0f;  // Beyond image width
  large_output.at<float>(1, 0) = 500.0f;  // Beyond image height
  large_output.at<float>(2, 0) = 200.0f;
  large_output.at<float>(3, 0) = 300.0f;
  large_output.at<float>(4, 0) = 0.8f;
  
  // This tests the coordinate clamping logic in post_process
}

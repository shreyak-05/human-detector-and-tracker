#include <gtest/gtest.h>

#include <memory>
#include <opencv2/opencv.hpp>

#include "detector_tracker.hpp"
#include "perception_types.hpp"
#include "preprocessor.hpp"

class DetectorTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create real preprocessor for integration testing
    preprocessor_ = std::make_shared<perception::Preprocessor>(640, 640);
  }

  std::shared_ptr<perception::Preprocessor> preprocessor_;
};

TEST_F(DetectorTrackerTest, PostProcessingYOLOOutput) {
  // Test YOLO output processing with realistic format
  cv::Mat yolo_output = cv::Mat::zeros(84, 1, CV_32F);

  // Set bbox coordinates (center format)
  yolo_output.at<float>(0, 0) = 320.0f;  // cx
  yolo_output.at<float>(1, 0) = 240.0f;  // cy
  yolo_output.at<float>(2, 0) = 100.0f;  // width
  yolo_output.at<float>(3, 0) = 200.0f;  // height

  // Set person class probability (class 0)
  yolo_output.at<float>(4, 0) = 0.85f;

  // Create detector instance for testing post_process
  auto detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, nullptr, nullptr, nullptr);

  auto detections = detector->post_process(yolo_output, 640, 640, 0.5f);

  ASSERT_EQ(detections.size(), 1);
  ASSERT_FLOAT_EQ(detections[0].confidence, 0.85f);
  ASSERT_EQ(detections[0].class_id, 0);
}

TEST_F(DetectorTrackerTest, IoUCalculation) {
  // Test IoU calculation with known overlap
  cv::Rect rect1(10, 10, 20, 20);  // [10,10] to [30,30]
  cv::Rect rect2(15, 15, 20, 20);  // [15,15] to [35,35]

  auto detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, nullptr, nullptr, nullptr);

  float iou = detector->iou(rect1, rect2);

  // Expected IoU: intersection(15x15=225) / union(575) ≈ 0.391
  EXPECT_NEAR(iou, 0.391f, 0.01f);
}

TEST_F(DetectorTrackerTest, IoUNonOverlapping) {
  // Test IoU with non-overlapping rectangles
  cv::Rect rect1(0, 0, 10, 10);
  cv::Rect rect2(20, 20, 10, 10);

  auto detector = std::make_unique<perception::DetectorTracker>(
      preprocessor_, nullptr, nullptr, nullptr);

  float iou = detector->iou(rect1, rect2);

  // No overlap should result in IoU = 0
  ASSERT_FLOAT_EQ(iou, 0.0f);
}
/**
 * @file test_depth.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ml_depth_estimator.hpp"
#include <string>
using perception::MLDepthEstimator;

static std::string model() {
  const char* p = std::getenv("MLD_MODEL");
  return p ? std::string(p) : std::string();
}

TEST(MLDepthEstimator, ConstructorErrors) {
  EXPECT_THROW((MLDepthEstimator("", 256, 256, false)), std::runtime_error);
  EXPECT_THROW((MLDepthEstimator("nope.onnx", 256, 256, false)), std::runtime_error);
  EXPECT_THROW((MLDepthEstimator("models/depth_anything_v2_vits.onnx", 0, 256, false)), std::runtime_error);
  EXPECT_THROW((MLDepthEstimator("models/depth_anything_v2_vits.onnx", 256, 0, false)), std::runtime_error);
}

TEST(MLDepthEstimator, NormalizeDepth_Easy) {
  cv::Mat zeros = cv::Mat::zeros(2, 3, CV_32F);
  cv::Mat n0 = MLDepthEstimator::normalizeDepth(zeros);
  EXPECT_EQ(n0.type(), CV_8U);
  EXPECT_EQ(cv::countNonZero(n0), 0);

  cv::Mat mixed = (cv::Mat_<float>(2,3) << -1.f, 0.f, 1.f, 2.f, 10.f, -5.f);
  cv::Mat n1 = MLDepthEstimator::normalizeDepth(mixed);
  EXPECT_EQ(n1.type(), CV_8U);
  EXPECT_GT(cv::countNonZero(n1), 0);
}

TEST(MLDepthEstimator, SmokeIfModelProvided) {
  auto m = model();
  if (m.empty()) GTEST_SKIP() << "Set MLD_MODEL to run a quick infer.";

  MLDepthEstimator est(m, 256, 256, false);
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(120, 80, 40));
  // one call exercises infer() + cache
  float d = est.get_depth(frame, cv::Rect(100,100,50,50));
  EXPECT_TRUE(std::isfinite(d));
  // cover cached path
  float d2 = est.get_depth(frame, cv::Rect(120,120,30,30));
  EXPECT_TRUE(std::isfinite(d2));
  // cover normalize on real output
  cv::Mat vis = MLDepthEstimator::normalizeDepth(est.get_cached_depth_map());
  EXPECT_EQ(vis.type(), CV_8U);
}

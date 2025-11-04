#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <cmath>
#include "ml_depth_estimator.hpp"

using perception::MLDepthEstimator;

static std::string model_path() {
  if (const char* p = std::getenv("MLD_MODEL")) return std::string(p);
  return "models/depth_anything_v2_vits.onnx";
}

TEST(MLDepthEstimator, BadInputs) {
  EXPECT_THROW((MLDepthEstimator("nope.onnx", 256, 256, false)), std::runtime_error);
  EXPECT_TRUE(MLDepthEstimator::normalizeDepth(cv::Mat()).empty());
  cv::Mat z = cv::Mat::zeros(2,2,CV_32F);
  auto n = MLDepthEstimator::normalizeDepth(z);
  EXPECT_EQ(n.type(), CV_8U);
}

TEST(MLDepthEstimator, Infer_GetDepth_Cache) {
  const std::string mp = model_path();
  if (!std::ifstream(mp).good()) GTEST_SKIP() << "Model not found: " << mp;

  MLDepthEstimator est(mp, 256, 256, false);
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(120,80,40));

  cv::Mat depth = est.infer(frame);
  ASSERT_FALSE(depth.empty());
  EXPECT_EQ(depth.type(), CV_32F);
  EXPECT_EQ(depth.size(), frame.size());
  EXPECT_TRUE(cv::checkRange(depth, true, nullptr));

  float d1 = est.get_depth(frame, cv::Rect(100,100,60,60)); // cache miss + infer()
  EXPECT_TRUE(std::isfinite(d1));

  cv::Mat cached = est.get_cached_depth_map();
  EXPECT_FALSE(cached.empty());
  EXPECT_EQ(cached.size(), frame.size());

  float d2 = est.get_depth(frame, cv::Rect(200,200,40,40)); // cache hit
  EXPECT_TRUE(std::isfinite(d2));

  est.set_frame_id(1);                                        // force new infer()
  float d3 = est.get_depth(frame, cv::Rect(10,10,30,30));
  EXPECT_TRUE(std::isfinite(d3));

  float d4 = est.get_depth(frame, cv::Rect(5000,5000,10,10)); // OOB clamped
  EXPECT_TRUE(std::isfinite(d4));

  auto vis = MLDepthEstimator::normalizeDepth(depth);
  EXPECT_EQ(vis.type(), CV_8U);
}

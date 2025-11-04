/**
 * @file test_depth.cpp
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */
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

  float d1 = est.get_depth(frame, cv::Rect(100,100,60,60));
  EXPECT_TRUE(std::isfinite(d1));

  cv::Mat cached = est.get_cached_depth_map();
  EXPECT_FALSE(cached.empty());
  EXPECT_EQ(cached.size(), frame.size());

  float d2 = est.get_depth(frame, cv::Rect(200,200,40,40));
  EXPECT_TRUE(std::isfinite(d2));

  est.set_frame_id(1);
  float d3 = est.get_depth(frame, cv::Rect(10,10,30,30));
  EXPECT_TRUE(std::isfinite(d3));

  float d4 = est.get_depth(frame, cv::Rect(5000,5000,10,10));
  EXPECT_TRUE(std::isfinite(d4));

  auto vis = MLDepthEstimator::normalizeDepth(depth);
  EXPECT_EQ(vis.type(), CV_8U);
}

TEST(MLDepthEstimator, MultipleInferCalls) {
  const std::string mp = model_path();
  if (!std::ifstream(mp).good()) GTEST_SKIP() << "Model not found: " << mp;

  MLDepthEstimator est(mp, 256, 256, false);
  
  std::vector<cv::Mat> frames;
  frames.push_back(cv::Mat(480, 640, CV_8UC3, cv::Scalar(120, 80, 40)));
  frames.push_back(cv::Mat(720, 1280, CV_8UC3, cv::Scalar(255, 0, 0)));
  frames.push_back(cv::Mat(360, 480, CV_8UC3, cv::Scalar(0, 255, 0)));
  frames.push_back(cv::Mat(240, 320, CV_8UC3, cv::Scalar(0, 0, 255)));
  
  cv::Mat random_frame(480, 640, CV_8UC3);
  cv::randu(random_frame, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
  frames.push_back(random_frame);
  
  for (size_t i = 0; i < frames.size(); ++i) {
    cv::Mat depth = est.infer(frames[i]);
    ASSERT_FALSE(depth.empty());
    EXPECT_EQ(depth.type(), CV_32F);
    EXPECT_EQ(depth.size(), frames[i].size());
    
    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    EXPECT_TRUE(std::isfinite(minVal));
    EXPECT_TRUE(std::isfinite(maxVal));
  }
}

TEST(MLDepthEstimator, CacheHitMissScenarios) {
  const std::string mp = model_path();
  if (!std::ifstream(mp).good()) GTEST_SKIP() << "Model not found: " << mp;

  MLDepthEstimator est(mp, 256, 256, false);
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
  
  est.set_frame_id(0);
  float d1 = est.get_depth(frame, cv::Rect(100, 100, 50, 50));
  EXPECT_TRUE(std::isfinite(d1));
  
  float d2 = est.get_depth(frame, cv::Rect(200, 200, 50, 50));
  EXPECT_TRUE(std::isfinite(d2));
  
  float d3 = est.get_depth(frame, cv::Rect(300, 300, 30, 30));
  EXPECT_TRUE(std::isfinite(d3));
  
  est.set_frame_id(1);
  float d4 = est.get_depth(frame, cv::Rect(150, 150, 50, 50));
  EXPECT_TRUE(std::isfinite(d4));
  
  cv::Mat cached = est.get_cached_depth_map();
  EXPECT_FALSE(cached.empty());
  EXPECT_EQ(cached.type(), CV_32F);
}

TEST(MLDepthEstimator, BoundaryConditions) {
  const std::string mp = model_path();
  if (!std::ifstream(mp).good()) GTEST_SKIP() << "Model not found: " << mp;

  MLDepthEstimator est(mp, 256, 256, false);
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
  
  est.set_frame_id(2);
  
  float d1 = est.get_depth(frame, cv::Rect(5000, 5000, 100, 100));
  EXPECT_TRUE(std::isfinite(d1));
  
  float d2 = est.get_depth(frame, cv::Rect(-500, -500, 100, 100));
  EXPECT_TRUE(std::isfinite(d2));
  
  float d3 = est.get_depth(frame, cv::Rect(600, 450, 100, 100));
  EXPECT_TRUE(std::isfinite(d3));
  
  float d4 = est.get_depth(frame, cv::Rect(0, 0, 10, 10));
  EXPECT_TRUE(std::isfinite(d4));
  
  float d5 = est.get_depth(frame, cv::Rect(630, 470, 10, 10));
  EXPECT_TRUE(std::isfinite(d5));
  
  float d6 = est.get_depth(frame, cv::Rect(320, 240, 1, 1));
  EXPECT_TRUE(std::isfinite(d6));
}

TEST(MLDepthEstimator, EmptyFrameHandling) {
  const std::string mp = model_path();
  if (!std::ifstream(mp).good()) GTEST_SKIP() << "Model not found: " << mp;

  MLDepthEstimator est(mp, 256, 256, false);
  cv::Mat empty_frame;
  
  est.set_frame_id(999);
  float d = est.get_depth(empty_frame, cv::Rect(10, 10, 20, 20));
  EXPECT_EQ(d, 1.0f);
}

TEST(MLDepthEstimator, NormalizeDepthEdgeCases) {
  cv::Mat empty;
  cv::Mat norm_empty = MLDepthEstimator::normalizeDepth(empty);
  EXPECT_TRUE(norm_empty.empty());
  
  cv::Mat zeros = cv::Mat::zeros(100, 100, CV_32F);
  cv::Mat norm_zeros = MLDepthEstimator::normalizeDepth(zeros);
  EXPECT_EQ(norm_zeros.type(), CV_8U);
  
  cv::Mat uniform = cv::Mat::ones(50, 50, CV_32F) * 7.5f;
  cv::Mat norm_uniform = MLDepthEstimator::normalizeDepth(uniform);
  EXPECT_EQ(norm_uniform.type(), CV_8U);
  
  cv::Mat normal = (cv::Mat_<float>(4, 4) << 
    0.1f, 0.3f, 0.5f, 0.7f,
    0.9f, 1.1f, 1.3f, 1.5f,
    1.7f, 1.9f, 2.1f, 2.3f,
    2.5f, 2.7f, 2.9f, 3.1f);
  cv::Mat norm_normal = MLDepthEstimator::normalizeDepth(normal);
  EXPECT_EQ(norm_normal.type(), CV_8U);
  
  cv::Mat negative = (cv::Mat_<float>(3, 3) 
    -10.f, -5.f, 0.f,
    5.f, 10.f, 15.f,
    20.f, 25.f, 30.f);
  cv::Mat norm_neg = MLDepthEstimator::normalizeDepth(negative);
  EXPECT_EQ(norm_neg.type(), CV_8U);
}

TEST(MLDepthEstimator, SetFrameIdVariations) {
  const std::string mp = model_path();
  if (!std::ifstream(mp).good()) GTEST_SKIP() << "Model not found: " << mp;

  MLDepthEstimator est(mp, 256, 256, false);
  
  std::vector<int> frame_ids = {0, 1, 5, 10, 42, 100, -1, -10};
  for (int fid : frame_ids) {
    est.set_frame_id(fid);
  }
  
  SUCCEED();
}
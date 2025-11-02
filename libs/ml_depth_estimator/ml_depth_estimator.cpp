#include "ml_depth_estimator.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace perception;

MLDepthEstimator::MLDepthEstimator(const std::string& path, int w, int h, bool use_gpu)
    : input_w_(w), input_h_(h),
      env_(ORT_LOGGING_LEVEL_WARNING, "DepthEstimator"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (use_gpu) {
#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
#endif
  }

  session_ = std::make_unique<Ort::Session>(env_, path.c_str(), session_options_);
}

Mat MLDepthEstimator::infer(const Mat& bgr) {
  // Pre-process
  Mat resized, rgb;
  cv::resize(bgr, resized, Size(input_w_, input_h_));
  cv::cvtColor(resized, rgb, COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // Normalize using ImageNet stats
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> std = {0.229f, 0.224f, 0.225f};
  
  std::vector<Mat> channels(3);
  cv::split(rgb, channels);
  
  for (int i = 0; i < 3; i++) {
    channels[i] = (channels[i] - mean[i]) / std[i];
  }
  
  cv::merge(channels, rgb);

  // Convert HWC to CHW format
  std::vector<float> input_data;
  input_data.reserve(3 * input_h_ * input_w_);
  
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < input_h_; h++) {
      for (int w = 0; w < input_w_; w++) {
        input_data.push_back(channels[c].at<float>(h, w));
      }
    }
  }

  // Create tensor
  std::array<int64_t, 4> input_shape = {1, 3, input_h_, input_w_};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  // Run model
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session_->GetInputNameAllocated(0, allocator);
  auto output_name = session_->GetOutputNameAllocated(0, allocator);

  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};

  auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                      input_names, &input_tensor, 1,
                                      output_names, 1);

  float* output_data = output_tensors.front().GetTensorMutableData<float>();

  // Convert output to cv::Mat
  // NOTE: models usually output (1,1,H,W) or (1,H,W)
  size_t out_h = input_h_;
  size_t out_w = input_w_;
  Mat depth(out_h, out_w, CV_32F, output_data);
  Mat result = depth.clone();  // clone to detach from ORT memory
  
  // Resize back to original size
  Mat final_depth;
  cv::resize(result, final_depth, bgr.size(), 0, 0, cv::INTER_LINEAR);
  
  return final_depth;
}


Mat MLDepthEstimator::normalizeDepth(const Mat& d) {
  Mat norm;
  if (d.empty()) return Mat();
  cv::normalize(d, norm, 0, 255, NORM_MINMAX, CV_8U);
  return norm;
}

float MLDepthEstimator::get_depth(const Mat& frame, Rect bbox) {
  // Get the center of the bounding box
  cv::Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
  
  // Use cached depth map if available
  if (cached_depth_map_.empty() || cached_depth_frame_id_ != current_frame_id_) {
    cached_depth_map_ = infer(frame);
    cached_depth_frame_id_ = current_frame_id_;
  }
  
  if (cached_depth_map_.empty()) {
    return 1.0f;  // Default depth of 1 meter
  }
  
  // Ensure coordinates are within bounds
  int center_x = std::max(0, std::min(static_cast<int>(center.x), cached_depth_map_.cols - 1));
  int center_y = std::max(0, std::min(static_cast<int>(center.y), cached_depth_map_.rows - 1));
  
  // Extract the raw depth value at the center
  float raw_depth = cached_depth_map_.at<float>(center_y, center_x);
  
  // Return raw depth value from Depth Anything V2
  return std::abs(raw_depth);
}

void MLDepthEstimator::set_frame_id(int frame_id) {
  current_frame_id_ = frame_id;
}

Mat MLDepthEstimator::get_cached_depth_map() const {
  return cached_depth_map_;
}

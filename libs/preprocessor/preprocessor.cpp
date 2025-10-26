#include "preprocessor.hpp"

#include <opencv2/dnn.hpp>

using namespace cv;
using namespace perception;

Preprocessor::Preprocessor(int input_w, int input_h, bool swap_rb)
    : input_w_(input_w), input_h_(input_h), swap_rb_(swap_rb) {}

Mat Preprocessor::process(const Mat& frame) {
  // Create a blob from the image. This function will resize, pad (scale=1.0/255),
  // swap RB channels, and not crop.
  return cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(input_w_, input_h_), 
                                  cv::Scalar(0, 0, 0), swap_rb_, false);
}

Mat Preprocessor::makeBlob(const Mat& img) const {
  // TODO: Create normalized blob for DNN
  Mat blob = Mat::zeros(1, 1, CV_32F);
  return blob;
}

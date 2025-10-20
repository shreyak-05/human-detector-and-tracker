#include "preprocessor.hpp"
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace perception;

Preprocessor::Preprocessor(int input_w, int input_h, bool swap_rb)
    : input_w_(input_w), input_h_(input_h), swap_rb_(swap_rb) {}

Mat Preprocessor::makeBlob(const Mat& img) const {
  // TODO: Create normalized blob for DNN
  Mat blob = Mat::zeros(1, 1, CV_32F);
  return blob;
}

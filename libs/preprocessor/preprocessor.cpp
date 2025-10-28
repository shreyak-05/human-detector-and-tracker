/*
 * Copyright 2025 Shreya Kalyanaraman and Tirth Sadaria
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

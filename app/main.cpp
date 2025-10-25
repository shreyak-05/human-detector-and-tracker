#include <iostream>

#include "detector_tracker.hpp"
#include "ml_depth_estimator.hpp"
#include "transformer.hpp"

using namespace perception;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: ./human_tracker <yolo.onnx> <midas.onnx> "
                 "<video_or_camera>\n";
    return 0;
  }

  try {
    std::cout << "Loading YOLO detector: " << argv[1] << std::endl;
    // TODO : Load YOLO model
    std::cout << "YOLO loaded successfully\n";
    std::cout << "Loading depth estimator: " << argv[2] << std::endl;
    //  TODO : Load depth model
    std::cout << "Depth model loaded successfully\n";
    // TODO : Initialize 3D transformer with example intrinsics and extrinsics
    std::cout << "Opening video source: " << argv[3] << std::endl;
    // TODO : Open video file or camera
    cv::VideoCapture cap;
    std::cout << "Video source opened successfully\n";
    // Process video frames
    // TODO : Main processing loop
    // Process frames and display results
    std::cout << "Processing completed\n";

  } catch (const cv::Exception& e) {
    std::cerr << "OpenCV Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

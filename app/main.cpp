#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include "detector_tracker.hpp"
#include "ml_depth_estimator.hpp"
#include "onnx_network.hpp"
#include "preprocessor.hpp"
#include "transformer.hpp"

using namespace cv;
using namespace perception;

int main(int argc, char** argv) {
  // Define model paths and constants
  const std::string detector_path = "models/yolov8n.onnx";
  const std::string depth_path = "models/depth_anything_small.onnx";
  const Size yolo_input_size(640, 640);
  
  // IMPORTANT: Replace with your REAL camera matrix
  Mat camera_matrix = (Mat_<double>(3, 3) << 500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1.0);
  
  try {
    std::cout << "Loading YOLO detector: " << detector_path << std::endl;
    
    // Instantiate all concrete classes using std::make_shared
    auto preprocessor = std::make_shared<Preprocessor>(yolo_input_size.width, yolo_input_size.height);
    auto detector_network = std::make_shared<OnnxNetwork>(detector_path);
    auto depth_estimator = std::make_shared<MLDepthEstimator>(depth_path);
    auto transformer = std::make_shared<Transformer3D>(camera_matrix);
    
    std::cout << "Creating HumanDetector with injected dependencies..." << std::endl;
    
    // Create the HumanDetector by injecting all dependencies
    DetectorTracker detector(preprocessor, detector_network, depth_estimator, transformer, 0.5f);
    
    std::cout << "Opening video source (webcam 0)..." << std::endl;
    
    // Open a video source (try a webcam 0 or replace with a video file path "my_video.mp4")
    VideoCapture cap(0);
    if (!cap.isOpened()) {
      std::cerr << "Error: Cannot open video source" << std::endl;
      return -1;
    }
    
    std::cout << "Video source opened successfully. Press 'q' to quit." << std::endl;
    
    Mat frame;
    int frame_count = 0;
    
    // Create the main while loop
    while (cap.read(frame)) {
      frame_count++;
      std::cout << "Processing frame " << frame_count << std::endl;
      
      // Run the full 3D detection pipeline
      auto positions = detector.get_3d_positions(frame);
      
      // Draw the results on the frame
      for (size_t i = 0; i < positions.size(); i++) {
        const auto& det = positions[i];
        
        // Display 3D position
        std::cout << "Human " << i << " at: (" << det.position.x << ", " 
                  << det.position.y << ", " << det.position.z << ")" << std::endl;
        
        // Draw bounding box
        cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
        
        // Add text overlay
        std::string text = cv::format("ID %d: (%.1fm, %.1fm, %.1fm)",
                                       det.detection_id, det.position.x, det.position.y, det.position.z);
        
        // Position text above the bounding box
        cv::Point text_origin(det.bbox.x, det.bbox.y - 10);
        cv::putText(frame, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
      }
      
      cv::imshow("Human Detector Demo", frame);
      if (cv::waitKey(1) == 'q') break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Processing completed" << std::endl;
    
  } catch (const cv::Exception& e) {
    std::cerr << "OpenCV Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}

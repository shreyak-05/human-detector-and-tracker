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

#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cstdlib>

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
  // Note: Run from project root directory
  const std::string detector_path = "models/yolov8n.onnx";
  const std::string depth_path = "models/depth_anything_v2_vits.onnx";
  const Size yolo_input_size(640, 640);
  
  // IMPORTANT: Replace with your REAL camera matrix
  Mat camera_matrix = (Mat_<double>(3, 3) << 500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1.0);
  
  // Parse command-line arguments
  enum InputMode { CAMERA, VIDEO, IMAGE };
  InputMode mode = CAMERA;
  std::string input_path;
  
  if (argc > 1) {
    std::string arg = std::string(argv[1]);
    
    if (arg == "camera" || arg == "cam") {
      mode = CAMERA;
      std::cout << "Mode: Camera" << std::endl;
    } else if (arg == "test_video") {
      mode = VIDEO;
      input_path = "models/test_video.mp4";
      std::cout << "Mode: Test Video (" << input_path << ")" << std::endl;
    } else if (arg == "test_image") {
      mode = IMAGE;
      input_path = "models/test_image.jpg";  // You can add a test image
      std::cout << "Mode: Test Image (" << input_path << ")" << std::endl;
    } else {
      // Check if it's a video or image by extension
      std::string ext = arg.substr(arg.find_last_of(".") + 1);
      if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
        mode = IMAGE;
        input_path = arg;
        std::cout << "Mode: Image File" << std::endl;
      } else {
        mode = VIDEO;
        input_path = arg;
        std::cout << "Mode: Video File" << std::endl;
      }
    }
  } else {
    std::cout << "Usage: " << argv[0] << " [mode]" << std::endl;
    std::cout << "\nAvailable modes:" << std::endl;
    std::cout << "  1. camera       - Use webcam (default if no argument)" << std::endl;
    std::cout << "  2. test_video   - Use models/test_video.mp4" << std::endl;
    std::cout << "  3. test_image   - Use models/test_image.jpg" << std::endl;
    std::cout << "  4. <video_path> - Custom video file" << std::endl;
    std::cout << "  5. <image_path> - Custom image file" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << argv[0] << " camera              # Use webcam" << std::endl;
    std::cout << "  " << argv[0] << " test_video          # Use test video" << std::endl;
    std::cout << "  " << argv[0] << " path/to/image.jpg   # Process image" << std::endl;
    std::cout << "  " << argv[0] << " path/to/video.mp4   # Process video" << std::endl;
    std::cout << "\nDefaulting to camera mode..." << std::endl;
  }
  
  try {
    // Instantiate all concrete classes using std::make_shared
    auto preprocessor = std::make_shared<Preprocessor>(yolo_input_size.width, yolo_input_size.height);
    auto detector_network = std::make_shared<OnnxNetwork>(detector_path);
    auto depth_estimator = std::make_shared<MLDepthEstimator>(depth_path, 518, 518, false);
    auto transformer = std::make_shared<Transformer3D>(camera_matrix);
    
    // Create the HumanDetector by injecting all dependencies
    DetectorTracker detector(preprocessor, detector_network, depth_estimator, transformer, 0.5f);
    
    std::cout << "Human detector loaded successfully." << std::endl;
    
    // Determine input source based on mode
    Mat frame;
    int frame_count = 0;
    
    if (mode == IMAGE) {
      // Load image from file
      std::cout << "Loading image from: " << input_path << std::endl;
      frame = cv::imread(input_path);
      
      if (frame.empty()) {
        std::cerr << "Error: Cannot open image file: " << input_path << std::endl;
        return -1;
      }
      
      std::cout << "Image loaded successfully. Press 'q' to quit." << std::endl;
      
      std::cout << "=== Processing Image ===" << std::endl;
      
      // Run the full 3D detection pipeline
      auto positions = detector.get_3d_positions(frame);
      
      // Draw the results on the frame
      for (size_t i = 0; i < positions.size(); i++) {
        const auto& det = positions[i];
        
        // Display 3D position in structured format
        std::cout << "Human " << i << ": 3D position (" << std::fixed << std::setprecision(2) 
                  << det.position.x << ", " << det.position.y << ", " << det.position.z << ") meters" << std::endl;
        
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
      
      // Ensure results directory exists
      system("mkdir -p results");
      
      // Save the annotated image
      std::string output_path = "results/output_detected.jpg";
      cv::imwrite(output_path, frame);
      std::cout << "Saved annotated image to: " << output_path << std::endl;
      
      // Save detection data to text file
      std::ofstream output_file("results/output_detections.txt");
      output_file << "Detection Results\n";
      output_file << "=================\n";
      output_file << "Input: " << input_path << "\n";
      output_file << "Detections: " << positions.size() << "\n\n";
      
      for (size_t i = 0; i < positions.size(); i++) {
        const auto& det = positions[i];
        output_file << "Human " << i << ":\n";
        output_file << "  Detection ID: " << det.detection_id << "\n";
        output_file << "  Bounding Box: (" << det.bbox.x << ", " << det.bbox.y 
                    << ", " << det.bbox.width << "x" << det.bbox.height << ")\n";
        output_file << "  3D Position: (" << det.position.x << ", " 
                    << det.position.y << ", " << det.position.z << ") meters\n\n";
      }
      output_file.close();
      std::cout << "Saved detection data to: results/output_detections.txt" << std::endl;
      
      cv::waitKey(0);  // Wait for key press to close
      
    } else {
      // Handle camera or video
      VideoCapture cap;
      
      if (mode == CAMERA) {
        std::cout << "Opening camera (webcam 0)..." << std::endl;
        cap.open(0);
      } else {  // VIDEO
        std::cout << "Opening video file: " << input_path << std::endl;
        cap.open(input_path);
      }
      
      if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video source" << std::endl;
        return -1;
      }
      
      std::cout << "Video source opened successfully. Press 'q' to quit." << std::endl;
      
      // Ensure results directory exists
      system("mkdir -p results");
      
      // Set up video writer for output
      VideoWriter video_writer;
      bool save_output = (mode == VIDEO);  // Save video for VIDEO mode only
      int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      double fps = cap.get(cv::CAP_PROP_FPS);
      cv::Size frame_size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), 
                         (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      
      if (save_output) {
        std::string output_video = "results/output_detected.mp4";
        video_writer.open(output_video, fourcc, fps, frame_size);
        std::cout << "Saving output video to: " << output_video << std::endl;
      }
      
      // Process video frames
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
        
        // Write frame to output video
        if (save_output && video_writer.isOpened()) {
          video_writer.write(frame);
        }
        
        if (cv::waitKey(1) == 'q') break;
      }
      
      if (save_output && video_writer.isOpened()) {
        video_writer.release();
        std::cout << "Output video saved successfully!" << std::endl;
      }
      
      cap.release();
    }
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

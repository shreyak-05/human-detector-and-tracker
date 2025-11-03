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

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "detector_tracker.hpp"
#include "ml_depth_estimator.hpp"
#include "onnx_network.hpp"
#include "preprocessor.hpp"
#include "transformer.hpp"

using namespace cv;
using namespace perception;

int main(int argc, char** argv) {
  // Configuration
  const std::string detector_path = "models/yolov8n.onnx";
  const std::string depth_path = "models/depth_anything_v2_vits.onnx";
  const Size yolo_input_size(640, 640);
  Mat camera_matrix =
      (Mat_<double>(3, 3) << 500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1.0);

  // Input mode
  enum InputMode { CAMERA, VIDEO, IMAGE };
  InputMode mode = CAMERA;
  std::string input_path;

  // Parse arguments
  if (argc > 1) {
    std::string arg = argv[1];
    if (arg == "test_video") {
      mode = VIDEO;
      input_path = "models/test_video.mp4";
    } else if (arg == "test_image") {
      mode = IMAGE;
      input_path = "models/test_image.jpg";
    } else if (arg.find('.') != std::string::npos) {
      std::string ext = arg.substr(arg.find_last_of(".") + 1);
      mode = (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp")
                 ? IMAGE
                 : VIDEO;
      input_path = arg;
    }
  }

  try {
    // Initialize components
    auto preprocessor = std::make_shared<Preprocessor>(yolo_input_size.width,
                                                       yolo_input_size.height);
    auto detector_network = std::make_shared<OnnxNetwork>(detector_path);
    auto depth_estimator =
        std::make_shared<MLDepthEstimator>(depth_path, 518, 518, false);
    auto transformer = std::make_shared<Transformer3D>(camera_matrix);
    DetectorTracker detector(preprocessor, detector_network, depth_estimator,
                             transformer, 0.5f);

    if (mode == IMAGE) {
      // === IMAGE MODE ===
      Mat frame = cv::imread(input_path);
      if (frame.empty()) {
        std::cerr << "Error: Cannot open image file: " << input_path
                  << std::endl;
        return -1;
      }

      auto positions = detector.get_3d_positions(frame);

      // Draw results
      for (const auto& det : positions) {
        cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::string text =
            cv::format("ID %d: (%.1fm, %.1fm, %.1fm)", det.detection_id,
                       det.position.x, det.position.y, det.position.z);
        cv::Point text_origin(det.bbox.x, det.bbox.y - 10);
        cv::putText(frame, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
      }

      cv::imshow("Human Detector", frame);
      cv::waitKey(0);

    } else {
      // === VIDEO/CAMERA MODE ===
      // Setup capture
      VideoCapture cap;
      cap.open(mode == CAMERA ? 0 : input_path);
      if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video source" << std::endl;
        return -1;
      }

      // Setup output video writer
      VideoWriter video_writer;
      if (mode == VIDEO) {
        system("mkdir -p results");
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frame_size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                            (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        video_writer.open("results/output_detected.mp4", fourcc, fps,
                          frame_size);
      }

      // Optimization settings
      const int depth_skip_interval = 30;     // Run depth every 30 frames
      const int detection_skip_interval = 3;  // Run detection every 3 frames
      const cv::Point3f default_position(0, 0, 2.0f);
      const float iou_threshold = 0.3f;  // IoU threshold for track association

      // State variables
      std::vector<Detection3D> last_positions;
      std::vector<Detection> cached_detections;
      std::vector<Track> active_tracks;  // Persistent tracks with IoU matching
      int next_track_id = 1;
      auto start_time = std::chrono::high_resolution_clock::now();
      int frame_count = 0;

      // Main processing loop
      Mat frame;
      while (cap.read(frame)) {
        frame_count++;

        // Performance monitoring
        if (frame_count % 30 == 0) {
          auto current_time = std::chrono::high_resolution_clock::now();
          auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             current_time - start_time)
                             .count();
          float fps = frame_count / (float)std::max(1, (int)elapsed);
          std::cout << "Frame " << frame_count << " (FPS: " << std::fixed
                    << std::setprecision(1) << fps << ")" << std::endl;
        }

        std::vector<Detection3D> positions;

        // Processing strategy: depth + tracking
        if (frame_count % depth_skip_interval == 1) {
          // Full processing: detection + depth + tracking
          depth_estimator->set_frame_id(frame_count);
          auto detections = detector.detect(frame);
          cached_detections = detections;

          // Update tracks with IoU matching
          std::vector<bool> detection_matched(detections.size(), false);
          std::vector<bool> track_matched(active_tracks.size(), false);

          // Match detections to existing tracks using IoU
          for (size_t i = 0; i < detections.size(); i++) {
            float best_iou = 0.0f;
            int best_track_idx = -1;

            for (size_t j = 0; j < active_tracks.size(); j++) {
              if (track_matched[j]) continue;

              float current_iou =
                  detector.iou(detections[i].box, active_tracks[j].det.box);
              if (current_iou > best_iou && current_iou > iou_threshold) {
                best_iou = current_iou;
                best_track_idx = static_cast<int>(j);
              }
            }

            if (best_track_idx >= 0) {
              // Update existing track
              active_tracks[best_track_idx].det = detections[i];
              active_tracks[best_track_idx].age++;
              active_tracks[best_track_idx].time_since_update = 0;
              detection_matched[i] = true;
              track_matched[best_track_idx] = true;
            }
          }

          // Create new tracks for unmatched detections
          for (size_t i = 0; i < detections.size(); i++) {
            if (!detection_matched[i]) {
              Track new_track;
              new_track.id = next_track_id++;
              new_track.det = detections[i];
              new_track.age = 1;
              new_track.time_since_update = 0;
              active_tracks.push_back(new_track);
            }
          }

          // Remove old tracks
          active_tracks.erase(
              std::remove_if(active_tracks.begin(), active_tracks.end(),
                             [](Track& track) {
                               track.time_since_update++;
                               return track.time_since_update >
                                      30;  // Remove after 30 frames
                             }),
              active_tracks.end());

          // Create 3D positions
          if (!active_tracks.empty()) {
            for (const auto& track : active_tracks) {
              float depth = depth_estimator->get_depth(frame, track.det.box);
              cv::Point2f center_pixel(
                  track.det.box.x + track.det.box.width / 2.0f,
                  track.det.box.y + track.det.box.height / 2.0f);
              cv::Point3f pos = transformer->project_to_3d(center_pixel, depth);
              positions.push_back({track.id, track.det.box, pos});
            }
            last_positions = positions;
          }
        } else if (frame_count % detection_skip_interval == 1) {
          // Detection only with track update
          auto detections = detector.detect(frame);
          cached_detections = detections;

          // Quick track update using IoU
          for (auto& track : active_tracks) {
            track.time_since_update++;

            // Find best matching detection
            float best_iou = 0.0f;
            int best_det_idx = -1;
            for (size_t i = 0; i < detections.size(); i++) {
              float current_iou =
                  detector.iou(detections[i].box, track.det.box);
              if (current_iou > best_iou && current_iou > iou_threshold) {
                best_iou = current_iou;
                best_det_idx = static_cast<int>(i);
              }
            }

            if (best_det_idx >= 0) {
              track.det = detections[best_det_idx];
              track.time_since_update = 0;
            }
          }

          // Generate positions from updated tracks
          if (!active_tracks.empty() && !last_positions.empty()) {
            for (const auto& track : active_tracks) {
              if (track.time_since_update == 0) {  // Only recently updated
                // Use cached depth
                cv::Point3f pos = last_positions.empty()
                                      ? default_position
                                      : last_positions[0].position;
                positions.push_back({track.id, track.det.box, pos});
              }
            }
          }
        } else {
          // Ultra-fast mode: reuse track data
          if (!active_tracks.empty()) {
            for (const auto& track : active_tracks) {
              if (track.time_since_update < 5) {  // Only recent tracks
                cv::Point3f pos = last_positions.empty()
                                      ? default_position
                                      : last_positions[0].position;
                positions.push_back({track.id, track.det.box, pos});
              }
            }
          }
        }

        // Draw results
        for (const auto& det : positions) {
          cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
          std::string text =
              cv::format("ID %d: (%.1fm, %.1fm, %.1fm)", det.detection_id,
                         det.position.x, det.position.y, det.position.z);
          cv::Point text_origin(det.bbox.x, det.bbox.y - 10);
          cv::putText(frame, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                      cv::Scalar(0, 255, 0), 2);
        }

        // Display and save
        cv::imshow("Human Detector", frame);
        if (video_writer.isOpened()) video_writer.write(frame);
        if (cv::waitKey(1) == 'q') break;
      }

      // Cleanup
      cap.release();
      if (video_writer.isOpened()) video_writer.release();
    }
    cv::destroyAllWindows();

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

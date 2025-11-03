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

/**
 * @file perception_types.hpp
 * @brief Common data structures for human detection and tracking
 * @author Shreya Kalyanaraman
 * @author Tirth Sadaria
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace perception {

/**
 * @brief Detected human object from object detection network.
 *
 * Represents a single human detection from YOLOv8 inference,
 * containing bounding box coordinates, confidence score, and class ID.
 * Used as intermediate data structure before 3D position estimation.
 */
struct Detection {
  cv::Rect box;            ///< Bounding box in pixel coordinates (x, y, width, height)
  float confidence{0.0f};   ///< Detection confidence score [0.0, 1.0]
  int class_id{0};         ///< Class ID (0 = person in COCO dataset)
};

/**
 * @brief Tracked human with persistent ID across frames.
 *
 * Used for multi-object tracking to maintain identity of detected humans
 * across video frames. Includes tracking metadata such as age (frame count)
 * and time since last update for track management.
 */
struct Track {
  int id{0};                 ///< Unique track ID assigned during tracking
  Detection det;             ///< Current detection associated with this track
  int age{0};                ///< Total number of frames this track has been active
  int time_since_update{0};  ///< Frames elapsed since last successful association
};

/**
 * @brief 3D point in robot coordinate frame.
 *
 * Represents a point in 3D space using robot frame coordinates.
 * Used for navigation and path planning in autonomous robotics.
 */
struct GroundPoint {
  float X{0.0f};  ///< X coordinate in meters (forward/backward relative to robot)
  float Y{0.0f};  ///< Y coordinate in meters (left/right relative to robot)
  float Z{0.0f};  ///< Z coordinate in meters (height relative to robot)
};

/**
 * @brief 3D detection with position and 2D bounding box.
 *
 * Complete detection result combining 2D bounding box with estimated 3D position.
 * This is the final output of the detection pipeline, containing both visual
 * (bbox) and spatial (position) information for each detected human.
 */
struct Detection3D {
  int detection_id;      ///< Unique detection/track ID for this human
  cv::Rect bbox;         ///< 2D bounding box in pixel coordinates
  cv::Point3f position;  ///< 3D position in camera/robot frame (x, y, z in meters)
};

}  // namespace perception

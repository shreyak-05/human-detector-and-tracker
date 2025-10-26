#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace perception {

/**
 * @brief Detected human object.
 */
struct Detection {
  cv::Rect box;            ///< Bounding box (x, y, width, height)
  float confidence{0.0f};  ///< Confidence score [0.0, 1.0]
  int class_id{0};         ///< Class ID (0 = person)
};

/**
 * @brief Tracked human with persistent ID.
 */
struct Track {
  int id{0};                 ///< Unique track ID
  Detection det;             ///< Current detection
  int age{0};                ///< Total frames tracked
  int time_since_update{0};  ///< Frames since last update
};

/**
 * @brief 3D point in robot coordinates.
 */
struct GroundPoint {
  float X{0.0f};  ///< X coordinate (meters)
  float Y{0.0f};  ///< Y coordinate (meters)
  float Z{0.0f};  ///< Z coordinate (meters)
};

/**
 * @brief 3D detection with position.
 */
struct Detection3D {
  int detection_id;    ///< Unique detection ID
  cv::Point3f position; ///< 3D position in camera/robot frame
};

}  // namespace perception

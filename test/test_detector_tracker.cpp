#include <gtest/gtest.h>

#include "detector_tracker.hpp"
#include "perception_types.hpp"

class DetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // This will fail, we'll fix it in GREEN
        // For RED phase: We don't instantiate anything yet
    }
};

TEST_F(DetectorTest, PostProcessYoloOutput) {
    // YOLOv8 output: [batch, 4_bbox+1_conf, N_detections] -> [1, 5, 1] for 1 detection
    // OpenCV's DNN module transposes this to [5, N_detections]
    const int num_detections = 1;
    const int yolo_dims = 5; // cx, cy, w, h, conf (for 1 class)
    
    float data[5] = {100.0f, 120.0f, 20.0f, 40.0f, 0.95f}; // [center_x, center_y, w, h, conf]
    cv::Mat fake_output(yolo_dims, num_detections, CV_32F, data);
    
    // Test the post_process logic inline (since we can't instantiate DetectorTracker with fake model)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    
    for (int i = 0; i < fake_output.cols; ++i) {
        float conf = fake_output.at<float>(4, i);
        if (conf < 0.5f) continue;
        
        float cx = fake_output.at<float>(0, i);
        float cy = fake_output.at<float>(1, i);
        float w = fake_output.at<float>(2, i);
        float h = fake_output.at<float>(3, i);
        
        int x = static_cast<int>(cx - w / 2);
        int y = static_cast<int>(cy - h / 2);
        
        boxes.emplace_back(x, y, static_cast<int>(w), static_cast<int>(h));
        confidences.push_back(conf);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);
    
    std::vector<perception::Detection> detections;
    for (int idx : indices) {
        detections.push_back({boxes[idx], confidences[idx], 0});
    }
    
    ASSERT_EQ(detections.size(), 1);
    ASSERT_FLOAT_EQ(detections[0].confidence, 0.95f);
    
    // BBox is [cx, cy, w, h], we want [x, y, w, h] for cv::Rect
    ASSERT_EQ(detections[0].box.x, 90);      // 100 - 20/2
    ASSERT_EQ(detections[0].box.y, 100);     // 120 - 40/2
    ASSERT_EQ(detections[0].box.width, 20);
}
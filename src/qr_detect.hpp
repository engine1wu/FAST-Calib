/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef QR_DETECT_HPP
#define QR_DETECT_HPP

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include "common_lib.h"

class QRDetect 
{
  private:
    double marker_size_, delta_width_qr_center_, delta_height_qr_center_;
    double delta_width_circles_, delta_height_circles_;
    int min_detected_markers_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    std::shared_ptr<Camera> camera_;

    bool projectPoint(const cv::Point3d& point_cam, cv::Point2f& uv) const
    {
      cv::Point2d pixel;
      if (!camera_->project(point_cam, pixel)) return false;
      uv = cv::Point2f(static_cast<float>(pixel.x), static_cast<float>(pixel.y));
      return true;
    }

    cv::Point3d transformBoardPoint(const cv::Matx33d& R, const cv::Vec3d& t,
                                    const cv::Point3f& point) const
    {
      cv::Vec3d p(point.x, point.y, point.z);
      cv::Vec3d q = R * p + t;
      return cv::Point3d(q[0], q[1], q[2]);
    }

    void drawAxis(const cv::Vec3d& rvec, const cv::Vec3d& tvec, double length)
    {
      cv::Mat Rmat;
      cv::Rodrigues(rvec, Rmat);
      cv::Matx33d R;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R(r, c) = Rmat.at<double>(r, c);
      }

      cv::Point2f origin, x_axis, y_axis, z_axis;
      if (!projectPoint(transformBoardPoint(R, tvec, cv::Point3f(0, 0, 0)), origin)) return;
      if (projectPoint(transformBoardPoint(R, tvec, cv::Point3f(length, 0, 0)), x_axis)) {
        cv::line(imageCopy_, origin, x_axis, cv::Scalar(0, 0, 255), 2);
      }
      if (projectPoint(transformBoardPoint(R, tvec, cv::Point3f(0, length, 0)), y_axis)) {
        cv::line(imageCopy_, origin, y_axis, cv::Scalar(0, 255, 0), 2);
      }
      if (projectPoint(transformBoardPoint(R, tvec, cv::Point3f(0, 0, length)), z_axis)) {
        cv::line(imageCopy_, origin, z_axis, cv::Scalar(255, 0, 0), 2);
      }
    }

    bool collectNormalizedCorners(const std::vector<std::vector<cv::Point3f>>& boardCorners,
                                  const std::vector<int>& boardIds,
                                  const std::vector<std::vector<cv::Point2f>>& corners,
                                  const std::vector<int>& ids,
                                  std::vector<cv::Point3f>& objectPoints,
                                  std::vector<cv::Point2f>& imagePoints,
                                  int& usedMarkers) const
    {
      objectPoints.clear();
      imagePoints.clear();
      usedMarkers = 0;

      for (size_t i = 0; i < ids.size(); ++i) {
        auto board_it = std::find(boardIds.begin(), boardIds.end(), ids[i]);
        if (board_it == boardIds.end() || corners[i].size() != 4) continue;

        std::array<cv::Point2d, 4> normalized;
        bool ok = true;
        for (int j = 0; j < 4; ++j) {
          ok = camera_->pixelToNormalized(corners[i][j], normalized[j]);
          if (!ok) break;
        }
        if (!ok) {
          ROS_WARN("[Mono] Marker %d has corner(s) outside normalized plane, skipping it.", ids[i]);
          continue;
        }

        const size_t board_idx = std::distance(boardIds.begin(), board_it);
        for (int j = 0; j < 4; ++j) {
          objectPoints.push_back(boardCorners[board_idx][j]);
          imagePoints.emplace_back(static_cast<float>(normalized[j].x),
                                   static_cast<float>(normalized[j].y));
        }
        ++usedMarkers;
      }

      return usedMarkers >= min_detected_markers_ && objectPoints.size() >= 4;
    }
  
  public:
    ros::Publisher qr_pub_;
    cv::Mat imageCopy_;

    QRDetect(ros::NodeHandle &nh, Params& params) 
    {
      marker_size_ = params.marker_size;
      delta_width_qr_center_ = params.delta_width_qr_center;
      delta_height_qr_center_ = params.delta_height_qr_center;
      delta_width_circles_ = params.delta_width_circles;
      delta_height_circles_ = params.delta_height_circles;
      min_detected_markers_ = params.min_detected_markers;
      camera_ = createCamera(cameraConfigFromParams(params));

      // Initialize QR dictionary
#if CV_MAJOR_VERSION > 4 || (CV_MAJOR_VERSION == 4 && CV_MINOR_VERSION >= 7)
      dictionary_ = cv::makePtr<cv::aruco::Dictionary>(
          cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250));
#else
      dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
#endif

      qr_pub_ = nh.advertise<sensor_msgs::PointCloud2>("qr_cloud", 1);
    }

    const Camera& camera() const { return *camera_; }

    void comb(int N, int K, std::vector<std::vector<int>> &groups) {
      int upper_factorial = 1;
      int lower_factorial = 1;

      for (int i = 0; i < K; i++) {
        upper_factorial *= (N - i);
        lower_factorial *= (K - i);
      }
      int n_permutations = upper_factorial / lower_factorial;

      if (DEBUG)
        cout << N << " centers found. Iterating over " << n_permutations
            << " possible sets of candidates" << endl;

      std::string bitmask(K, 1);  // K leading 1's
      bitmask.resize(N, 0);       // N-K trailing 0's

      do {
        std::vector<int> group;
        for (int i = 0; i < N; ++i)
        {
          if (bitmask[i]) {
            group.push_back(i);
          }
        }
        groups.push_back(group);
      } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

      assert(groups.size() == n_permutations);
    }

    void detect_qr(cv::Mat &image, pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud) 
    {      
      image.copyTo(imageCopy_);

      // Markers order:
      // 0-------1
      // |       |
      // |   C   |
      // |       |
      // 3-------2
      //
      // Marker 0 -> aRuCo ID: 1
      // Marker 1 -> aRuCo ID: 2
      // Marker 2 -> aRuCo ID: 4
      // Marker 3 -> aRuCo ID: 3

      std::vector<std::vector<cv::Point3f>> boardCorners(4);
      std::vector<cv::Point3f> boardCircleCenters;
      float width = delta_width_qr_center_;
      float height = delta_height_qr_center_;
      float circle_width = delta_width_circles_ / 2.;
      float circle_height = delta_height_circles_ / 2.;
      for (int i = 0; i < 4; ++i) {
        int x_qr_center = (i % 3) == 0 ? -1 : 1;
        int y_qr_center = (i < 2) ? 1 : -1;
        float x_center = x_qr_center * width;
        float y_center = y_qr_center * height;

        boardCircleCenters.emplace_back(x_qr_center * circle_width,
                                        y_qr_center * circle_height, 0);
        for (int j = 0; j < 4; ++j) {
          int x_qr = (j % 3) == 0 ? -1 : 1;
          int y_qr = (j < 2) ? 1 : -1;
          boardCorners[i].emplace_back(x_center + x_qr * marker_size_ / 2.,
                                       y_center + y_qr * marker_size_ / 2., 0);
        }
      }

      std::vector<int> boardIds{1, 2, 4, 3};

      cv::Ptr<cv::aruco::DetectorParameters> parameters =
          cv::makePtr<cv::aruco::DetectorParameters>();

    #if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
      parameters->doCornerRefinement = true;
    #else
      parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    #endif

      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners;
      cv::aruco::detectMarkers(image, dictionary_, corners, ids, parameters);

      if (!ids.empty()) cv::aruco::drawDetectedMarkers(imageCopy_, corners, ids);

      std::vector<cv::Point3f> objectPoints;
      std::vector<cv::Point2f> normalizedImagePoints;
      int usedMarkers = 0;
      if (!collectNormalizedCorners(boardCorners, boardIds, corners, ids, objectPoints,
                                    normalizedImagePoints, usedMarkers)) {
        ROS_WARN("%d usable marker(s) found, %d expected. Skipping frame...",
                 usedMarkers, min_detected_markers_);
        return;
      }

      cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);
      cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
      if (!cv::solvePnP(objectPoints, normalizedImagePoints, identity, cv::Mat(),
                        rvec, tvec, false, cv::SOLVEPNP_IPPE)) {
        ROS_WARN("[Mono] solvePnP failed on normalized marker corners.");
        return;
      }

      drawAxis(rvec, tvec, 0.2);

      cv::Mat Rmat;
      cv::Rodrigues(rvec, Rmat);
      cv::Matx33d R;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R(r, c) = Rmat.at<double>(r, c);
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& circleCenter : boardCircleCenters) {
        cv::Point3d center3d = transformBoardPoint(R, tvec, circleCenter);

        cv::Point2f uv;
        if (projectPoint(center3d, uv)) circle(imageCopy_, uv, 5, Scalar(0, 255, 0), -1);

        pcl::PointXYZ qr_center;
        qr_center.x = center3d.x;
        qr_center.y = center3d.y;
        qr_center.z = center3d.z;
        candidates_cloud->push_back(qr_center);
      }

      std::vector<std::vector<int>> groups;
      comb(candidates_cloud->size(), TARGET_NUM_CIRCLES, groups);
      std::vector<double> groups_scores(groups.size(), -1.0);

      for (int i = 0; i < groups.size(); ++i)
      {
        std::vector<pcl::PointXYZ> candidates;
        for (int j = 0; j < groups[i].size(); ++j) {
          candidates.push_back(candidates_cloud->at(groups[i][j]));
        }

        Square square_candidate(candidates, delta_width_circles_,
                                delta_height_circles_);
        groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1;
      }

      int best_candidate_idx = -1;
      double best_candidate_score = -1;
      for (int i = 0; i < groups.size(); ++i)
      {
        if (best_candidate_score == 1 && groups_scores[i] == 1) {
          ROS_ERROR(
              "[Mono] More than one set of candidates fit target's geometry. "
              "Please, make sure your parameters are well set. Exiting callback");
          return;
        }
        if (groups_scores[i] > best_candidate_score) {
          best_candidate_score = groups_scores[i];
          best_candidate_idx = i;
        }
      }

      if (best_candidate_idx == -1)
      {
        ROS_WARN(
            "[Mono] Unable to find a candidate set that matches target's "
            "geometry");
        return;
      }

      for (int j = 0; j < groups[best_candidate_idx].size(); ++j)
      {
        centers_cloud->push_back(candidates_cloud->at(groups[best_candidate_idx][j]));
      }

      if (DEBUG)
      {
        for (int i = 0; i < centers_cloud->size(); i++) {
          cv::Point3d pt_circle(centers_cloud->at(i).x, centers_cloud->at(i).y,
                                centers_cloud->at(i).z);
          cv::Point2f uv_circle;
          if (projectPoint(pt_circle, uv_circle)) {
            circle(imageCopy_, uv_circle, 2, Scalar(255, 0, 255), -1);
          }
        }
      }
    }
};
typedef std::shared_ptr<QRDetect> QRDetectPtr;

#endif

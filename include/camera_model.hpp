#ifndef CAMERA_MODEL_HPP
#define CAMERA_MODEL_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/core.hpp>

struct CameraConfig {
  std::string camera_model = "pinhole";
  std::string distortion_model = "radtan";
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  double xi = 0.0;
  double k1 = 0.0, k2 = 0.0, k3 = 0.0, k4 = 0.0;
  double p1 = 0.0, p2 = 0.0;
};

class Camera {
 public:
  explicit Camera(CameraConfig config) : config_(std::move(config)) {}
  virtual ~Camera() = default;

  const CameraConfig& config() const { return config_; }

  bool project(const cv::Point3d& point_cam, cv::Point2d& pixel) const {
    cv::Point2d ideal, distorted;
    if (!projectIdeal(point_cam, ideal) || !distort(ideal, distorted)) return false;
    pixel.x = config_.fx * distorted.x + config_.cx;
    pixel.y = config_.fy * distorted.y + config_.cy;
    return std::isfinite(pixel.x) && std::isfinite(pixel.y);
  }

  bool pixelToRay(const cv::Point2d& pixel, cv::Vec3d& ray) const {
    cv::Point2d distorted((pixel.x - config_.cx) / config_.fx,
                          (pixel.y - config_.cy) / config_.fy);
    cv::Point2d ideal;
    return undistort(distorted, ideal) && liftIdeal(ideal, ray);
  }

  bool pixelToNormalized(const cv::Point2d& pixel, cv::Point2d& normalized) const {
    cv::Vec3d ray;
    if (!pixelToRay(pixel, ray) || std::fabs(ray[2]) < 1e-12 || ray[2] <= 0.0) return false;
    normalized.x = ray[0] / ray[2];
    normalized.y = ray[1] / ray[2];
    return std::isfinite(normalized.x) && std::isfinite(normalized.y);
  }

 protected:
  virtual bool projectIdeal(const cv::Point3d& point_cam, cv::Point2d& ideal) const = 0;
  virtual bool liftIdeal(const cv::Point2d& ideal, cv::Vec3d& ray) const = 0;

  bool distort(const cv::Point2d& ideal, cv::Point2d& distorted) const {
    if (config_.distortion_model == "none") {
      distorted = ideal;
      return true;
    }
    if (config_.distortion_model == "radtan") return distortRadtan(ideal, distorted);
    if (config_.distortion_model == "equidist") return distortEquidist(ideal, distorted);
    return false;
  }

  bool undistort(const cv::Point2d& distorted, cv::Point2d& ideal) const {
    if (config_.distortion_model == "none") {
      ideal = distorted;
      return true;
    }
    if (config_.distortion_model == "radtan") return undistortRadtan(distorted, ideal);
    if (config_.distortion_model == "equidist") return undistortEquidist(distorted, ideal);
    return false;
  }

  bool distortRadtan(const cv::Point2d& ideal, cv::Point2d& distorted) const {
    const double x = ideal.x, y = ideal.y;
    const double x2 = x * x, y2 = y * y, xy = x * y;
    const double r2 = x2 + y2, r4 = r2 * r2, r6 = r4 * r2;
    const double radial = 1.0 + config_.k1 * r2 + config_.k2 * r4 + config_.k3 * r6;
    distorted.x = x * radial + 2.0 * config_.p1 * xy + config_.p2 * (r2 + 2.0 * x2);
    distorted.y = y * radial + config_.p1 * (r2 + 2.0 * y2) + 2.0 * config_.p2 * xy;
    return std::isfinite(distorted.x) && std::isfinite(distorted.y);
  }

  bool undistortRadtan(const cv::Point2d& distorted, cv::Point2d& ideal) const {
    ideal = distorted;
    cv::Point2d redistorted;
    for (int i = 0; i < 20; ++i) {
      const double x = ideal.x, y = ideal.y;
      const double x2 = x * x, y2 = y * y, xy = x * y;
      const double r2 = x2 + y2, r4 = r2 * r2, r6 = r4 * r2;
      const double radial = 1.0 + config_.k1 * r2 + config_.k2 * r4 + config_.k3 * r6;
      if (std::fabs(radial) < 1e-12) return false;
      const double dx = 2.0 * config_.p1 * xy + config_.p2 * (r2 + 2.0 * x2);
      const double dy = config_.p1 * (r2 + 2.0 * y2) + 2.0 * config_.p2 * xy;
      ideal.x = (distorted.x - dx) / radial;
      ideal.y = (distorted.y - dy) / radial;
      if (!distortRadtan(ideal, redistorted)) return false;
      if (cv::norm(redistorted - distorted) < 1e-10) return true;
    }
    return distortRadtan(ideal, redistorted) && cv::norm(redistorted - distorted) < 1e-7;
  }

  bool distortEquidist(const cv::Point2d& ideal, cv::Point2d& distorted) const {
    const double r = std::sqrt(ideal.x * ideal.x + ideal.y * ideal.y);
    if (r < 1e-12) {
      distorted = ideal;
      return true;
    }
    const double theta = std::atan(r);
    const double t2 = theta * theta, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
    const double theta_d = theta * (1.0 + config_.k1 * t2 + config_.k2 * t4 +
                                    config_.k3 * t6 + config_.k4 * t8);
    const double scale = theta_d / r;
    distorted.x = ideal.x * scale;
    distorted.y = ideal.y * scale;
    return std::isfinite(distorted.x) && std::isfinite(distorted.y);
  }

  bool undistortEquidist(const cv::Point2d& distorted, cv::Point2d& ideal) const {
    const double theta_d = std::sqrt(distorted.x * distorted.x + distorted.y * distorted.y);
    if (theta_d < 1e-12) {
      ideal = distorted;
      return true;
    }

    double theta = theta_d;
    for (int i = 0; i < 20; ++i) {
      const double t2 = theta * theta, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
      const double f = theta * (1.0 + config_.k1 * t2 + config_.k2 * t4 +
                                config_.k3 * t6 + config_.k4 * t8) - theta_d;
      const double df = 1.0 + 3.0 * config_.k1 * t2 + 5.0 * config_.k2 * t4 +
                        7.0 * config_.k3 * t6 + 9.0 * config_.k4 * t8;
      if (std::fabs(df) < 1e-12) return false;
      theta -= f / df;
      if (std::fabs(f) < 1e-12) break;
    }

    const double r = std::tan(theta);
    const double scale = r / theta_d;
    ideal.x = distorted.x * scale;
    ideal.y = distorted.y * scale;
    return std::isfinite(ideal.x) && std::isfinite(ideal.y);
  }

  CameraConfig config_;
};

class PinholeCamera : public Camera {
 public:
  using Camera::Camera;

 protected:
  bool projectIdeal(const cv::Point3d& point_cam, cv::Point2d& ideal) const override {
    if (point_cam.z <= 1e-12) return false;
    ideal.x = point_cam.x / point_cam.z;
    ideal.y = point_cam.y / point_cam.z;
    return true;
  }

  bool liftIdeal(const cv::Point2d& ideal, cv::Vec3d& ray) const override {
    ray = cv::Vec3d(ideal.x, ideal.y, 1.0);
    return true;
  }
};

class UCMCamera : public Camera {
 public:
  using Camera::Camera;

 protected:
  bool projectIdeal(const cv::Point3d& point_cam, cv::Point2d& ideal) const override {
    const double norm = std::sqrt(point_cam.x * point_cam.x + point_cam.y * point_cam.y +
                                  point_cam.z * point_cam.z);
    const double denom = point_cam.z + config_.xi * norm;
    if (denom <= 1e-12) return false;
    ideal.x = point_cam.x / denom;
    ideal.y = point_cam.y / denom;
    return true;
  }

  bool liftIdeal(const cv::Point2d& ideal, cv::Vec3d& ray) const override {
    const double r2 = ideal.x * ideal.x + ideal.y * ideal.y;
    const double inside = 1.0 + (1.0 - config_.xi * config_.xi) * r2;
    if (inside < 0.0) return false;
    const double denom = config_.xi * std::sqrt(inside) + 1.0;
    if (std::fabs(denom) < 1e-12) return false;
    ray = cv::Vec3d(ideal.x, ideal.y, (1.0 - config_.xi * config_.xi * r2) / denom);
    return std::isfinite(ray[2]);
  }
};

inline std::string cameraModelName(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  return name;
}

inline std::shared_ptr<Camera> createCamera(CameraConfig config) {
  config.camera_model = cameraModelName(config.camera_model);
  config.distortion_model = cameraModelName(config.distortion_model);

  if (config.fx == 0.0 || config.fy == 0.0) throw std::invalid_argument("Camera fx/fy must be non-zero");
  if (config.distortion_model != "none" && config.distortion_model != "radtan" &&
      config.distortion_model != "equidist") {
    throw std::invalid_argument("Unsupported distortion_model: " + config.distortion_model);
  }
  if (config.camera_model == "pinhole") return std::make_shared<PinholeCamera>(config);
  if (config.camera_model == "ucm") return std::make_shared<UCMCamera>(config);
  throw std::invalid_argument("Unsupported camera_model: " + config.camera_model);
}

#endif

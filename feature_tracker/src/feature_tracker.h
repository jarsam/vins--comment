#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    vector<cv::Point2f> undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;

    //! 这个prev_img没有看到实际作用，cur_img是上一帧，forw_img是当前帧
    cv::Mat prev_img, cur_img, forw_img;
    //! 除了KLT得到的Freatures之外，新提取的Features以满足最大特征点数目要求
    vector<cv::Point2f> n_pts;
    //! 和图像image同理,cur_pts：上一帧特征点，forw_pts是当前帧特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;

    //! 每个Feature的跟踪次数
    vector<int> ids;
    vector<int> track_cnt;
    camodocal::CameraPtr m_camera;

    static int n_id;
};

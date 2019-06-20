/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

// ----------------------------------------------------------------------------
#include "loop_closure.h"

LoopClosure::LoopClosure(const char *_voc_file, int _image_w, int _image_h)
:demo(_voc_file,_image_w, _image_h), IMAGE_W(_image_w), IMAGE_H(_image_h)
{
    printf(" loop closure init finish\n");
}

void LoopClosure::initCameraModel(const std::string &calib_file)
{
      demo.initCameraModel(calib_file);
}

/**
 * [LoopClosure::startLoopClosure 闭环检测启动程序]
 * @param  keys        [关键点]
 * @param  descriptors [描述子]
 * @param  cur_pts     [description]
 * @param  old_pts     [description]
 * @param  old_index   [description]
 * @return             [description]
 */
bool LoopClosure::startLoopClosure(std::vector<cv::KeyPoint> &keys, std::vector<BRIEF::bitset> &descriptors,
                                   std::vector<cv::Point2f> &cur_pts,
                                   std::vector<cv::Point2f> &old_pts,
                                   int &old_index)
{
  try 
  {
    bool loop_succ = false;
    loop_succ = demo.run("BRIEF", keys, descriptors, cur_pts, old_pts, old_index);
    return loop_succ;
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
    return false;
  }
}

void LoopClosure::eraseIndex(std::vector<int> &erase_index)
{
  demo.eraseIndex(erase_index);
}
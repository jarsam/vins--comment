#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
    failure_occur = 0;
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    //！设置相机测量误差的信息矩阵
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();

        //! IMU初始的Bias都是设为0的
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    relocalize = false;
    retrive_data_vector.clear();
    relocalize_t = Eigen::Vector3d(0, 0, 0);
    relocalize_r = Eigen::Matrix3d::Identity();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();
}

/**
 * [Estimator::processIMU description]
 * @param dt                  [两次IMU测量的时间间隔]
 * @param linear_acceleration [description]
 * @param angular_velocity    [description]
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
//如果是第一帧imu数据
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    //! 当滑窗不满的时候，把当前测量值加入到滑窗指定位置，所以在这个阶段做预计分的时候相当于是在和自己做预计分
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    //! 进入预计分环节            崔华坤 imu 部分第一步
    if (frame_count != 0)
    {
        
        //! 添加IMU数据，进行预积分  Prei-Second  //IntegrationBase是一个类型，定义在factor/integration_bash.h中
//pre_integrations[WINDOW_SIZE+1]是针对一个滑动窗口的,其中的每个元素保存的都是两帧之间IMU的预积分数据,       //对于每一个IMU数据,要做的事情是:1.更新预积分测量值2.更新预积分协方差矩阵
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

        //if(solver_flag != NON_LINEAR)
//为imu和视觉对齐做准备
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
//dt等数据通过estimate.node的send_imu得到timestamp，缓冲的数据保存是为了slidewindow的计算
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        //! 这个地方又拿IMU的数据做了一次积分, Prei-Third，更新估计器内系统的位姿  中值法离散形式j=公式中的i
 //提供imu计算的当前位置，速度，作为优化的初值    这里是实际的，上一个预积分的是增量
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * [Estimator::processImage 特征点处理程序]
 * @param image  [特征点]
 * @param header [帧头]
 */
void Estimator::processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    //! 基于视差来选择关键帧(经过旋转补偿)
    //! 向Featuresmanger中添加Features并确定共视关系及视差角的大小
    //! 以此来选择边缘化的方式
    if (f_manager.addFeatureCheckParallax(frame_count, image))
        marginalization_flag = MARGIN_OLD;              // Keyframe
    else
        marginalization_flag = MARGIN_SECOND_NEW;       //Non-Keyframe

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    //! 分别将读到的features和imu_mea加入到各自的序列当中。
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //! 把这部分代码放到后面的初始化的代码中，是不是更合理一些
//参数要是设置imu-camera的外参数未知，也可以帮你求解的
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            //! 选取两帧之间共有的Features
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            //! 校准相机与IMU之间的旋转
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    //! 初始化
    if (solver_flag == INITIAL)
    {
        //！滑窗中的Keyframe达到指定大小的时候，才开始优化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                //！相机初始化//构造sfm，优化imu偏差，加速度g，尺度的确定
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else//先凑够window-size的数量的Frame
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //! Step1：通过计算预积分加速度的标准差，检测IMU的可观性
    //check imu observibility
    {
        //! 计算均值
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        //! 计算方差
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        //！计算标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        //！以标准差判断可观性
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }

    //！Step2：滑窗内全局的SFM
    // global sfm
    //！存入 global sfm用到的Features
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;

    //! Step2.1:遍历滑窗内所有的Features，以vector<SFMFeature>, SFMFeature: <vector<pair<int,Vector2d>>形式保存滑窗内所有特征点
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int l;

    //! Step2.2 求取滑窗内与当前帧共视关系较强的关键帧，来三角化滑窗内的3D点
    //! T:  当前帧到共视帧  now ===> l
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    GlobalSFM sfm;
    //! Step2.3 三角化恢复滑窗内的Features
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        //！如果帧头和滑窗内关键帧的帧头相同，则直接读取位姿即可
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            //！一次性转换到IMU坐标系下
            frame_it->second.is_key_frame = true;

            //! Question: 这里为什么要对R_{bc}做转置
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }

        //! 将滑窗内第i帧的变换矩阵当做初始值
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        //! 如果这部分位姿不作为关键帧的化，求解出来就没有意义啊
        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;

        //! 遍历该帧的所有Features
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            //cout << "feature id " << feature_id;
            for (auto &i_p : id_pts.second)
            {
                //cout << " pts image_frame " << (i_p.second.head<2>() * 460 ).transpose() << endl;
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            //cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        //! PnP求解出的位姿要取逆
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        //！转换到IMU坐标系下
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

/**
 * [Estimator::visualInitialAlign 视觉与IMU的对齐]
 * @return [description]
 *
 * 1.修正陀螺仪的Bias
 * 2.求取滑窗内每一帧IMU对应的速度、重力向量以及尺度因子
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;

    //! Step1 视觉与IMU对齐
    //! 要注意这个地方求解出的g是在C0坐标系下
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

/********后面这部分代码需要重新注释**********************************/    

    //！替换滑窗内相机位姿的旧值，并将滑窗内的视觉帧全部设为关键帧
    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    //! Step2: 将滑窗内Features的逆深度都设为1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //！Step3：根据滑窗内关键帧位姿，三角化特征点，得到不具有尺度的深度值
    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    //! Step4: 根据初始化得到的陀螺仪Bias，在滑窗内重新做一次预积分。
    //！question：在得到Bias之后就做了一次，为什么这个地方还要做一次，上一次是所有帧
    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }

    //! Step5: 求滑窗内其他Frame以第一帧为参考系下的坐标
    //! Question: 这个地方的计算是怎么做的
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;

    //! Step6: 得到滑窗内关键帧的速度
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    //! Step7: 为滑窗内特征点的深度值加上尺度
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    //! Step8: 得到真实的pitch和roll，纠正pitch和roll
    //! 得到wqc0，并且使yaw轴为0
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    //! 将重力g转到后面的世界坐标系下
    g = R0 * g;

    //! 同理，将滑窗内的状态量转到世界坐标系下
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/**
 * [Estimator::relativePose 在滑窗中寻找与最新的关键帧共视关系较强的关键帧]
 * @param  relative_R [description]
 * @param  relative_T [description]  window_size ==> i
 * @param  l          [description]
 * @return            [description]
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    //！在滑窗内寻找与最新的关键帧共视点超过20(像素点)的关键帧
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        //! 共视的Features应该大于20
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;

            //! 求取匹配的特征点在图像上的视差和(归一化平面上)
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            //! 求取所有匹配的特征点的平均视差
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //! 视差大于一定阈值，并且能够有效地求解出变换矩阵
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

/**
 * [Estimator::solveOdometry 进入SLAM的后端优化阶段]
 */
void Estimator::solveOdometry()
{
    //！如果滑窗内视觉帧的个数小于设定大小，则返回
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        //！三角化还没有得到深度的Features
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        //！进入优化阶段
        optimization();
    }
}

/**
 * [Estimator::vector2double 将Stl Vector 转为Ceres可以处理的数组]
 */
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
}

/**
 * [Estimator::double2vector 提取滑窗内状态量并计算闭环帧的漂移]
 */
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    //！这个位子是没有经过优化的滑窗第一帧位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());

    //！求取yaw的漂移，那么这个地方的漂移来自什么地方呢？
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    //！提取滑窗内的所有状态量
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    //！
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (LOOP_CLOSURE && relocalize && retrive_data_vector[0].relative_pose && !retrive_data_vector[0].relocalized)
    {
        for (int i = 0; i < (int)retrive_data_vector.size();i++)
            retrive_data_vector[i].relocalized = true;
        Matrix3d vio_loop_r;
        Vector3d vio_loop_t;
        //！计算发生yaw角漂移之后的闭环帧位姿
        vio_loop_r = rot_diff * Quaterniond(retrive_data_vector[0].loop_pose[6], retrive_data_vector[0].loop_pose[3], retrive_data_vector[0].loop_pose[4], retrive_data_vector[0].loop_pose[5]).normalized().toRotationMatrix();
        vio_loop_t = rot_diff * Vector3d(retrive_data_vector[0].loop_pose[0] - para_Pose[0][0],
                                retrive_data_vector[0].loop_pose[1] - para_Pose[0][1],
                                retrive_data_vector[0].loop_pose[2] - para_Pose[0][2]) + origin_P0;

        //！漂移之后的位姿差(平移量和yaw)
        Quaterniond vio_loop_q(vio_loop_r);
        double relocalize_yaw;
        relocalize_yaw = Utility::R2ypr(retrive_data_vector[0].R_old).x() - Utility::R2ypr(vio_loop_r).x();
        relocalize_r = Utility::ypr2R(Vector3d(relocalize_yaw, 0, 0));
        relocalize_t = retrive_data_vector[0].P_old- relocalize_r * vio_loop_t;
    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        return true;
    }
    return false;
}

/**
 * [Estimator::optimization 后端的优化]
 */
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    //！设置柯西损失函数因子
    loss_function = new ceres::CauchyLoss(1.0);

    //！Step1:添加待优化状态量
    //！Step1.1：添加[p,q](7)，[speed,ba,bg](9)
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    //！Step1.2：添加相机与IMU的外参[p_cb,q_cb](7)
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param"); 
    }

    //！将优化量存入数组为ceres参数赋予初值
    TicToc t_whole, t_prepare;
    vector2double();

    //！Step2.1:添加边缘化的残差
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    //！Step2.2:添加IMU的residual
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        //！添加代价函数
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        //！注意在添加残差的组成部分，由前后两帧的[p,q,v,b]组成，在计算雅克比的时候[p,q](7),[v,b](9)分开计算
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    //！Step2.3:添加视觉的residual
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        //！得到观测到该特征点的首帧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        //！得到首帧观测到的特征点
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            //！得到第二个特征点
            Vector3d pts_j = it_per_frame.point;
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            f_m_cnt++;
        }
    }
    relocalize = false;

    //！Step2.3:添加视觉的residual
    //！添加闭环校正时候的状态量和残差
    //loop close factor
    if(LOOP_CLOSURE)
    {
        int loop_constraint_num = 0;
        //！遍历闭环检测 database
        for (int k = 0; k < (int)retrive_data_vector.size(); k++)
        {    
            //！遍历滑窗内的Keyframe
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                //！为什么这样就闭环成功了呢？这个地方通过时间戳就闭环了？
                //！建立闭环约束
                if(retrive_data_vector[k].header == Headers[i].stamp.toSec())
                {
                    relocalize = true;
                    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
                    problem.AddParameterBlock(retrive_data_vector[k].loop_pose, SIZE_POSE, local_parameterization);
                    loop_window_index = i;
                    loop_constraint_num++;
                    int retrive_feature_index = 0;
                    int feature_index = -1;

                    //！遍历滑窗内的特征点
                    for (auto &it_per_id : f_manager.feature)
                    {
                        it_per_id.used_num = it_per_id.feature_per_frame.size();
                        //！至少有两帧图像观测到该特征点且不是滑窗内的最后两帧
                        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                            continue;

                        ++feature_index;
                        int start = it_per_id.start_frame;
                        //！如果该Feature是被滑窗中本帧之前观测到
                        if(start <= i)
                        {   
                            while(retrive_data_vector[k].features_ids[retrive_feature_index] < it_per_id.feature_id)
                            {
                                retrive_feature_index++;
                            }

                            //！将拥有固定位姿的闭环帧加入到Visual-Inertail BA中
                            if(retrive_data_vector[k].features_ids[retrive_feature_index] == it_per_id.feature_id)
                            {
                                Vector3d pts_j = Vector3d(retrive_data_vector[k].measurements[retrive_feature_index].x, retrive_data_vector[k].measurements[retrive_feature_index].y, 1.0);
                                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                                
                                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                                problem.AddResidualBlock(f, loss_function, para_Pose[start], retrive_data_vector[k].loop_pose, para_Ex_Pose[0], para_Feature[feature_index]);
                            
                                retrive_feature_index++;
                            }     
                        }
                    }
                            
                }
            }
        }
        ROS_DEBUG("loop constraint num: %d", loop_constraint_num);
    }
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    //！设置求解器的属性
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    //！求解problem
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    /*****************优化后的内容********************/
    //！求解两个闭环帧之间的关系
    // relative info between two loop frame
    if(LOOP_CLOSURE && relocalize)
    { 
        for (int k = 0; k < (int)retrive_data_vector.size(); k++)
        {
            for(int i = 0; i< WINDOW_SIZE; i++)
            {
                //！闭环检测成功
                if(retrive_data_vector[k].header == Headers[i].stamp.toSec())
                {
                    retrive_data_vector[k].relative_pose = true;
                    Matrix3d Rs_i = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
                    Vector3d Ps_i = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);

                    //！闭环帧的位姿
                    Quaterniond Qs_loop;
                    Qs_loop = Quaterniond(retrive_data_vector[k].loop_pose[6],  retrive_data_vector[k].loop_pose[3],  retrive_data_vector[k].loop_pose[4],  retrive_data_vector[k].loop_pose[5]).normalized().toRotationMatrix();
                    Matrix3d Rs_loop = Qs_loop.toRotationMatrix();
                    Vector3d Ps_loop = Vector3d( retrive_data_vector[k].loop_pose[0],  retrive_data_vector[k].loop_pose[1],  retrive_data_vector[k].loop_pose[2]);

                    //！求匹配帧到闭环帧之间的相对位姿
                    retrive_data_vector[k].relative_t = Rs_loop.transpose() * (Ps_i - Ps_loop);
                    retrive_data_vector[k].relative_q = Rs_loop.transpose() * Rs_i;
                    //！因为pitch和roll可观，所以仅考虑在yaw上的漂移
                    retrive_data_vector[k].relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs_i).x() - Utility::R2ypr(Rs_loop).x());
                    //！
                    if (abs(retrive_data_vector[k].relative_yaw) > 30.0 || retrive_data_vector[k].relative_t.norm() > 20.0)
                        retrive_data_vector[k].relative_pose = false;
                        
                }
            } 
        } 
    }

    double2vector();

    //！Step3：构造边缘化的Problem
    //！边缘化旧帧
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        //! 先验误差会一直保存，而不是只使用一次
        //! 如果上一次边缘化的信息存在
        //! 要边缘化的参数块是 para_Pose[0] para_SpeedBias[0] 以及 para_Feature[feature_index](滑窗内的第feature_index个点的逆深度)
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                //！查询待估计参数中是首帧状态量的序号
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }

            //! 构造边缘化的的Factor
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            //! 添加上一次边缘化的参数块
            //！cost_function, loss_function, 待估计参数(last_marginalization_parameter_blocks, drop_set)
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //！添加IMU的先验，只包含旧帧的IMU测量残差
        //！Question：不应该是pre_integrations[0]么
        //!
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //！添加视觉的先验，只添加起始帧是旧帧且观测次数大于2的Features
        {
            int feature_index = -1;
            //! 遍历滑窗内所有的Features
            for (auto &it_per_id : f_manager.feature)
            {
                //! 该特征点被观测到的次数
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                //! Feature的观测次数不小于2次，且起始帧不属于最后两帧
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
               
                //! 只选择起始帧为旧帧的Features
                //! 这里不对啊，论文中讲的是只要Features被旧帧观测了就要被marg掉的啊
                //! 这里是合适的，因为只要

                if (imu_i != 0)
                    continue;

                //! 得到该Feature在起始下的归一化坐标
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;

                    //! 不需要起始观测帧
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                   vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                   vector<int>{0, 3});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        //! 将三个ResidualBlockInfo中的参数块综合到marginalization_info中
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        //!
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //！将滑窗里关键帧位姿移位，为什么是向右移位了呢？
        //! 这里是保存了所有状态量的信息，为什么没有保存逆深度的状态量呢
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }

    //！边缘化倒数第二帧
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    //！寻找导数第二帧的位姿
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}
//WINDOW_SIZE中的参数的之间调整，同时FeatureManager进行管理feature，有些点要删除掉，有些点的深度要在下一frame表示(start frame已经删除了)
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                double t_0 = Headers[0].stamp.toSec();
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();

}


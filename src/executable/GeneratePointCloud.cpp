#include "Parameter.h"
#include "SpherePlanning.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "cnpy/cnpy.h"

int generate_point_cloud(const std::string &brick_config_path, const std::string &point_cloud_path) {
    cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);

    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::PROJ;
    param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);
    SpherePlanning instance(param);

    const unsigned int N = 20000;
    Eigen::Matrix3Xd points(3, N);
    Eigen::Vector3d point;

    for (unsigned int i = 0; i < N; i++) {
        while (true) {
            point.setRandom();
            point /= point.norm();
            if (!instance.isValid(point)) {
                break;
            }
        }
        points.col(i) = point;

        std::cout << "\rProgress: " << i + 1 << " / " << N << std::flush;
    }
    std::cout << std::endl;

    cnpy::npy_save(point_cloud_path, points.data(), {N, 3}, "w");
    return 0;
}

int generate_old_point_cloud(const std::string &point_cloud_path) {
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::PROJ;
    param.is_brick_env = false;
    SpherePlanning instance(param);

    const unsigned int N = 20000;
    Eigen::Matrix3Xd points(3, N);
    Eigen::Vector3d point;

    for (unsigned int i = 0; i < N; i++) {
        while (true) {
            point.setRandom();
            point /= point.norm();
            if (!instance.isValid(point)) {
                break;
            }
        }
        points.col(i) = point;

        std::cout << "\rProgress: " << i + 1 << " / " << N << std::flush;
    }
    std::cout << std::endl;

    cnpy::npy_save(point_cloud_path, points.data(), {N, 3}, "w");
    return 0;
}

int main(int argc, char **argv) {
    srand(0);
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);

    std::cout << "Generating point cloud for old environment " << std::endl;
    generate_old_point_cloud("./data/point_cloud/envOld.npy");

    for (unsigned int i = 0; i < 50; i++) {
        std::string brick_config_path = (boost::format("./data/brick_config/env%d.npy") % i).str();
        std::string point_cloud_path = (boost::format("./data/point_cloud/env%d.npy") % i).str();
        std::cout << "Generating point cloud for environment " << i << std::endl;
        generate_point_cloud(brick_config_path, point_cloud_path);
    }
}

#include "Parameter.h"
#include "SpherePlanning.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <ompl/util/Console.h>
#include <ostream>
#include <string>
#include <tuple>

#include "cnpy/cnpy.h"
#include "planner/MPNetPlanner.h"

int plan0(const std::string &brick_config_path, const std::string &path_path, Eigen::Vector3d start, Eigen::Vector3d goal) {
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::ATLAS;
    cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);
    param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);
    param.is_brick_env = false;

    SpherePlanning instance(param);
    if (instance.planOnce(start, goal)) {
        auto path = instance.getSmoothPath(20);
        unsigned int i = 0;
        std::string path_name = (boost::format("path_%d") % i).str();
        std::string mode = (i == 0) ? "w" : "a";
        cnpy::npz_save(path_path, path_name, path.data(), {static_cast<unsigned long>(path.cols()), 3}, mode);
    }
    return 0;
}

int plan1(const std::string &brick_config_path, const std::string &path_path, Eigen::Vector3d start, Eigen::Vector3d goal) {
    Parameter param;
    param.planner = Parameter::CoMPNet;
    param.space = Parameter::ATLAS;
    cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);
    param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);
    param.is_brick_env = true;

    SpherePlanning instance(param);
    if (instance.planOnce(start, goal)) {
        auto path = instance.getSmoothPath(20);
        unsigned int i = 1;
        std::string path_name = (boost::format("path_%d") % i).str();
        std::string mode = (i == 0) ? "w" : "a";
        // cnpy::npz_save(path_path, path_name, path.data(), {static_cast<unsigned long>(path.cols()), 3}, mode);
    }
    instance._planner->as<ompl::geometric::MPNetPlanner>()->exportSamples("./data/samples.ply");
    return 0;
}

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_DEBUG);
    Eigen::Vector3d start, goal;
    start << -0.39508, 0.86617, -0.30606;
    goal << -0.77, 0.08, 0.63;
    start /= start.norm();
    goal /= goal.norm();

    // plan0("./data/brick_config/env2.npy", "./data/test/temp.npz", start, goal);
    plan1("./data/brick_config/env0.npy", "./data/test/temp.npz", start, goal);
}

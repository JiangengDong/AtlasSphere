#include "Parameter.h"
#include "SpherePlanning.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <iostream>
#include <ompl/util/Console.h>
#include <ostream>
#include <string>

#include "cnpy/cnpy.h"

int plan(const std::string &brick_config_path, const std::string &start_path, const std::string &goal_path, const std::string &path_path) {
    cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::PROJ;
    param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);

    const unsigned int N = 20000;
    cnpy::NpyArray arr_start = cnpy::npy_load(start_path);
    Eigen::Matrix3Xd starts = Eigen::Map<Eigen::Matrix3Xd>(arr_start.data<double>(), 3, N);
    cnpy::NpyArray arr_goal = cnpy::npy_load(goal_path);
    Eigen::Matrix3Xd goals = Eigen::Map<Eigen::Matrix3Xd>(arr_goal.data<double>(), 3, N);

    unsigned fail_count = 0;
    for (unsigned int i = 0; i < N; i++) {
        SpherePlanning instance(param);
        if (instance.planOnce(starts.col(i), goals.col(i))) {
            auto path = instance.getPath();
            std::string path_name = (boost::format("path_%d") % i).str();
            std::string mode = (i == 0) ? "w" : "a";
            cnpy::npz_save(path_path, path_name, path.data(), {static_cast<unsigned long>(path.cols()), 3}, mode);
        } else {
            fail_count++;
        }
        std::cout << "\rProgress: " << i + 1 << " / " << N << ", fail: " << fail_count << std::flush;
    }
    std::cout << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    srand(0);
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    for (unsigned int i = 1; i < 10; i++) {
        std::string brick_config_path = (boost::format("./data/brick_config/env%d.npy") % i).str();
        std::string start_path = (boost::format("./data/train/env%d_start.npy") % i).str();
        std::string goal_path = (boost::format("./data/train/env%d_goal.npy") % i).str();
        std::string path_path = (boost::format("./data/train/env%d_path.npz") % i).str();
        std::cout << "Planning for environment " << i << std::endl;
        plan(brick_config_path, start_path, goal_path, path_path);
    }
}

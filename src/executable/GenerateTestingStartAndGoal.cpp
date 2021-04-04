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

int generate_start_and_goal(const std::string &brick_config_path, const std::string &start_path, const std::string &goal_path) {
    cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);

    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::PROJ;
    param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);
    SpherePlanning instance(param);

    const unsigned int N = 100;
    Eigen::Matrix3Xd starts(3, N), goals(3, N);
    Eigen::Vector3d start, goal;

    for (unsigned int i = 0; i < N; i++) {
        while (true) {
            start.setRandom();
            start /= start.norm();
            if (instance.isValid(start)) {
                break;
            }
        }
        starts.col(i) = start;
        while (true) {
            goal.setRandom();
            goal /= goal.norm();
            if (instance.isValid(goal)) {
                break;
            }
        }
        goals.col(i) = goal;

        std::cout << "\rProgress: " << i + 1 << " / " << N << std::flush;
    }
    std::cout << std::endl;

    cnpy::npy_save(start_path, starts.data(), {N, 3}, "w");
    cnpy::npy_save(goal_path, goals.data(), {N, 3}, "w");
    return 0;
}

int generate_start_and_goal_old(const std::string &start_path, const std::string &goal_path) {
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::PROJ;
    param.is_brick_env = false;
    SpherePlanning instance(param);

    const unsigned int N = 100;
    Eigen::Matrix3Xd starts(3, N), goals(3, N);
    Eigen::Vector3d start, goal;

    for (unsigned int i = 0; i < N; i++) {
        while (true) {
            start.setRandom();
            start /= start.norm();
            if (instance.isValid(start)) {
                break;
            }
        }
        starts.col(i) = start;
        while (true) {
            goal.setRandom();
            goal /= goal.norm();
            if (instance.isValid(goal)) {
                break;
            }
        }
        goals.col(i) = goal;

        std::cout << "\rProgress: " << i + 1 << " / " << N << std::flush;
    }
    std::cout << std::endl;

    cnpy::npy_save(start_path, starts.data(), {N, 3}, "w");
    cnpy::npy_save(goal_path, goals.data(), {N, 3}, "w");
    return 0;
}

int main(int argc, char **argv) {
    srand(0);
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    generate_start_and_goal_old("./data/test/envOld_start.npy", "./data/test/envOld_goal.npy");
    // for (unsigned int i = 0; i < 20; i++) {
    //     std::string brick_config_path = (boost::format("./data/brick_config/env%d.npy") % i).str();
    //     std::string start_path = (boost::format("./data/test/env%d_start.npy") % i).str();
    //     std::string goal_path = (boost::format("./data/test/env%d_goal.npy") % i).str();
    //     std::cout << "Generating starts and goals for environment " << i << std::endl;
    //     generate_start_and_goal(brick_config_path, start_path, goal_path);
    // }
}

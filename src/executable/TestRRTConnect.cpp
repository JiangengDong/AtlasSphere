#include "Parameter.h"
#include "SpherePlanning.h"
#include <ATen/Functions.h>
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

using PlanResult = std::tuple<unsigned int, unsigned int, double, double>;

PlanResult plan(const Parameter &param, const std::string &start_path, const std::string &goal_path, const std::string &path_path) {
    const unsigned int N = 100;
    cnpy::NpyArray arr_start = cnpy::npy_load(start_path);
    Eigen::Matrix3Xd starts = Eigen::Map<Eigen::Matrix3Xd>(arr_start.data<double>(), 3, N);
    cnpy::NpyArray arr_goal = cnpy::npy_load(goal_path);
    Eigen::Matrix3Xd goals = Eigen::Map<Eigen::Matrix3Xd>(arr_goal.data<double>(), 3, N);

    unsigned int fail_count = 0;
    Eigen::VectorXd time(N);
    Eigen::VectorXd length(N);
    double total_time = 0.0;
    double total_length = 0.0;

    for (unsigned int i = 0; i < N; i++) {
        SpherePlanning instance(param);
        if (instance.planOnce(starts.col(i), goals.col(i))) {
            auto path = instance.getPath();
            std::string path_name = (boost::format("path_%d") % i).str();
            std::string mode = (i == 0) ? "w" : "a";
            cnpy::npz_save(path_path, path_name, path.data(), {static_cast<unsigned long>(path.cols()), 3}, mode);
            time(i) = instance.getTime();
            length(i) = instance.getPathLength();
            total_time += instance.getTime();
            total_length += instance.getPathLength();
        } else {
            fail_count++;
            time(i) = std::numeric_limits<double>::infinity();
            length(i) = std::numeric_limits<double>::infinity();
        }
        std::cout << "\rProgress: " << i + 1 << " / " << N << ", fail: " << fail_count << std::flush;
    }
    std::cout << std::endl;

    cnpy::npz_save(path_path, "time", time.data(), {N}, "a");
    cnpy::npz_save(path_path, "length", length.data(), {N}, "a");

    return std::make_tuple(N, N - fail_count, total_time, total_length);
}

int main(int argc, char **argv) {
    srand(0);
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);

    // container for data
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::ATLAS;

    // old environment test
    std::cout << "Planning for old environment " << std::endl;
    std::string start_path = "./data/test/envOld_start.npy";
    std::string goal_path = "./data/test/envOld_goal.npy";
    std::string path_path = "./data/test/envOld_path.npz";
    // load env
    param.is_brick_env = false;
    // plan and print result
    const auto &[N_, success_count_, total_time_, total_length_] = plan(param, start_path, goal_path, path_path);
    std::cout << "Success rate: " << success_count_ * 1.0 / N_ << std::endl;
    std::cout << "Average time: " << total_time_ / success_count_ << std::endl;
    std::cout << "Average len : " << total_length_ / success_count_ << std::endl;

    // new environment test
    unsigned int N = 0, success_count = 0;
    double total_time = 0.0, total_length = 0.0;
    for (unsigned int i = 0; i < 50; i++) {
        std::cout << "Planning for environment " << i << std::endl;
        std::string brick_config_path = (boost::format("./data/brick_config/env%d.npy") % i).str();
        std::string start_path = (boost::format("./data/test/env%d_start.npy") % i).str();
        std::string goal_path = (boost::format("./data/test/env%d_goal.npy") % i).str();
        std::string path_path = (boost::format("./data/test/env%d_path.npz") % i).str();

        // load env. Use braces to limit the lifetime of arr
        {
            cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);
            param.is_brick_env = true;
            param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);
        }
        // plan and print result
        const auto &[N_, success_count_, total_time_, total_length_] = plan(param, start_path, goal_path, path_path);

        N += N_;
        success_count += success_count_;
        total_time += total_time_;
        total_length += total_length_;
    }

    std::cout << std::endl
              << "Total result: " << std::endl;
    std::cout << "Success rate: " << (N - success_count) * 1.0 / N << std::endl;
    std::cout << "Average time: " << total_time / (N - success_count) << std::endl;
    std::cout << "Average len : " << total_length / (N - success_count) << std::endl;
}

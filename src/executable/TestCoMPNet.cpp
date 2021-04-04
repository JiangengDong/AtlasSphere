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

int plan_old(const std::string &start_path, const std::string &goal_path, const std::string &path_path) {
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::ATLAS;
    param.is_brick_env = false;

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
    std::cout << "Success rate: " << (N - fail_count) * 1.0 / N << std::endl;
    std::cout << "Average time: " << total_time / (N - fail_count) << std::endl;
    std::cout << "Average len : " << total_length / (N - fail_count) << std::endl;

    return 0;
}

using PlanResult = std::tuple<unsigned int, unsigned int, double, double>;
PlanResult plan(const std::string &brick_config_path, const std::string &start_path, const std::string &goal_path, const std::string &path_path) {
    Parameter param;
    param.planner = Parameter::RRTConnect;
    param.space = Parameter::ATLAS;
    cnpy::NpyArray arr = cnpy::npy_load(brick_config_path);
    param.brick_configs = Eigen::Map<Eigen::Matrix2Xd>(arr.data<double>(), 2, 500);

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
    std::cout << "Success rate: " << (N - fail_count) * 1.0 / N << std::endl;
    std::cout << "Average time: " << total_time / (N - fail_count) << std::endl;
    std::cout << "Average len : " << total_length / (N - fail_count) << std::endl;

    return std::make_tuple(N, fail_count, total_time, total_length);
}

int main(int argc, char **argv) {
    srand(0);
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    plan_old("./data/test/envOld_start.npy", "./data/test/envOld_goal.npy", "./data/test/envOld_path.npz");
    unsigned int N = 0, fail_count = 0;
    double total_time = 0.0, total_length = 0.0;
    for (unsigned int i = 0; i < 10; i++) {
        std::string brick_config_path = (boost::format("./data/brick_config/env%d.npy") % i).str();
        std::string start_path = (boost::format("./data/test/env%d_start.npy") % i).str();
        std::string goal_path = (boost::format("./data/test/env%d_goal.npy") % i).str();
        std::string path_path = (boost::format("./data/test/env%d_path.npz") % i).str();
        std::cout << "Planning for environment " << i << std::endl;
        auto result = plan(brick_config_path, start_path, goal_path, path_path);
        N += std::get<0>(result);
        fail_count += std::get<1>(result);
        total_time += std::get<2>(result);
        total_length += std::get<3>(result);
    }

    std::cout << std::endl
              << "Total result: " << std::endl;
    std::cout << "Success rate: " << (N - fail_count) * 1.0 / N << std::endl;
    std::cout << "Average time: " << total_time / (N - fail_count) << std::endl;
    std::cout << "Average len : " << total_length / (N - fail_count) << std::endl;
}

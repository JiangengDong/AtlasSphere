#include "Parameter.h"
#include "SpherePlanning.h"
#include "MPNetPlanner.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

Eigen::MatrixXd read_dataset(const std::string& filename) {
    std::ifstream input_file(filename);
    Eigen::MatrixXd m(1000, 3);
    for (size_t i = 0; i < 1000; i++) {
        input_file >> m(i, 0) >> m(i, 1) >> m(i, 2);
    }
    input_file.close();
    return m;
}

int main(int argc, char **argv) {
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    Parameter param;
    param.planner = Parameter::RRTstar;
    param.space = Parameter::ATLAS;

    srand(0);
    SpherePlanning instance(param);
    Eigen::Vector3d start, goal;
    Eigen::MatrixXd starts = read_dataset("./data/training_samples/starts.txt");
    Eigen::MatrixXd goals = read_dataset("./data/training_samples/goals.txt");

    std::ofstream len_file("./data/result/test.csv");
    for (size_t iter=0; iter<50; iter++){
        // sample start and goal
        instance.setStartGoal(starts.row(iter), goals.row(iter));

        // repeat 2 sec planning for 10 times
        ompl::base::PlannerStatus status;
        double total_time=0;
        double length = std::numeric_limits<double>::infinity();
        std::cout << "experiment " << iter << std::endl;
        for (size_t i = 0; i< 10; i++){
            status = instance._simple_setup->solve(2);
            total_time += instance._simple_setup->getLastPlanComputationTime();
            if (status == ompl::base::PlannerStatus::EXACT_SOLUTION) {
                length = instance._simple_setup->getSolutionPath().length();
            }
            std::cout << "\ttime: " << total_time << " length: " << length << std::endl;
            len_file << (status == ompl::base::PlannerStatus::EXACT_SOLUTION?total_time:std::numeric_limits<double>::infinity()) << ", " << length;
        }
        len_file << std::endl;
        instance.clear();
    }
    len_file.close();
}

#include "Parameter.h"
#include "SpherePlanning.h"
#include "MPNetPlanner.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    // get config
    // Parameter param;
    // param.planner = Parameter::CoMPNet;
    // param.space = Parameter::ATLAS;
    // param.input_dir = "/workspaces/AtlasSphere/data/pytorch_models";
    // param.output_dir = "/workspaces/AtlasSphere/data/result";

    // SpherePlanning instance(param);
    // Eigen::Vector3d start, goal;
    // start << sqrt(3.0/2)/2, -sqrt(3.0/2)/2, 1.0/2;
    // goal << sqrt(3.0/2)/2, -sqrt(3.0/2)/2, -1.0/2;

    // instance.setStartGoal(start, goal);
    // instance._planner->as<ompl::geometric::MPNetPlanner>()->use_mpnet = true;
    // for (unsigned int i=0; i<40; i++){
    //     instance._simple_setup->solve(10);
    //     instance._simple_setup->clear();
    // }
    // // instance._planner->as<ompl::geometric::MPNetPlanner>()->exportSamples("/workspaces/AtlasSphere/data/result/visual/uniform.ply");
    // instance._planner->as<ompl::geometric::MPNetPlanner>()->exportSamples("/workspaces/AtlasSphere/data/result/visual/mpnet.ply");

    // // get config
    // ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    // Parameter param;
    // param.planner = Parameter::BITstar;
    // param.space = Parameter::ATLAS;
    // param.input_dir = "/workspaces/AtlasSphere/data/pytorch_models";
    // param.output_dir = "/workspaces/AtlasSphere/data/result";

    // srand(0);
    // SpherePlanning instance(param);
    // Eigen::Vector3d start, goal;
    // start << sqrt(3.0/2)/2, -sqrt(3.0/2)/2, 1.0/2;
    // goal << sqrt(3.0/2)/2, -sqrt(3.0/2)/2, -1.0/2;

    // std::ofstream len_file("/workspaces/AtlasSphere/data/result/visual/bitstar-mpnet.csv");
    // for (size_t iter=0; iter<10; iter++){
    //     instance.setStartGoal(start, goal);
    //     ompl::base::PlannerStatus status;
    //     double total_time=0;
    //     double length = std::numeric_limits<double>::infinity();
    //     for (size_t i = 0; i< 20; i++){
    //         status = instance._simple_setup->solve(2);
    //         total_time += instance._simple_setup->getLastPlanComputationTime();
    //         if (status == ompl::base::PlannerStatus::EXACT_SOLUTION) {
    //             length = instance._simple_setup->getSolutionPath().length();
    //         }
    //         std::cout << "time: " << total_time << " length: " << length << std::endl;
    //         len_file << length << ", ";
    //     }
    //     len_file << std::endl;
    //     instance.clear();
    // }
    // len_file.close();

    ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
    Parameter param;
    param.planner = Parameter::FMT;
    param.space = Parameter::ATLAS;
    param.input_dir = "/workspaces/AtlasSphere/data/pytorch_models";
    param.output_dir = "/workspaces/AtlasSphere/data/result";

    srand(0);
    SpherePlanning instance(param);
    Eigen::Vector3d start, goal;

    std::ofstream len_file("/workspaces/AtlasSphere/data/result/visual/fmt-compnet-time.csv");
    for (size_t iter=0; iter<50; iter++){
        do{
            start = Eigen::Vector3d::Random();
            start /= start.norm();
        }while(!instance.isValid(start));
        do {
            goal = Eigen::Vector3d::Random();
            goal /= goal.norm();
        }while(!instance.isValid(goal));
        instance.setStartGoal(start, goal);
        ompl::base::PlannerStatus status;
        double total_time=0;
        double length = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i< 1; i++){
            status = instance._simple_setup->solve(20);
            total_time += instance._simple_setup->getLastPlanComputationTime();
            if (status == ompl::base::PlannerStatus::EXACT_SOLUTION) {
                length = instance._simple_setup->getSolutionPath().length();
            }
            std::cout << "time: " << total_time << " length: " << length << std::endl;
            len_file << (status == ompl::base::PlannerStatus::EXACT_SOLUTION?total_time:std::numeric_limits<double>::infinity()) << ", " << length;
        }
        len_file << std::endl;
        instance.clear();
    }
    len_file.close();
}
